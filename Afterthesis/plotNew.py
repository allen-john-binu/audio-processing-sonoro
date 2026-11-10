
#!/usr/bin/env python3
"""Interactive DOA heatmap with SPL filtering and timestep inspection.

Latest CSV format:
- elapsed_seconds
- spl_db
- peak1_angle_deg, peak2_angle_deg
- response_000deg through response_357deg (3-degree spacing)

Click a heatmap column to inspect the full spatial response for that timestep.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider


RESPONSE_PATTERN = re.compile(r"^response_(\d{3})deg$")


def load_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("The CSV contains no data rows.")

    if "spl_db" not in df.columns:
        raise ValueError("CSV must contain a 'spl_db' column.")

    spl = pd.to_numeric(
        df["spl_db"], errors="coerce"
    ).to_numpy(dtype=float)

    if "elapsed_seconds" in df.columns:
        times = pd.to_numeric(
            df["elapsed_seconds"], errors="coerce"
        ).to_numpy(dtype=float)
        time_label = "Elapsed time (s)"

    elif "timestamp_utc" in df.columns:
        timestamps = pd.to_datetime(
            df["timestamp_utc"], utc=True, errors="coerce"
        )
        times = (
            timestamps - timestamps.iloc[0]
        ).dt.total_seconds().to_numpy(dtype=float)
        time_label = "Elapsed time (s)"

    elif "timestamp" in df.columns:
        times = pd.to_numeric(
            df["timestamp"], errors="coerce"
        ).to_numpy(dtype=float)

        finite = np.isfinite(times)
        if finite.any():
            times = times - times[np.flatnonzero(finite)[0]]

        time_label = "Elapsed time (s)"

    else:
        raise ValueError(
            "CSV must contain elapsed_seconds, timestamp_utc, or timestamp."
        )

    # Identify response columns from their names.
    angle_columns = []

    for column in df.columns:
        match = RESPONSE_PATTERN.match(str(column))
        if match:
            angle = int(match.group(1))
            if 0 <= angle < 360:
                angle_columns.append((angle, column))

    angle_columns.sort(key=lambda item: item[0])

    expected_angles = np.arange(0, 360, 3, dtype=float)
    angles = np.array(
        [angle for angle, _ in angle_columns], dtype=float
    )

    if len(angle_columns) != 120 or not np.array_equal(
        angles, expected_angles
    ):
        raise ValueError(
            "Expected 120 spatial-response columns: "
            "response_000deg, response_003deg, ..., response_357deg."
        )

    response = df[
        [column for _, column in angle_columns]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Use peaks recorded by the acquisition script when available.
    peak1 = (
        pd.to_numeric(
            df["peak1_angle_deg"], errors="coerce"
        ).to_numpy(dtype=float)
        if "peak1_angle_deg" in df.columns
        else np.full(len(df), np.nan)
    )

    peak2 = (
        pd.to_numeric(
            df["peak2_angle_deg"], errors="coerce"
        ).to_numpy(dtype=float)
        if "peak2_angle_deg" in df.columns
        else np.full(len(df), np.nan)
    )

    valid = np.isfinite(times) & np.isfinite(spl)
    times = times[valid]
    spl = spl[valid]
    response = response[valid, :]
    peak1 = peak1[valid]
    peak2 = peak2[valid]

    order = np.argsort(times)
    times = times[order]
    spl = spl[order]
    response = response[order, :]
    peak1 = peak1[order]
    peak2 = peak2[order]

    if len(times) == 0:
        raise ValueError("No rows have valid timestamps and SPL values.")

    return times, spl, angles, response, peak1, peak2, time_label


def main():
    parser = argparse.ArgumentParser(
        description="Interactive DOA heatmap and single-timestep inspection."
    )
    parser.add_argument("csv", help="Path to the DOA discovery CSV")
    parser.add_argument(
        "--min-spl",
        type=float,
        default=None,
        help="Initial minimum SPL threshold (default: minimum in CSV).",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Display spatial response as 10*log10(response).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the initial heatmap image.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)

    (
        times,
        spl,
        angles,
        response,
        peak1,
        peak2,
        time_label,
    ) = load_csv(csv_path)

    # Keep the original response for raw peak values, if needed.
    if args.db:
        with np.errstate(divide="ignore", invalid="ignore"):
            plot_response = np.where(
                response > 0,
                10.0 * np.log10(response),
                np.nan,
            )
        color_label = "Spatial response (dB, 10·log10)"
    else:
        plot_response = response.copy()
        color_label = "Spatial response"

    spl_min = float(np.min(spl))
    spl_max = float(np.max(spl))

    if spl_min == spl_max:
        spl_min -= 1.0
        spl_max += 1.0

    initial_threshold = (
        spl_min if args.min_spl is None else args.min_spl
    )
    initial_threshold = float(
        np.clip(initial_threshold, spl_min, spl_max)
    )

    # --------------------------------------------------------
    # Heatmap figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.subplots_adjust(
        left=0.10, right=0.73, top=0.91, bottom=0.25
    )

    slider_ax = fig.add_axes([0.16, 0.09, 0.66, 0.035])
    spl_slider = Slider(
        ax=slider_ax,
        label="Minimum dB SPL",
        valmin=spl_min,
        valmax=spl_max,
        valinit=initial_threshold,
        valstep=max((spl_max - spl_min) / 200.0, 0.01),
    )

    count_text = fig.text(0.5, 0.155, "", ha="center")
    colorbar = None

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="None", markersize=7,
            markerfacecolor="red", markeredgecolor="black",
            label="Strongest recorded peak",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="None", markersize=7,
            markerfacecolor="orange", markeredgecolor="black",
            label="Second-strongest recorded peak",
        ),
    ]

    # --------------------------------------------------------
    # Selected-timestep spatial response figure
    # --------------------------------------------------------

    selected_fig, selected_ax = plt.subplots(figsize=(10, 5))
    selected_fig.subplots_adjust(bottom=0.16)
    selected_fig.canvas.manager.set_window_title(
        "Selected timestep — full spatial response"
    )

    selected_line, = selected_ax.plot(
        [], [], "-o", markersize=3, linewidth=1
    )

    selected_peak1 = selected_ax.scatter(
        [], [], c="red", s=65, edgecolors="black",
        label="Strongest recorded peak", zorder=5
    )
    selected_peak2 = selected_ax.scatter(
        [], [], c="orange", s=65, edgecolors="black",
        label="Second-strongest recorded peak", zorder=5
    )

    selected_ax.set_xlim(0, 360)
    selected_ax.set_xticks(np.arange(0, 361, 30))
    selected_ax.set_xlabel("DOA angle (degrees)")
    selected_ax.set_ylabel(color_label)
    selected_ax.set_title("Click a heatmap column to inspect its response")
    selected_ax.grid(True, alpha=0.3)
    selected_ax.legend()

    # State used to map visible heatmap columns to CSV rows.
    visible_indices = np.arange(len(times), dtype=int)

    def inspect_timestep(event):
        """Show the full spatial response for the clicked heatmap column."""
        if event.inaxes is not ax or event.xdata is None:
            return

        # Heatmap columns are packed together after SPL filtering.
        column_index = int(round(event.xdata))

        if not 0 <= column_index < len(visible_indices):
            return

        original_index = int(visible_indices[column_index])

        row = plot_response[original_index]
        selected_line.set_data(angles, row)

        p1 = peak1[original_index]
        p2 = peak2[original_index]

        # Peak angles are stored in degrees 0..359. Map each to
        # the nearest measured angular sample for its marker height.
        if np.isfinite(p1):
            angle1 = float(p1 % 360.0)
            idx1 = int(np.argmin(
                np.abs((angles - angle1 + 180.0) % 360.0 - 180.0)
            ))
            selected_peak1.set_offsets(
                np.array([[angle1, row[idx1]]])
            )
        else:
            selected_peak1.set_offsets(np.empty((0, 2)))

        if np.isfinite(p2):
            angle2 = float(p2 % 360.0)
            idx2 = int(np.argmin(
                np.abs((angles - angle2 + 180.0) % 360.0 - 180.0)
            ))
            selected_peak2.set_offsets(
                np.array([[angle2, row[idx2]]])
            )
        else:
            selected_peak2.set_offsets(np.empty((0, 2)))

        selected_ax.set_title(
            f"Spatial response at t = {times[original_index]:.3f} s "
            f"(SPL = {spl[original_index]:.1f} dB)"
        )

        finite_values = row[np.isfinite(row)]
        if finite_values.size:
            ymin = float(np.min(finite_values))
            ymax = float(np.max(finite_values))
            margin = max((ymax - ymin) * 0.08, 0.1)
                    # Automatically scale the selected-timestep graph to its data.
        finite_values = row[np.isfinite(row)]

        if finite_values.size:
            ymin = float(np.min(finite_values))
            ymax = float(np.max(finite_values))

            data_range = ymax - ymin

            if data_range > 0:
                # Add 8% padding above and below the data.
                margin = data_range * 0.08
            else:
                # Handle a flat or all-zero response.
                margin = max(abs(ymin) * 0.1, 1e-6)

            selected_ax.set_ylim(
                ymin - margin,
                ymax + margin,
            )

            # Autoscale horizontally to show the full circle.
            selected_ax.set_xlim(0, 360)
            selected_ax.set_xticks(np.arange(0, 361, 30))

            selected_fig.canvas.draw_idle()

    # --------------------------------------------------------
    # Heatmap and slider update
    # --------------------------------------------------------

    def update(_value=None):
        nonlocal colorbar, visible_indices

        threshold = float(spl_slider.val)
        keep = spl >= threshold

        visible_indices = np.flatnonzero(keep)
        kept_times = times[visible_indices]
        kept_response = plot_response[visible_indices, :]
        kept_peak1 = peak1[visible_indices]
        kept_peak2 = peak2[visible_indices]

        if colorbar is not None:
            colorbar.remove()
            colorbar = None

        ax.clear()

        if len(kept_times) == 0:
            ax.text(
                0.5, 0.5,
                "No measurements at or above this SPL threshold",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_xlim(-0.5, 0.5)

        else:
            x_edges = np.arange(
                len(kept_times) + 1, dtype=float
            ) - 0.5

            # Angles are 0..357 degrees, with one 3-degree bin per sample.
            angle_edges = np.arange(-1.5, 360.0, 3.0)

            mesh = ax.pcolormesh(
                x_edges,
                angle_edges,
                np.ma.masked_invalid(kept_response.T),
                shading="flat",
                cmap="viridis",
            )

            colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            colorbar.set_label(color_label)

            for x_index, (p1, p2) in enumerate(
                zip(kept_peak1, kept_peak2)
            ):
                if np.isfinite(p1):
                    ax.scatter(
                        [x_index], [p1 % 360.0],
                        marker="o", s=42, c="red",
                        edgecolors="black", linewidths=0.5,
                        zorder=6,
                    )

                if np.isfinite(p2):
                    ax.scatter(
                        [x_index], [p2 % 360.0],
                        marker="o", s=42, c="orange",
                        edgecolors="black", linewidths=0.5,
                        zorder=6,
                    )

            tick_count = min(10, len(kept_times))
            tick_indices = np.unique(
                np.linspace(
                    0, len(kept_times) - 1, tick_count
                ).round().astype(int)
            )

            ax.set_xticks(tick_indices)
            ax.set_xticklabels([
                f"{kept_times[i]:.2f}" for i in tick_indices
            ])
            ax.set_xlim(-0.5, len(kept_times) - 0.5)

        ax.set_xlabel(
            f"{time_label} (filtered timesteps packed together; "
            "labels show original time)"
        )
        ax.set_ylabel("DOA angle (degrees)")
        ax.set_ylim(-1.5, 358.5)
        ax.set_yticks(np.arange(0, 361, 30))
        ax.set_title(
            f"DOA spatial-response heatmap — {csv_path.name}"
        )

        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=9,
            title="Peak marker legend",
            frameon=True,
        )

        count_text.set_text(
            f"Showing {len(visible_indices)} of {len(spl)} measurements "
            f"(SPL ≥ {threshold:.1f} dB)"
        )

        fig.canvas.draw_idle()

    spl_slider.on_changed(update)
    fig.canvas.mpl_connect("button_press_event", inspect_timestep)

    update()

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved initial-threshold heatmap to: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
