
#!/usr/bin/env python3
"""Interactive DOA heatmap for dynamically sized spatial-response CSV data.

Expected columns:
- timestamp: Unix timestamp in seconds
- db_spl
- peak_angle_1, peak_height_1
- peak_angle_2, peak_height_2
- spatial_resp_0 through spatial_resp_N-1

Spatial-response columns are detected automatically from the CSV header.
Their angular positions are distributed uniformly over a complete 360-degree
circle.

Click a heatmap column to inspect the spatial response for that measurement.
The time axis is elapsed seconds from the first valid Unix timestamp.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider


# ------------------------------------------------------------
# CSV configuration
# ------------------------------------------------------------

REQUIRED_COLUMNS = [
    "timestamp",
    "db_spl",
    "peak_angle_1",
    "peak_height_1",
    "peak_angle_2",
    "peak_height_2",
]

SPATIAL_COLUMN_PATTERN = re.compile(r"^spatial_resp_(\d+)$")


def detect_spatial_columns(columns):
    """Detect, validate and numerically sort spatial-response columns."""

    spatial_columns = {}

    for column in columns:
        match = SPATIAL_COLUMN_PATTERN.fullmatch(str(column))

        if match:
            index = int(match.group(1))

            if index in spatial_columns:
                raise ValueError(
                    f"Duplicate spatial-response index detected: {index}"
                )

            spatial_columns[index] = column

    if not spatial_columns:
        raise ValueError(
            "No spatial-response columns found. Expected columns matching "
            "'spatial_resp_<integer>', e.g. spatial_resp_0."
        )

    indices = sorted(spatial_columns.keys())

    expected_indices = list(range(len(indices)))

    if indices != expected_indices:
        missing = sorted(set(expected_indices) - set(indices))

        raise ValueError(
            "Spatial-response columns must have contiguous indices "
            "starting at zero.\n"
            f"Detected indices: {indices}\n"
            f"Missing indices: {missing}"
        )

    return [spatial_columns[i] for i in indices]


# ------------------------------------------------------------
# CSV loading and validation
# ------------------------------------------------------------

def load_csv(csv_path: Path):
    """Load and validate CSV data with a dynamically sized spatial response."""

    # First scan the header to discover the available spatial-response columns.
    header = pd.read_csv(csv_path, nrows=0).columns

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in header
    ]

    if missing:
        raise ValueError(
            "CSV does not match the expected AudioProcessor format. "
            f"Missing columns: {', '.join(missing)}"
        )

    # Detect and validate spatial-response columns before loading all data.
    response_columns = detect_spatial_columns(header)

    n_spatial_samples = len(response_columns)

    # Distribute samples uniformly over a complete 360-degree circle.
    angular_resolution = 360.0 / n_spatial_samples

    angles = (
        np.arange(n_spatial_samples, dtype=float)
        * angular_resolution
    )

    # Now load the complete CSV.
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("The CSV contains no data rows.")

    unix_timestamps = pd.to_numeric(
        df["timestamp"], errors="coerce"
    ).to_numpy(dtype=float)

    spl = pd.to_numeric(
        df["db_spl"], errors="coerce"
    ).to_numpy(dtype=float)

    finite_timestamps = np.isfinite(unix_timestamps)

    if not finite_timestamps.any():
        raise ValueError("The CSV contains no valid Unix timestamps.")

    # Preserve elapsed time relative to the first valid timestamp in file order.
    first_timestamp = unix_timestamps[
        np.flatnonzero(finite_timestamps)[0]
    ]

    times = unix_timestamps - first_timestamp
    time_label = "Elapsed time (s)"

    # Read spatial-response data in numeric index order.
    response = df[response_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    peak1 = pd.to_numeric(
        df["peak_angle_1"], errors="coerce"
    ).to_numpy(dtype=float)

    peak2 = pd.to_numeric(
        df["peak_angle_2"], errors="coerce"
    ).to_numpy(dtype=float)

    # Peak-height columns remain required as part of the CSV schema.
    # Marker heights are obtained from the nearest spatial-response sample,
    # preserving the original plotting behavior.

    # Keep rows with valid timestamps and SPL values.
    valid = np.isfinite(times) & np.isfinite(spl)

    times = times[valid]
    spl = spl[valid]
    response = response[valid, :]
    peak1 = peak1[valid]
    peak2 = peak2[valid]

    # Sort all measurements by elapsed time.
    order = np.argsort(times)

    times = times[order]
    spl = spl[order]
    response = response[order, :]
    peak1 = peak1[order]
    peak2 = peak2[order]

    if len(times) == 0:
        raise ValueError(
            "No rows have valid timestamps and SPL values."
        )

    print(f"Detected spatial-response samples: {n_spatial_samples}")
    print(f"Angular resolution: {angular_resolution:g} degrees")
    print(f"Spatial-response columns: {response_columns[0]} "
          f"through {response_columns[-1]}")

    return (
        times,
        spl,
        angles,
        response,
        peak1,
        peak2,
        time_label,
        angular_resolution,
    )


# ------------------------------------------------------------
# Main plotting application
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Interactive DOA heatmap and single-timestep inspection."
    )

    parser.add_argument(
        "csv",
        help="Path to the DOA discovery CSV"
    )

    parser.add_argument(
        "--min-spl",
        type=float,
        default=None,
        help="Initial minimum SPL threshold (default: minimum in CSV)."
    )

    parser.add_argument(
        "--db",
        action="store_true",
        help="Display spatial response as 10*log10(response)."
    )

    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the initial heatmap image."
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
        angular_resolution,
    ) = load_csv(csv_path)

    n_spatial_samples = len(angles)

    # --------------------------------------------------------
    # Prepare spatial-response plotting data
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SPL slider configuration
    # --------------------------------------------------------

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
        left=0.10,
        right=0.73,
        top=0.91,
        bottom=0.25
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

    count_text = fig.text(
        0.5, 0.155, "", ha="center"
    )

    colorbar = None

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor="red",
            markeredgecolor="black",
            label="Strongest recorded peak",
        ),

        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor="orange",
            markeredgecolor="black",
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
        [], [],
        "-o",
        markersize=3,
        linewidth=1
    )

    selected_peak1 = selected_ax.scatter(
        [], [],
        c="red",
        s=65,
        edgecolors="black",
        label="Strongest recorded peak",
        zorder=5
    )

    selected_peak2 = selected_ax.scatter(
        [], [],
        c="orange",
        s=65,
        edgecolors="black",
        label="Second-strongest recorded peak",
        zorder=5
    )

    selected_ax.set_xlim(0, 360)
    selected_ax.set_xticks(np.arange(0, 361, 30))

    selected_ax.set_xlabel("DOA angle (degrees)")
    selected_ax.set_ylabel(color_label)

    selected_ax.set_title(
        "Click a heatmap column to inspect its response"
    )

    selected_ax.grid(True, alpha=0.3)
    selected_ax.legend()

    # --------------------------------------------------------
    # Dynamic heatmap angular bin edges
    # --------------------------------------------------------

    # Samples are centered at:
    # 0, angular_resolution, 2*angular_resolution, ...
    #
    # Therefore, each bin extends half an angular step on either side.
    #
    # This produces exactly N+1 edges for N spatial-response samples.

    angle_edges = (
        np.arange(n_spatial_samples + 1, dtype=float)
        * angular_resolution
        - angular_resolution / 2.0
    )

    # State used to map visible heatmap columns to CSV rows.
    visible_indices = np.arange(len(times), dtype=int)

    # --------------------------------------------------------
    # Selected-timestep inspection
    # --------------------------------------------------------

    def inspect_timestep(event):
        """Show the full spatial response for the clicked heatmap column."""

        if event.inaxes is not ax or event.xdata is None:
            return

        # Heatmap columns are packed together after SPL filtering.
        column_index = int(round(event.xdata))

        if not 0 <= column_index < len(visible_indices):
            return

        original_index = int(
            visible_indices[column_index]
        )

        row = plot_response[original_index]

        selected_line.set_data(angles, row)

        p1 = peak1[original_index]
        p2 = peak2[original_index]

        # Map each recorded peak angle to the nearest detected spatial
        # response sample, accounting for circular angular distance.

        if np.isfinite(p1):

            angle1 = float(p1 % 360.0)

            idx1 = int(np.argmin(
                np.abs(
                    (angles - angle1 + 180.0) % 360.0 - 180.0
                )
            ))

            selected_peak1.set_offsets(
                np.array([[angle1, row[idx1]]])
            )

        else:

            selected_peak1.set_offsets(
                np.empty((0, 2))
            )

        if np.isfinite(p2):

            angle2 = float(p2 % 360.0)

            idx2 = int(np.argmin(
                np.abs(
                    (angles - angle2 + 180.0) % 360.0 - 180.0
                )
            ))

            selected_peak2.set_offsets(
                np.array([[angle2, row[idx2]]])
            )

        else:

            selected_peak2.set_offsets(
                np.empty((0, 2))
            )

        selected_ax.set_title(
            f"Spatial response at t = {times[original_index]:.3f} s "
            f"(SPL = {spl[original_index]:.1f} dB)"
        )

        finite_values = row[np.isfinite(row)]

        if finite_values.size:

            ymin = float(np.min(finite_values))
            ymax = float(np.max(finite_values))

            data_range = ymax - ymin

            if data_range > 0:
                margin = data_range * 0.08
            else:
                margin = max(abs(ymin) * 0.1, 1e-6)

            selected_ax.set_ylim(
                ymin - margin,
                ymax + margin
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
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            ax.set_xlim(-0.5, 0.5)

        else:

            x_edges = (
                np.arange(len(kept_times) + 1, dtype=float)
                - 0.5
            )

            # Use dynamically calculated angular bin edges.
            mesh = ax.pcolormesh(
                x_edges,
                angle_edges,
                np.ma.masked_invalid(kept_response.T),
                shading="flat",
                cmap="viridis",
            )

            colorbar = fig.colorbar(
                mesh,
                ax=ax,
                pad=0.02
            )

            colorbar.set_label(color_label)

            # Plot recorded peak angles.
            for x_index, (p1, p2) in enumerate(
                zip(kept_peak1, kept_peak2)
            ):

                if np.isfinite(p1):

                    ax.scatter(
                        [x_index],
                        [p1 % 360.0],
                        marker="o",
                        s=42,
                        c="red",
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=6,
                    )

                if np.isfinite(p2):

                    ax.scatter(
                        [x_index],
                        [p2 % 360.0],
                        marker="o",
                        s=42,
                        c="orange",
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=6,
                    )

            # Time-axis labels.
            tick_count = min(10, len(kept_times))

            tick_indices = np.unique(
                np.linspace(
                    0,
                    len(kept_times) - 1,
                    tick_count
                ).round().astype(int)
            )

            ax.set_xticks(tick_indices)

            ax.set_xticklabels([
                f"{kept_times[i]:.2f}"
                for i in tick_indices
            ])

            ax.set_xlim(
                -0.5,
                len(kept_times) - 0.5
            )

        # ----------------------------------------------------
        # Heatmap axes
        # ----------------------------------------------------

        ax.set_xlabel(
            f"{time_label} (filtered timesteps packed together; "
            "labels show original time)"
        )

        ax.set_ylabel("DOA angle (degrees)")

        # Display the complete extent of the angular bins.
        ax.set_ylim(
            angle_edges[0],
            angle_edges[-1]
        )

        # Keep readable angular tick labels within the displayed range.
        angular_ticks = np.arange(0, 360, 30, dtype=float)

        angular_ticks = angular_ticks[
            (angular_ticks >= angle_edges[0])
            & (angular_ticks <= angle_edges[-1])
        ]

        ax.set_yticks(angular_ticks)

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

    # --------------------------------------------------------
    # Connect callbacks
    # --------------------------------------------------------

    spl_slider.on_changed(update)

    fig.canvas.mpl_connect(
        "button_press_event",
        inspect_timestep
    )

    # Initial heatmap.
    update()

    # --------------------------------------------------------
    # Optional image saving
    # --------------------------------------------------------

    if args.save:

        fig.savefig(
            args.save,
            dpi=200,
            bbox_inches="tight"
        )

        print(
            f"Saved initial-threshold heatmap to: {args.save}"
        )

    # Display both interactive figures.
    plt.show()


if __name__ == "__main__":
    main()
