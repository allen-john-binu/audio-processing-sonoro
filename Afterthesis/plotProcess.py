
#!/usr/bin/env python3
"""Three-stage DOA spatial-response analysis.

STEP 1: Raw spatial-response heatmap.
STEP 2: Heatmap after db_spl > 50 filtering.
STEP 3: Divide recording into three equal-duration intervals.
        Select two spatial-response curves per interval, separated
        by more than 20 degrees, and add the complete curves.

All displayed DOA angles use the -180 to +180 degree convention.
Peak selection and curve addition use original linear response values.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPL_THRESHOLD = 50.0
ANGLE_EXCLUSION_DEGREES = 20.0
N_INTERVALS = 3

REQUIRED_COLUMNS = [
    "timestamp",
    "db_spl",
    "peak_angle_1",
    "peak_height_1",
    "peak_angle_2",
    "peak_height_2",
]

SPATIAL_COLUMN_PATTERN = re.compile(r"^spatial_resp_(\d+)$")


# ------------------------------------------------------------
# Detect and sort spatial-response columns
# ------------------------------------------------------------

def detect_spatial_columns(columns):

    found = {}

    for column in columns:

        match = SPATIAL_COLUMN_PATTERN.fullmatch(str(column))

        if match:
            index = int(match.group(1))

            if index in found:
                raise ValueError(
                    f"Duplicate spatial-response index: {index}"
                )

            found[index] = column

    if not found:
        raise ValueError(
            "No spatial_resp_<integer> columns were found."
        )

    indices = sorted(found)

    expected = list(range(len(indices)))

    if indices != expected:
        missing = sorted(set(expected) - set(indices))

        raise ValueError(
            "Spatial-response indices must be contiguous "
            "and start at zero.\n"
            f"Detected: {indices}\n"
            f"Missing: {missing}"
        )

    return [found[i] for i in indices]


# ------------------------------------------------------------
# Load CSV
# ------------------------------------------------------------

def load_csv(csv_path):

    header = pd.read_csv(csv_path, nrows=0).columns

    missing = [
        c for c in REQUIRED_COLUMNS if c not in header
    ]

    if missing:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(missing)
        )

    spatial_columns = detect_spatial_columns(header)

    n_angles = len(spatial_columns)

    angular_resolution = 360.0 / n_angles

    # Native angles: 0 to 360 degrees.
    native_angles = (
        np.arange(n_angles, dtype=float)
        * angular_resolution
    )

    # Convert to -180 to +180 degrees.
    signed_angles = (
        (native_angles + 180.0) % 360.0
    ) - 180.0

    # Sort angles and reorder response columns accordingly.
    angle_order = np.argsort(signed_angles)

    angles = signed_angles[angle_order]

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("The CSV contains no data rows.")

    timestamps = pd.to_numeric(
        df["timestamp"], errors="coerce"
    ).to_numpy(dtype=float)

    spl = pd.to_numeric(
        df["db_spl"], errors="coerce"
    ).to_numpy(dtype=float)

    response = df[spatial_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    response = response[:, angle_order]

    valid_timestamps = np.isfinite(timestamps)

    if not valid_timestamps.any():
        raise ValueError("No valid timestamps found.")

    first_timestamp = timestamps[
        np.flatnonzero(valid_timestamps)[0]
    ]

    elapsed = timestamps - first_timestamp

    valid_rows = (
        np.isfinite(elapsed)
        & np.isfinite(spl)
    )

    elapsed = elapsed[valid_rows]
    spl = spl[valid_rows]
    response = response[valid_rows]

    # Sort chronologically.
    order = np.argsort(elapsed)

    elapsed = elapsed[order]
    spl = spl[order]
    response = response[order]

    if elapsed.size == 0:
        raise ValueError("No valid measurements found.")

    print("\nCSV successfully loaded")
    print("-----------------------")
    print(f"Measurements: {len(elapsed)}")
    print(f"Spatial samples: {n_angles}")
    print(f"Angular resolution: {angular_resolution:g} degrees")
    print(f"Duration: {elapsed[-1]:.3f} seconds")

    return elapsed, spl, angles, response


# ------------------------------------------------------------
# Convert spatial response for display
# ------------------------------------------------------------

def response_for_display(response, as_db):

    if not as_db:
        return response.copy()

    with np.errstate(divide="ignore", invalid="ignore"):

        return np.where(
            response > 0,
            10.0 * np.log10(response),
            np.nan
        )


# ------------------------------------------------------------
# Shared color limits
# ------------------------------------------------------------

def shared_color_limits(arrays):

    finite_parts = [
        a[np.isfinite(a)]
        for a in arrays
        if np.isfinite(a).any()
    ]

    if not finite_parts:
        raise ValueError(
            "No finite spatial-response values available."
        )

    values = np.concatenate(finite_parts)

    low = float(values.min())
    high = float(values.max())

    if low == high:

        margin = max(abs(low) * 0.01, 1e-12)

        low -= margin
        high += margin

    return low, high


# ------------------------------------------------------------
# Time-axis formatting
# ------------------------------------------------------------

def format_time_ticks(ax, times, count=10):

    n = len(times)

    if n == 0:
        return

    indices = np.unique(
        np.linspace(
            0,
            n - 1,
            min(count, n)
        ).round().astype(int)
    )

    ax.set_xticks(indices)

    ax.set_xticklabels([
        f"{times[i]:.2f}"
        for i in indices
    ])


# ------------------------------------------------------------
# Heatmap plotting
# ------------------------------------------------------------

def plot_heatmap(
    ax,
    times,
    response,
    angles,
    angle_edges,
    title,
    color_label,
    vmin,
    vmax
):

    if len(times) == 0:

        ax.text(
            0.5,
            0.5,
            "No measurements available",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("DOA angle (degrees)")
        ax.set_ylim(-180, 180)

        return None

    x_edges = (
        np.arange(len(times) + 1, dtype=float) - 0.5
    )

    # Repeat first angular sample to close the circular seam.
    response_with_seam = np.column_stack([
        response,
        response[:, 0]
    ])

    mesh = ax.pcolormesh(
        x_edges,
        angle_edges,
        np.ma.masked_invalid(response_with_seam.T),
        shading="flat",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlim(
        -0.5,
        len(times) - 0.5
    )

    format_time_ticks(ax, times)

    ax.set_ylim(-180, 180)

    ax.set_yticks(
        np.arange(-180, 181, 30)
    )

    ax.set_ylabel("DOA angle (degrees)")
    ax.set_xlabel("Elapsed time (seconds)")

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=10
    )

    ax.grid(
        axis="y",
        alpha=0.15,
        linestyle="--"
    )

    return mesh


# ------------------------------------------------------------
# Circular angular distance
# ------------------------------------------------------------

def circular_distance_deg(a, b):

    difference = np.abs(a - b) % 360.0

    return np.minimum(
        difference,
        360.0 - difference
    )


# ------------------------------------------------------------
# STEP 3: Select two peaks and combine their curves
# ------------------------------------------------------------

def select_and_combine_interval(
    interval_times,
    interval_response,
    angles,
    interval_number
):

    if len(interval_times) == 0:
        return None

    finite = np.isfinite(interval_response)

    if not finite.any():
        return None

    safe = np.where(
        finite,
        interval_response,
        -np.inf
    )

    # Find the strongest response across all timesteps
    # and all angular samples in this interval.

    first_flat = int(np.argmax(safe))

    first_row, first_angle_index = np.unravel_index(
        first_flat,
        safe.shape
    )

    first_value = float(
        interval_response[first_row, first_angle_index]
    )

    first_angle = float(
        angles[first_angle_index]
    )

    # Exclude angular samples within 20 degrees
    # of the strongest peak.
    #
    # Circular distance is used, so -180 and +180
    # are treated as the same direction.

    allowed_angles = (
        circular_distance_deg(
            angles,
            first_angle
        ) > ANGLE_EXCLUSION_DEGREES
    )

    if not allowed_angles.any():
        return None

    # Search for the strongest remaining sample
    # across all timesteps in the interval.

    second_candidates = safe.copy()

    second_candidates[:, ~allowed_angles] = -np.inf

    if not np.isfinite(second_candidates).any():
        return None

    second_flat = int(
        np.argmax(second_candidates)
    )

    second_row, second_angle_index = np.unravel_index(
        second_flat,
        second_candidates.shape
    )

    second_value = float(
        interval_response[second_row, second_angle_index]
    )

    second_angle = float(
        angles[second_angle_index]
    )

    # Take the COMPLETE spatial-response curve
    # from each selected timestep and add them.

    combined = (
        interval_response[first_row, :]
        + interval_response[second_row, :]
    )

    return {
        "interval": interval_number,

        "first_row": int(first_row),
        "first_time": float(interval_times[first_row]),
        "first_angle": first_angle,
        "first_value": first_value,

        "second_row": int(second_row),
        "second_time": float(interval_times[second_row]),
        "second_angle": second_angle,
        "second_value": second_value,

        "combined": combined
    }


# ------------------------------------------------------------
# Divide recording into three equal-duration intervals
# ------------------------------------------------------------

def build_step3_results(
    times,
    spl,
    response,
    angles
):

    start = float(np.min(times))
    end = float(np.max(times))

    if not end > start:

        raise ValueError(
            "Recording duration is zero. "
            "Cannot create three time intervals."
        )

    boundaries = np.linspace(
        start,
        end,
        N_INTERVALS + 1
    )

    results = []

    for i in range(N_INTERVALS):

        # Half-open intervals avoid double-counting
        # measurements on interval boundaries.

        if i < N_INTERVALS - 1:

            mask = (
                (times >= boundaries[i])
                & (times < boundaries[i + 1])
            )

        else:

            mask = (
                (times >= boundaries[i])
                & (times <= boundaries[i + 1])
            )

        # Apply SPL filtering independently
        # inside each interval.

        mask &= spl > SPL_THRESHOLD

        interval_times = times[mask]

        interval_response = response[mask, :]

        result = select_and_combine_interval(
            interval_times,
            interval_response,
            angles,
            i + 1
        )

        # Skip intervals without a valid peak pair.

        if result is not None:

            result["boundary_start"] = float(
                boundaries[i]
            )

            result["boundary_end"] = float(
                boundaries[i + 1]
            )

            result["filtered_measurements"] = int(
                mask.sum()
            )

            results.append(result)

    return results, boundaries


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Three-stage DOA spatial-response analysis."
    )

    parser.add_argument(
        "csv",
        help="Path to the DOA discovery CSV"
    )

    parser.add_argument(
        "--db",
        action="store_true",
        help=(
            "Display spatial response as 10*log10(response). "
            "Peak selection and curve addition remain in linear units."
        )
    )

    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the complete figure."
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    times, spl, angles, response = load_csv(csv_path)

    n_angles = len(angles)

    angular_resolution = 360.0 / n_angles

    # Build angular bin edges, including the circular seam.
    # The first response sample is repeated at +180 degrees.

    extended_angles = np.r_[angles, angles[0] + 360.0]

    angle_edges = (
        extended_angles[:-1] + extended_angles[1:]
    ) / 2.0

    angle_edges = np.r_[
        (extended_angles[-2] - 360.0 + angles[0]) / 2.0,
        angle_edges,
        (extended_angles[-1] + angles[0] + 360.0) / 2.0
    ]

    # Keep the visible angular range exactly -180 to +180.
    angle_edges[0] = -180.0
    angle_edges[-1] = 180.0

    # --------------------------------------------------------
    # STEP 1 and STEP 2: Prepare heatmap data
    # --------------------------------------------------------

    display_response = response_for_display(
        response,
        args.db
    )

    filtered_mask = spl > SPL_THRESHOLD

    filtered_times = times[filtered_mask]

    filtered_response = display_response[filtered_mask, :]

    # --------------------------------------------------------
    # STEP 3: Selection and combination
    # --------------------------------------------------------

    step3_results, interval_boundaries = build_step3_results(
        times,
        spl,
        response,
        angles
    )

    if step3_results:

        combined_linear = np.vstack([
            r["combined"]
            for r in step3_results
        ])

    else:

        combined_linear = np.empty(
            (0, n_angles),
            dtype=float
        )

    combined_display = response_for_display(
        combined_linear,
        args.db
    )

    # --------------------------------------------------------
    # Shared color scale
    # --------------------------------------------------------

    color_label = (
        "Spatial response (dB, 10·log10)"
        if args.db
        else "Spatial response"
    )

    vmin, vmax = shared_color_limits([
        display_response,
        combined_display
    ])

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------

    fig = plt.figure(figsize=(18, 13))

    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.12],
        width_ratios=[1.05, 1.0],
        hspace=0.38,
        wspace=0.28
    )

    ax1 = fig.add_subplot(outer[0, 0])

    ax2 = fig.add_subplot(
        outer[0, 1],
        sharey=ax1
    )

    ax3 = fig.add_subplot(
        outer[1, 0],
        sharey=ax1
    )

    # Three line graphs on the bottom-right.
    line_grid = outer[1, 1].subgridspec(
        3,
        1,
        hspace=0.48
    )

    line_axes = [
        fig.add_subplot(line_grid[i, 0])
        for i in range(3)
    ]

    fig.suptitle(
        f"DOA Spatial Response Analysis — {csv_path.name}",
        fontsize=17,
        fontweight="bold",
        y=0.985
    )

    # --------------------------------------------------------
    # STEP 1: RAW HEATMAP
    # --------------------------------------------------------

    mesh1 = plot_heatmap(
        ax1,
        times,
        display_response,
        angles,
        angle_edges,
        f"STEP 1 — RAW SPATIAL RESPONSE "
        f"({len(times)} measurements)",
        color_label,
        vmin,
        vmax
    )

    if mesh1 is not None:

        fig.colorbar(
            mesh1,
            ax=ax1,
            pad=0.02
        ).set_label(color_label)

    # --------------------------------------------------------
    # STEP 2: FILTERED HEATMAP
    # --------------------------------------------------------

    mesh2 = plot_heatmap(
        ax2,
        filtered_times,
        filtered_response,
        angles,
        angle_edges,
        f"STEP 2 — DB_SPL FILTERING "
        f"(db_spl > {SPL_THRESHOLD:g}; "
        f"{len(filtered_times)} measurements)",
        color_label,
        vmin,
        vmax
    )

    if mesh2 is not None:

        fig.colorbar(
            mesh2,
            ax=ax2,
            pad=0.02
        ).set_label(color_label)

    # --------------------------------------------------------
    # STEP 3: COMBINED HEATMAP
    # --------------------------------------------------------

    ax3.set_title(
        "STEP 3 — COMBINED SPATIAL RESPONSE BY INTERVAL",
        fontsize=12,
        fontweight="bold",
        pad=10
    )

    ax3.set_xlabel("Time interval")
    ax3.set_ylabel("DOA angle (degrees)")

    ax3.set_ylim(-180, 180)

    ax3.set_yticks(
        np.arange(-180, 181, 30)
    )

    ax3.grid(
        axis="y",
        alpha=0.15,
        linestyle="--"
    )

    if step3_results:

        x_edges = (
            np.arange(len(step3_results) + 1, dtype=float)
            - 0.5
        )

        # Repeat first angle sample to close the seam.
        combined_with_seam = np.vstack([
            combined_display.T,
            combined_display[:, 0][np.newaxis, :]
        ])

        mesh3 = ax3.pcolormesh(
            x_edges,
            angle_edges,
            np.ma.masked_invalid(combined_with_seam),
            shading="flat",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )

        ax3.set_xlim(
            -0.5,
            len(step3_results) - 0.5
        )

        ax3.set_xticks(
            np.arange(len(step3_results))
        )

        ax3.set_xticklabels([
            f"Interval {r['interval']}"
            for r in step3_results
        ])

        fig.colorbar(
            mesh3,
            ax=ax3,
            pad=0.02
        ).set_label(color_label)

    else:

        ax3.text(
            0.5,
            0.5,
            "No interval had a valid pair of peaks",
            ha="center",
            va="center",
            transform=ax3.transAxes
        )

    # --------------------------------------------------------
    # STEP 3: THREE LINE GRAPHS
    # --------------------------------------------------------

    result_by_interval = {
        r["interval"]: r
        for r in step3_results
    }

    for interval_number, ax in enumerate(
        line_axes,
        start=1
    ):

        result = result_by_interval.get(
            interval_number
        )

        ax.set_title(
            f"STEP 3 — INTERVAL {interval_number}",
            fontsize=11,
            fontweight="bold"
        )

        ax.set_xlabel("DOA angle (degrees)")
        ax.set_ylabel(color_label)

        ax.set_xlim(-180, 180)

        ax.set_xticks(
            np.arange(-180, 181, 30)
        )

        ax.grid(
            True,
            alpha=0.25,
            linestyle="--"
        )

        if result is None:

            ax.text(
                0.5,
                0.5,
                "Skipped — no valid pair of peaks",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

            continue

        # Convert combined response for display if --db is used.
        line_values = response_for_display(
            result["combined"][np.newaxis, :],
            args.db
        )[0]

        # Close the curve at the circular seam.
        line_angles = np.r_[angles, 180.0]

        line_values = np.r_[
            line_values,
            line_values[0]
        ]

        ax.plot(
            line_angles,
            line_values,
            linewidth=1.8,
            label="Combined response"
        )

        ax.axvline(
            result["first_angle"],
            linestyle="--",
            alpha=0.65,
            label=f"Peak 1: {result['first_angle']:.1f}°"
        )

        ax.axvline(
            result["second_angle"],
            linestyle=":",
            alpha=0.8,
            label=f"Peak 2: {result['second_angle']:.1f}°"
        )

        ax.legend(
            loc="best",
            fontsize=8
        )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 64)
    print("DOA SPATIAL RESPONSE ANALYSIS")
    print("=" * 64)

    print(f"Measurements: {len(times)}")
    print(f"Spatial samples: {n_angles}")
    print(f"Angular resolution: {angular_resolution:g} degrees")

    print(
        f"Recording time range: "
        f"{times.min():.3f} to {times.max():.3f} seconds"
    )

    print(
        f"STEP 2 retained (db_spl > {SPL_THRESHOLD:g}): "
        f"{filtered_mask.sum()} / {len(times)}"
    )

    print("\nSTEP 3 — INTERVAL SELECTION")

    for i in range(N_INTERVALS):

        print(
            f"\nInterval {i + 1}: "
            f"{interval_boundaries[i]:.3f} to "
            f"{interval_boundaries[i + 1]:.3f} seconds"
        )

        result = result_by_interval.get(i + 1)

        if result is None:

            print(
                "  SKIPPED: no valid pair of peaks."
            )

            continue

        separation = circular_distance_deg(
            result["first_angle"],
            result["second_angle"]
        )

        print(
            f"  Peak 1: {result['first_value']:.6g} "
            f"at {result['first_angle']:.1f} degrees "
            f"(time {result['first_time']:.3f} s)"
        )

        print(
            f"  Peak 2: {result['second_value']:.6g} "
            f"at {result['second_angle']:.1f} degrees "
            f"(time {result['second_time']:.3f} s)"
        )

        print(
            f"  Angular separation: {separation:.1f} degrees"
        )

        print(
            "  Combined curve: sum of both complete timestep curves."
        )

    print("\n" + "=" * 64)

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    fig.subplots_adjust(
        top=0.93,
        bottom=0.07,
        left=0.07,
        right=0.96
    )

    if args.save:

        fig.savefig(
            args.save,
            dpi=200,
            bbox_inches="tight"
        )

        print(f"\nFigure saved to: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()


# Step 3 – Spatial Response Selection and Combination
# 1. Get 6 doa estimates where db_spl > 50.

# 2. Identify the strongest spatial-response sample across all retained timesteps and DOA angles within the interval.

# 3. Exclude all angular samples within ±20° of the strongest peak, using circular angular distance (considering the −180°/+180° boundary).

# 4. Identify the second-strongest spatial-response sample from the remaining angular samples across all retained timesteps.

# 5. The two selected peaks can originate from different timesteps.

# 6. Extract the complete spatial-response curves from the two selected timesteps.

# 7. Add the two complete curves element by element to obtain one combined spatial-response curve for that interval.

# 8. If no eligible second peak exists outside the 20° exclusion zone, skip that interval.

# 9. Repeat the process, producing combined spatial-response curves.