#!/usr/bin/env python3
"""Interactive DOA interval processing and visualization.

Expected CSV columns:
- timestamp: Unix timestamp in seconds
- db_spl
- peak_angle_1, peak_height_1
- peak_angle_2, peak_height_2
- spatial_resp_0 through spatial_resp_N-1

Spatial-response columns are detected automatically. Their angular positions
are assumed to be uniformly distributed over one complete 360-degree circle.

Processing:
1. Convert the angular ordering from [0, 360) to [-180, 180).
2. Divide the recording into complete, recording-start-anchored intervals.
3. Keep measurements with SPL >= SPL_THRESHOLD within each interval.
4. Skip intervals with no qualifying measurements.
5. Take the maximum response independently at every angular bin.
6. Min-max normalize each aggregated interval independently to [0, 1].
   A constant profile is returned as all ones.

The script opens three figures:
1. Complete raw DOA heatmap.
2. Raw heatmap with interval boundaries and an interactive interval selector,
   alongside a detailed heatmap of the selected interval.
3. One normalized, aggregated DOA profile per retained interval.

The public process_doas() function returns an array with shape
(number_of_retained_intervals, number_of_angular_bins).
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider


# ------------------------------------------------------------
# Configuration
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

# Processing parameters agreed for the current analysis.
INTERVAL_SECONDS = 5.0
SPL_THRESHOLD = 45.0

COLOR_MAP = "viridis"


# ------------------------------------------------------------
# Spatial-response column detection
# ------------------------------------------------------------

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

    indices = sorted(spatial_columns)
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
# Angular conversion
# ------------------------------------------------------------

def angular_order_and_coordinates(n_angles):
    """Return column order and angular coordinates in [-180, 180).

    spatial_resp_0 is assumed to correspond to 0 degrees, with the remaining
    bins uniformly distributed in increasing angular order around 360 degrees.
    """

    if n_angles < 1:
        raise ValueError("At least one angular bin is required.")

    original_angles = np.arange(n_angles, dtype=float) * (360.0 / n_angles)
    signed_angles = (original_angles + 180.0) % 360.0 - 180.0

    order = np.argsort(signed_angles)
    return order, signed_angles[order]


def signed_angle(angle):
    """Convert one angle (or an array of angles) to [-180, 180)."""
    return (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------
# CSV loading and validation
# ------------------------------------------------------------

def load_csv(csv_path):
    """Load and validate the DOA discovery CSV."""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0).columns

    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise ValueError(
            "CSV does not match the expected AudioProcessor format. "
            f"Missing columns: {', '.join(missing)}"
        )

    response_columns = detect_spatial_columns(header)
    n_spatial_samples = len(response_columns)
    angular_resolution = 360.0 / n_spatial_samples

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("The CSV contains no data rows.")

    unix_timestamps = pd.to_numeric(
        df["timestamp"], errors="coerce"
    ).to_numpy(dtype=float)

    spl = pd.to_numeric(
        df["db_spl"], errors="coerce"
    ).to_numpy(dtype=float)

    response = df[response_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    peak1 = pd.to_numeric(
        df["peak_angle_1"], errors="coerce"
    ).to_numpy(dtype=float)

    peak2 = pd.to_numeric(
        df["peak_angle_2"], errors="coerce"
    ).to_numpy(dtype=float)

    # Retain rows with valid timestamps and SPL values. Spatial response
    # values may contain NaNs; these are handled during aggregation/plotting.
    valid = np.isfinite(unix_timestamps) & np.isfinite(spl)

    unix_timestamps = unix_timestamps[valid]
    spl = spl[valid]
    response = response[valid, :]
    peak1 = peak1[valid]
    peak2 = peak2[valid]

    if len(unix_timestamps) == 0:
        raise ValueError("No rows have valid timestamps and SPL values.")

    # Sort first, then anchor elapsed time to the earliest valid timestamp.
    sort_order = np.argsort(unix_timestamps)

    unix_timestamps = unix_timestamps[sort_order]
    spl = spl[sort_order]
    response = response[sort_order, :]
    peak1 = peak1[sort_order]
    peak2 = peak2[sort_order]

    times = unix_timestamps - unix_timestamps[0]

    # Keep the original CSV column order for processing.  The plotting
    # response is separately arranged into signed-angle order so that the
    # interval processor can apply the same single conversion to raw columns.
    angular_order, angles = angular_order_and_coordinates(
        n_spatial_samples
    )
    response_original = response.copy()
    response = response[:, angular_order]

    # Convert the recorded peak angles to the same signed convention.
    peak1 = signed_angle(peak1)
    peak2 = signed_angle(peak2)

    print(f"Detected spatial-response samples: {n_spatial_samples}")
    print(f"Angular resolution: {angular_resolution:g} degrees")
    print(f"Angular range: {angles[0]:g} to {angles[-1]:g} degrees")
    print(f"Measurements after timestamp/SPL validation: {len(times)}")

    return {
        "times": times,
        "spl": spl,
        "angles": angles,
        "response": response,
        "response_original": response_original,
        "peak1": peak1,
        "peak2": peak2,
        "angular_resolution": angular_resolution,
        "csv_path": csv_path,
    }


# ------------------------------------------------------------
# DOA interval processing
# ------------------------------------------------------------

def _process_doas_with_details(
    timestamps,
    spl,
    raw_responses,
    T=INTERVAL_SECONDS,
    spl_threshold=SPL_THRESHOLD,
):
    """Process raw DOA profiles and retain interval metadata for plotting.

    Returns:
        processed_doas:
            Float array of shape (retained_intervals, angular_bins).

        details:
            Metadata describing all complete intervals and the measurements
            that passed the SPL threshold in each interval.
    """

    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be a finite number greater than zero.")

    timestamps = np.asarray(timestamps, dtype=float)
    spl = np.asarray(spl, dtype=float)
    raw_responses = np.asarray(raw_responses, dtype=float)

    if raw_responses.ndim != 2:
        raise ValueError("raw_responses must be a 2D array.")

    if not (
        len(timestamps) == len(spl) == raw_responses.shape[0]
    ):
        raise ValueError(
            "Timestamps, SPL and response rows must have equal lengths."
        )

    valid = np.isfinite(timestamps) & np.isfinite(spl)
    timestamps = timestamps[valid]
    spl = spl[valid]
    raw_responses = raw_responses[valid]

    if len(timestamps) == 0:
        raise ValueError("No valid measurements were provided.")

    # Sort and anchor the intervals to the earliest timestamp.
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    spl = spl[order]
    raw_responses = raw_responses[order]
    timestamps = timestamps - timestamps[0]

    n_angles = raw_responses.shape[1]
    angular_order, _ = angular_order_and_coordinates(n_angles)
    responses = raw_responses[:, angular_order]

    # An interval [k*T, (k+1)*T) is complete only when its end is no later
    # than the final timestamp. Any remaining partial interval is discarded.
    recording_duration = float(timestamps[-1])
    n_complete_intervals = int(np.floor(
        (recording_duration + 1e-10) / T
    ))

    processed_profiles = []
    retained_interval_indices = []
    interval_details = []

    for interval_index in range(n_complete_intervals):
        interval_start = interval_index * T
        interval_end = (interval_index + 1) * T

        in_interval = (
            (timestamps >= interval_start)
            & (timestamps < interval_end)
        )

        measurement_indices = np.flatnonzero(in_interval)
        qualifying_indices = np.flatnonzero(
            in_interval & (spl >= spl_threshold)
        )

        interval_details.append({
            "index": interval_index,
            "start": interval_start,
            "end": interval_end,
            "measurement_indices": measurement_indices,
            "qualifying_indices": qualifying_indices,
        })

        # Skip the interval if no measurement passes the SPL threshold.
        if qualifying_indices.size == 0:
            continue

        # Only SPL-qualified measurements contribute to the maximum.
        interval_responses = responses[qualifying_indices]

        # Ignore NaNs independently at each angle. Keep NaN if an entire
        # angular bin has no finite response in this interval.
        finite = np.isfinite(interval_responses)
        aggregated = np.max(
            np.where(finite, interval_responses, -np.inf),
            axis=0,
        )
        aggregated[~np.isfinite(aggregated)] = np.nan

        # Normalize this interval independently to [0, 1].
        normalized = np.full(n_angles, np.nan, dtype=float)
        valid_angles = np.isfinite(aggregated)

        if np.any(valid_angles):
            minimum = float(np.min(aggregated[valid_angles]))
            maximum = float(np.max(aggregated[valid_angles]))

            if maximum == minimum:
                # Agreed handling for a constant profile.
                normalized[valid_angles] = 1.0
            else:
                normalized[valid_angles] = (
                    (aggregated[valid_angles] - minimum)
                    / (maximum - minimum)
                )

        processed_profiles.append(normalized)
        retained_interval_indices.append(interval_index)

    if processed_profiles:
        processed_doas = np.vstack(processed_profiles)
    else:
        processed_doas = np.empty((0, n_angles), dtype=float)

    # Motor-coordinate mapping: after all interval aggregation and
    # normalization are complete, reverse only the spatial-bin order.
    # Keep the angular coordinate labels and all raw-data plots unchanged.
    processed_doas = processed_doas[:, ::-1].copy()

    details = {
        "timestamps": timestamps,
        "spl": spl,
        "responses": responses,
        "intervals": interval_details,
        "retained_interval_indices": retained_interval_indices,
        "n_complete_intervals": n_complete_intervals,
        "T": T,
        "spl_threshold": spl_threshold,
    }

    return processed_doas, details


def process_doas(
    timestamps,
    spl,
    raw_responses,
    T=INTERVAL_SECONDS,
    spl_threshold=SPL_THRESHOLD,
):
    """Return one normalized, maximum-aggregated DOA profile per retained interval.

    Output shape:
        (number_of_retained_intervals, number_of_angular_bins)
    """

    processed_doas, _ = _process_doas_with_details(
        timestamps,
        spl,
        raw_responses,
        T=T,
        spl_threshold=spl_threshold,
    )
    return processed_doas


# ------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------

def make_time_edges(n_measurements):
    """Heatmap x-edges for columns representing individual measurements."""
    return np.arange(n_measurements + 1, dtype=float) - 0.5


def make_angular_mesh_data(angles, data_by_angle):
    """Return cyclic angular edges and data with the first bin repeated.

    ``angles`` are the actual angular-bin centres, in sorted signed-angle
    order.  The extra copy closes the 360-degree seam.  All heatmaps use this
    same helper so the response bins, peak markers and y-axis share one
    angular coordinate system.
    """
    angles = np.asarray(angles, dtype=float)
    data_by_angle = np.asarray(data_by_angle)

    if angles.ndim != 1 or len(angles) == 0:
        raise ValueError("angles must be a non-empty 1D array.")
    if data_by_angle.shape[0] != len(angles):
        raise ValueError(
            "The first data dimension must match the number of angles."
        )

    n_angles = len(angles)
    if n_angles == 1:
        step = 360.0
    else:
        step = 360.0 / n_angles
        expected = angles[0] + step * np.arange(n_angles)
        if not np.allclose(angles, expected, atol=1e-7, rtol=0.0):
            raise ValueError(
                "Angular bin centres must be uniformly spaced and sorted."
            )

    # Midpoint between the last centre (wrapped one revolution backwards)
    # and the first centre.  The final edge is exactly one revolution later.
    first_edge = (angles[-1] - 360.0 + angles[0]) / 2.0
    edges = first_edge + step * np.arange(n_angles + 2, dtype=float)
    closed_data = np.concatenate(
        [data_by_angle, data_by_angle[:1, ...]], axis=0
    )
    return edges, closed_data


def set_elapsed_time_ticks(ax, times, visible_start=0, visible_end=None):
    """Label heatmap measurement-index positions with elapsed timestamps."""

    if len(times) == 0:
        return

    if visible_end is None:
        visible_end = len(times) - 1

    first = max(0, int(np.ceil(visible_start)))
    last = min(len(times) - 1, int(np.floor(visible_end)))

    if last < first:
        return

    tick_count = min(8, last - first + 1)
    tick_indices = np.unique(
        np.linspace(first, last, tick_count).round().astype(int)
    )

    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f"{times[i]:.2f}" for i in tick_indices])


def plot_peak_markers(ax, times, peak1, peak2, x_positions=None):
    """Overlay recorded peak angles on a raw heatmap."""

    if x_positions is None:
        x_positions = np.arange(len(times), dtype=float)

    valid1 = np.isfinite(peak1)
    valid2 = np.isfinite(peak2)

    if np.any(valid1):
        ax.scatter(
            x_positions[valid1],
            peak1[valid1],
            marker="o",
            s=22,
            c="red",
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
            label="Recorded peak 1",
        )

    if np.any(valid2):
        ax.scatter(
            x_positions[valid2],
            peak2[valid2],
            marker="o",
            s=22,
            c="orange",
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
            label="Recorded peak 2",
        )


def draw_raw_heatmap(
    ax,
    times,
    response,
    angles,
    peak1=None,
    peak2=None,
    title="Raw DOA spatial response",
    color_label="Spatial response",
    display_response=None,
):
    """Draw a complete raw-data heatmap with signed angular coordinates."""

    n_measurements = len(times)
    n_angles = len(angles)

    if display_response is None:
        display_response = response

    if n_measurements == 0:
        ax.text(
            0.5, 0.5, "No measurements",
            ha="center", va="center", transform=ax.transAxes,
        )
        return None

    x_edges = make_time_edges(n_measurements)
    y_edges, angular_values = make_angular_mesh_data(
        angles, display_response.T
    )

    values = np.ma.masked_invalid(angular_values)

    finite_values = display_response[np.isfinite(display_response)]
    if finite_values.size:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if vmax == vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        values,
        shading="flat",
        cmap=COLOR_MAP,
        vmin=vmin,
        vmax=vmax,
    )

    if peak1 is not None and peak2 is not None:
        plot_peak_markers(
            ax,
            times,
            peak1,
            peak2,
        )

    ax.set_title(title)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("DOA angle (degrees)")
    ax.set_ylim(-180, 180)
    ax.set_yticks(np.arange(-180, 181, 45))
    set_elapsed_time_ticks(ax, times)

    return mesh


# ------------------------------------------------------------
# Main plotting application
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive raw and interval-processed DOA heatmaps."
        )
    )

    parser.add_argument(
        "csv",
        help="Path to the DOA discovery CSV",
    )

    parser.add_argument(
        "--db",
        action="store_true",
        help="Display raw spatial response as 10*log10(response). "
             "Processing and returned normalized DOAs use the raw values.",
    )

    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the initial raw heatmap image.",
    )

    args = parser.parse_args()

    data = load_csv(Path(args.csv))

    times = data["times"]
    spl = data["spl"]
    angles = data["angles"]
    response = data["response"]
    peak1 = data["peak1"]
    peak2 = data["peak2"]
    csv_path = data["csv_path"]

    n_measurements, n_angles = response.shape

    # Optional dB conversion is for raw-data display only.
    # The processing function receives the original response values.
    if args.db:
        with np.errstate(divide="ignore", invalid="ignore"):
            display_response = np.where(
                response > 0,
                10.0 * np.log10(response),
                np.nan,
            )
        raw_color_label = "Spatial response (dB, 10·log10)"
    else:
        display_response = response.copy()
        raw_color_label = "Spatial response"

    # Process the data once. The public function returns only the array;
    # details are retained here for the interval-selection visualization.
    processed_doas, processing_details = _process_doas_with_details(
        times,
        spl,
        data["response_original"],
        T=INTERVAL_SECONDS,
        spl_threshold=SPL_THRESHOLD,
    )

    retained_indices = processing_details["retained_interval_indices"]
    n_complete_intervals = processing_details["n_complete_intervals"]

    print("\nProcessing summary")
    print("-" * 60)
    print(f"Interval duration: {INTERVAL_SECONDS:g} seconds")
    print(f"Fixed SPL threshold: {SPL_THRESHOLD:g} dB")
    print(f"Complete intervals: {n_complete_intervals}")
    print(f"Retained intervals: {len(retained_indices)}")
    print(f"Processed DOA array shape: {processed_doas.shape}")

    # --------------------------------------------------------
    # FIGURE 1: Complete raw DOA heatmap
    # --------------------------------------------------------

    fig1, ax1 = plt.subplots(figsize=(14, 7))
    fig1.subplots_adjust(
        left=0.09, right=0.86, top=0.90, bottom=0.13
    )
    fig1.canvas.manager.set_window_title("Figure 1 — Raw DOA")

    mesh1 = draw_raw_heatmap(
        ax1,
        times,
        response,
        angles,
        peak1,
        peak2,
        title=f"Raw DOA spatial response — {csv_path.name}",
        color_label=raw_color_label,
        display_response=display_response,
    )

    if mesh1 is not None:
        cbar1 = fig1.colorbar(mesh1, ax=ax1, pad=0.02)
        cbar1.set_label(raw_color_label)

    if np.isfinite(peak1).any() or np.isfinite(peak2).any():
        ax1.legend(loc="upper right", fontsize=8)

    # --------------------------------------------------------
    # FIGURE 2: Full raw heatmap + selected interval detail
    # --------------------------------------------------------

    fig2 = plt.figure(figsize=(17, 8))
    fig2.canvas.manager.set_window_title(
        "Figure 2 — Interval selection and inspection"
    )

    grid = fig2.add_gridspec(
        1, 2,
        left=0.07,
        right=0.96,
        top=0.90,
        bottom=0.22,
        width_ratios=[1.35, 1.0],
        wspace=0.30,
    )

    full_ax = fig2.add_subplot(grid[0, 0])
    detail_ax = fig2.add_subplot(grid[0, 1])

    full_mesh = draw_raw_heatmap(
        full_ax,
        times,
        response,
        angles,
        peak1,
        peak2,
        title="Complete recording — select an interval below",
        color_label=raw_color_label,
        display_response=display_response,
    )

    if full_mesh is not None:
        cbar_full = fig2.colorbar(
            full_mesh, ax=full_ax, pad=0.02, fraction=0.045
        )
        cbar_full.set_label(raw_color_label)

    if np.isfinite(peak1).any() or np.isfinite(peak2).any():
        full_ax.legend(loc="upper right", fontsize=8)

    # Draw all interval boundaries on the complete raw heatmap.
    # The x-coordinate is a measurement index, so use searchsorted to map
    # elapsed-time boundaries to heatmap column positions.
    for boundary_index in range(n_complete_intervals + 1):
        boundary_time = boundary_index * INTERVAL_SECONDS
        boundary_column = np.searchsorted(
            times, boundary_time, side="left"
        ) - 0.5

        full_ax.axvline(
            boundary_column,
            color="white",
            linestyle="--",
            linewidth=0.9,
            alpha=0.85,
            zorder=7,
        )

    # Highlight the currently selected interval.
    selected_rectangle = Rectangle(
        (-0.5, -180.0),
        1.0,
        360.0,
        linewidth=2.2,
        edgecolor="red",
        facecolor="red",
        alpha=0.15,
        visible=False,
        zorder=8,
    )
    full_ax.add_patch(selected_rectangle)

    # Slider and information text.
    slider_ax = fig2.add_axes([0.16, 0.105, 0.68, 0.035])
    info_text = fig2.text(
        0.5, 0.045, "",
        ha="center",
        va="center",
        fontsize=9,
    )

    interval_slider = None

    if n_complete_intervals > 0:
        interval_slider = Slider(
            ax=slider_ax,
            label="Interval index",
            valmin=0,
            valmax=n_complete_intervals - 1,
            valinit=0,
            valstep=1,
        )
    else:
        slider_ax.set_visible(False)
        detail_ax.text(
            0.5, 0.5,
            "No complete intervals available",
            ha="center",
            va="center",
            transform=detail_ax.transAxes,
        )

    # --------------------------------------------------------
    # Update selected interval detail
    # --------------------------------------------------------

    detail_colorbar = None

    def update_selected_interval(value):
        nonlocal detail_colorbar

        if n_complete_intervals == 0:
            return

        interval_index = int(round(value))
        start = interval_index * INTERVAL_SECONDS
        end = (interval_index + 1) * INTERVAL_SECONDS

        # Half-open interval [start, end).
        in_interval = (times >= start) & (times < end)
        interval_indices = np.flatnonzero(in_interval)

        qualifying = in_interval & (spl >= SPL_THRESHOLD)
        qualifying_indices = np.flatnonzero(qualifying)

        # Update rectangle on full recording heatmap.
        x_start = np.searchsorted(times, start, side="left") - 0.5
        x_end = np.searchsorted(times, end, side="left") - 0.5

        # Keep the rectangle within the displayed recording extent.
        x_start = max(-0.5, x_start)
        x_end = min(n_measurements - 0.5, x_end)

        selected_rectangle.set_xy((x_start, -180.0))
        selected_rectangle.set_width(max(x_end - x_start, 0.05))
        selected_rectangle.set_visible(True)

        # Redraw the selected interval detail heatmap and replace its colorbar.
        if detail_colorbar is not None:
            detail_colorbar.remove()
            detail_colorbar = None

        detail_ax.clear()

        if interval_indices.size == 0:
            detail_ax.text(
                0.5, 0.5,
                "No measurements in this interval",
                ha="center",
                va="center",
                transform=detail_ax.transAxes,
            )
            n_pass = 0
        else:
            interval_display = display_response[interval_indices, :]
            local_x_edges = make_time_edges(len(interval_indices))
            y_edges, angular_values = make_angular_mesh_data(
                angles, interval_display.T
            )

            finite = interval_display[np.isfinite(interval_display)]
            if finite.size:
                local_vmin = float(np.min(finite))
                local_vmax = float(np.max(finite))
                if local_vmax == local_vmin:
                    local_vmax = local_vmin + 1.0
            else:
                local_vmin, local_vmax = 0.0, 1.0

            detail_mesh = detail_ax.pcolormesh(
                local_x_edges,
                y_edges,
                np.ma.masked_invalid(angular_values),
                shading="flat",
                cmap=COLOR_MAP,
                vmin=local_vmin,
                vmax=local_vmax,
            )

            # Lightly shade measurements that pass/fail the fixed SPL test.
            local_spl = spl[interval_indices]
            for local_index, passes in enumerate(
                local_spl >= SPL_THRESHOLD
            ):
                detail_ax.axvspan(
                    local_index - 0.5,
                    local_index + 0.5,
                    color=("limegreen" if passes else "red"),
                    alpha=0.055,
                    linewidth=0,
                )

            # Overlay the recorded peak directions for this interval.
            local_peak1 = peak1[interval_indices]
            local_peak2 = peak2[interval_indices]
            plot_peak_markers(
                detail_ax,
                times[interval_indices],
                local_peak1,
                local_peak2,
                x_positions=np.arange(len(interval_indices), dtype=float),
            )

            detail_ax.set_xlim(
                -0.5, len(interval_indices) - 0.5
            )

            detail_ax.set_xticks(
                np.arange(len(interval_indices))
            )

            # Limit tick count while retaining the elapsed-time labels.
            tick_step = max(1, int(np.ceil(len(interval_indices) / 8)))
            tick_positions = np.arange(
                0, len(interval_indices), tick_step
            )
            detail_ax.set_xticks(tick_positions)
            detail_ax.set_xticklabels([
                f"{times[interval_indices[i]]:.2f}"
                for i in tick_positions
            ])

            n_pass = int(np.sum(local_spl >= SPL_THRESHOLD))

            detail_ax.legend(
                handles=[
                    Line2D(
                        [0], [0], color="limegreen", linewidth=5,
                        alpha=0.5, label=f"SPL >= {SPL_THRESHOLD:g} dB",
                    ),
                    Line2D(
                        [0], [0], color="red", linewidth=5,
                        alpha=0.5, label=f"SPL < {SPL_THRESHOLD:g} dB",
                    ),
                    Line2D(
                        [0], [0], marker="o", linestyle="None",
                        markerfacecolor="red", markeredgecolor="black",
                        label="Recorded peak 1",
                    ),
                    Line2D(
                        [0], [0], marker="o", linestyle="None",
                        markerfacecolor="orange", markeredgecolor="black",
                        label="Recorded peak 2",
                    ),
                ],
                loc="upper right",
                fontsize=7,
            )

            detail_colorbar = fig2.colorbar(
                detail_mesh,
                ax=detail_ax,
                pad=0.02,
                fraction=0.05,
            )
            detail_colorbar.set_label(raw_color_label)

        detail_ax.set_title(
            f"Interval {interval_index}: [{start:.1f}, {end:.1f}) s\n"
            f"{n_pass}/{len(interval_indices)} measurements pass SPL"
        )
        detail_ax.set_xlabel("Measurement index within interval (labels = seconds)")
        detail_ax.set_ylabel("DOA angle (degrees)")
        detail_ax.set_ylim(-180, 180)
        detail_ax.set_yticks(np.arange(-180, 181, 45))

        info_text.set_text(
            f"Selected interval: {interval_index}  |  "
            f"Time: [{start:.1f}, {end:.1f}) s  |  "
            f"Measurements: {len(interval_indices)}  |  "
            f"Passing SPL >= {SPL_THRESHOLD:g} dB: {n_pass}"
        )

        fig2.canvas.draw_idle()

    if interval_slider is not None:
        interval_slider.on_changed(update_selected_interval)
        update_selected_interval(0)

    # --------------------------------------------------------
    # FIGURE 3: Processed and normalized DOA profiles
    # --------------------------------------------------------

    fig3, ax3 = plt.subplots(figsize=(14, 7))
    fig3.subplots_adjust(
        left=0.09, right=0.86, top=0.90, bottom=0.13
    )
    fig3.canvas.manager.set_window_title(
        "Figure 3 — Processed normalized DOAs"
    )

    if processed_doas.shape[0] == 0:
        ax3.text(
            0.5, 0.5,
            "No intervals contain measurements at or above "
            f"{SPL_THRESHOLD:g} dB SPL",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("Processed DOAs")
        ax3.set_xlabel("Retained interval")
        ax3.set_ylabel("DOA angle (degrees)")
        ax3.set_ylim(-180, 180)
    else:
        processed_x_edges = (
            np.arange(processed_doas.shape[0] + 1, dtype=float) - 0.5
        )
        processed_y_edges, processed_angular_values = (
            make_angular_mesh_data(angles, processed_doas.T)
        )

        processed_mesh = ax3.pcolormesh(
            processed_x_edges,
            processed_y_edges,
            np.ma.masked_invalid(processed_angular_values),
            shading="flat",
            cmap=COLOR_MAP,
            vmin=0.0,
            vmax=1.0,
        )

        tick_step = max(
            1,
            int(np.ceil(processed_doas.shape[0] / 12)),
        )
        tick_positions = np.arange(
            0, processed_doas.shape[0], tick_step
        )

        ax3.set_xticks(tick_positions)
        ax3.set_xticklabels([
            (
                f"{retained_indices[i]} "
                f"({retained_indices[i] * INTERVAL_SECONDS:.0f}s)"
            )
            for i in tick_positions
        ])

        cbar3 = fig3.colorbar(
            processed_mesh,
            ax=ax3,
            pad=0.02,
        )
        cbar3.set_label("Per-interval normalized response [0, 1]")

        ax3.set_title(
            f"Maximum-aggregated and normalized DOAs — "
            f"{csv_path.name}"
        )
        ax3.set_xlabel(
            "Retained interval (original interval index and start time)"
        )
        ax3.set_ylabel("DOA angle (degrees)")
        ax3.set_ylim(-180, 180)
        ax3.set_yticks(np.arange(-180, 181, 45))

    # --------------------------------------------------------
    # Optional image saving
    # --------------------------------------------------------

    if args.save:
        fig1.savefig(
            args.save,
            dpi=200,
            bbox_inches="tight",
        )
        print(f"Saved raw heatmap to: {args.save}")

    # Keep the processed array available to callers that invoke main()
    # programmatically, as well as through the standalone process_doas().
    plt.show()

    return processed_doas


if __name__ == "__main__":
    main()
