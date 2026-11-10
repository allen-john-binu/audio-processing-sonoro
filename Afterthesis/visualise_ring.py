
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# CONFIGURATION
# ============================================================

N_UNITS = 120
ANGLE_STEP = 3.0

# Four bins = 12 degrees
PEAK_SEPARATION_BINS = 4

# Minimum prominence of a local maximum (in hext units).
# Adjust this if too many small fluctuations are detected.
MIN_PEAK_PROMINENCE = 0.002

OUTPUT_FILE = "ring_visualisation.png"


# ============================================================
# DATA LOADING
# ============================================================

def get_ordered_columns(df, prefix, expected_count):
    """Get numbered columns in numerical order."""

    pattern = re.compile(rf"^{prefix}_(\d+)$")

    columns = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            columns.append((int(match.group(1)), col))

    columns.sort(key=lambda x: x[0])

    indices = [x[0] for x in columns]

    if indices != list(range(expected_count)):
        raise ValueError(
            f"Expected {prefix}_0 to {prefix}_{expected_count - 1}, "
            f"but found indices: {indices}"
        )

    return [x[1] for x in columns]


def load_data(filename):

    df = pd.read_csv(filename)

    spin_cols = get_ordered_columns(df, "spin", N_UNITS)
    hext_cols = get_ordered_columns(df, "hext", N_UNITS)

    spins_history = df[spin_cols].to_numpy(dtype=float)
    hext_history = df[hext_cols].to_numpy(dtype=float)

    bump_angles = df["bump_angle"].to_numpy(dtype=float)

    if not (
        np.isfinite(spins_history).all()
        and np.isfinite(hext_history).all()
        and np.isfinite(bump_angles).all()
    ):
        raise ValueError("The CSV contains missing or non-finite values.")

    return spins_history, hext_history, bump_angles


# ============================================================
# CIRCULAR PEAK DETECTION
# ============================================================

def find_circular_peaks(profile, min_prominence=MIN_PEAK_PROMINENCE):
    """
    Find local maxima in a periodic angular profile.

    Tiling the profile three times allows scipy to detect peaks
    correctly across the -180 / +180 degree boundary.
    """

    n = len(profile)

    extended_profile = np.tile(profile, 3)

    peaks, properties = find_peaks(
        extended_profile,
        prominence=min_prominence
    )

    # Keep only peaks from the middle copy
    peaks = peaks[(peaks >= n) & (peaks < 2 * n)]

    # Convert back to original indices
    peaks = peaks - n

    return peaks


def detect_two_peaks(profile):
    """
    Find a pair of local maxima separated by exactly 4 bins.

    If a suitable pair does not exist:
        - Return the strongest peak as the red marker.
        - Return NaN for the orange marker.

    If a pair exists:
        - Select the pair with the largest combined amplitude.
        - Red = stronger peak.
        - Orange = weaker peak.
    """

    n = len(profile)

    local_peaks = find_circular_peaks(profile)

    # Fallback: strongest value in the profile
    strongest_peak = int(np.nanargmax(profile))

    if len(local_peaks) == 0:
        return strongest_peak, None

    # Find all pairs with the requested circular separation
    candidates = []

    for i in range(len(local_peaks)):
        for j in range(i + 1, len(local_peaks)):

            p1 = int(local_peaks[i])
            p2 = int(local_peaks[j])

            # Circular distance
            distance = min(
                (p2 - p1) % n,
                (p1 - p2) % n
            )

            if distance == PEAK_SEPARATION_BINS:

                combined_height = profile[p1] + profile[p2]

                candidates.append(
                    (combined_height, p1, p2)
                )

    # No suitable pair: mark only one peak
    if not candidates:
        return strongest_peak, None

    # Select the pair with the largest combined amplitude
    _, p1, p2 = max(candidates, key=lambda x: x[0])

    # Stronger peak is red
    if profile[p1] >= profile[p2]:
        return p1, p2
    else:
        return p2, p1


# ============================================================
# VISUALISATION
# ============================================================

def visualise(spins_history, hext_history, bump_angles):

    n_time = spins_history.shape[0]
    n_units = spins_history.shape[1]

    time_axis = np.arange(n_time)

    # Angular coordinate for all 120 units:
    # -180, -177, ..., 174, 177
    angles = -180 + ANGLE_STEP * np.arange(n_units)

    # Image boundaries: place each angular value at its centre
    x_extent = [-0.5, n_time - 0.5]
    y_extent = [
        angles[0] - ANGLE_STEP / 2,
        angles[-1] + ANGLE_STEP / 2
    ]

    extent = [
        x_extent[0],
        x_extent[1],
        y_extent[0],
        y_extent[1]
    ]

    # --------------------------------------------------------
    # Detect peaks at every time step
    # --------------------------------------------------------

    red_angles = np.full(n_time, np.nan)
    orange_angles = np.full(n_time, np.nan)

    for t in range(n_time):

        red_idx, orange_idx = detect_two_peaks(
            hext_history[t, :]
        )

        red_angles[t] = angles[red_idx]

        if orange_idx is not None:
            orange_angles[t] = angles[orange_idx]

    # --------------------------------------------------------
    # Create two vertically stacked plots
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 1.2],
            "hspace": 0.12
        }
    )

    ax_spin, ax_hext = axes

    # ========================================================
    # PANEL 1: SPIN RASTER
    # ========================================================

    im_spin = ax_spin.imshow(
        spins_history.T,
        aspect="auto",
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        extent=extent,
        vmin=0,
        vmax=1
    )

    # Overlay bump angle
    ax_spin.plot(
        time_axis,
        bump_angles,
        color="red",
        linewidth=2,
        label="Bump angle",
        zorder=3
    )

    ax_spin.set_title(
        "Network spins over time",
        fontsize=13
    )

    ax_spin.set_ylabel("Angle (degrees)")

    ax_spin.set_ylim(-180, 180)

    ax_spin.set_yticks(
        np.arange(-180, 181, 60)
    )

    ax_spin.legend(loc="upper right")

    # ========================================================
    # PANEL 2: EXTERNAL FIELD HEATMAP
    # ========================================================

    im_hext = ax_hext.imshow(
        hext_history.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=extent
    )

    # First peak: RED
    ax_hext.scatter(
        time_axis,
        red_angles,
        color="red",
        s=35,
        edgecolors="white",
        linewidths=0.5,
        label="First peak",
        zorder=4
    )

    # Second peak: ORANGE
    # NaN values are automatically omitted
    ax_hext.scatter(
        time_axis,
        orange_angles,
        color="orange",
        s=35,
        edgecolors="white",
        linewidths=0.5,
        label="Second peak",
        zorder=4
    )

    ax_hext.set_title(
        "External field and detected peaks",
        fontsize=13
    )

    ax_hext.set_ylabel("Angle (degrees)")
    ax_hext.set_xlabel("Time step")

    ax_hext.set_ylim(-180, 180)

    ax_hext.set_yticks(
        np.arange(-180, 181, 60)
    )

    ax_hext.legend(loc="upper right")

    # ========================================================
    # SHARED X AXIS AND COLORBAR
    # ========================================================

    ax_hext.set_xlim(-0.5, n_time - 0.5)

    ax_hext.set_xticks(
        np.arange(0, n_time, max(1, n_time // 10))
    )

    # External field colorbar
    cbar = fig.colorbar(
        im_hext,
        ax=axes,
        pad=0.02,
        fraction=0.025
    )

    cbar.set_label("External field (hext)")

    fig.align_ylabels(axes)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # --------------------------------------------------------
    # Print detected peak information
    # --------------------------------------------------------

    print("\nPeak detection results")
    print("----------------------")

    for t in range(n_time):

        if np.isnan(orange_angles[t]):

            print(
                f"Step {t:3d}: "
                f"Peak 1 = {red_angles[t]:7.1f}°, "
                f"Peak 2 = not detected"
            )

        else:

            separation = abs(
                (orange_angles[t] - red_angles[t] + 180) % 360 - 180
            )

            print(
                f"Step {t:3d}: "
                f"Peak 1 = {red_angles[t]:7.1f}°, "
                f"Peak 2 = {orange_angles[t]:7.1f}°, "
                f"Separation = {separation:5.1f}°"
            )

    print(f"\nFigure saved as: {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Visualise ring network spins and external fields."
    )

    parser.add_argument(
        "csv",
        help="Path to the ring statistics CSV file"
    )

    args = parser.parse_args()

    spins_history, hext_history, bump_angles = load_data(args.csv)

    visualise(
        spins_history,
        hext_history,
        bump_angles
    )
