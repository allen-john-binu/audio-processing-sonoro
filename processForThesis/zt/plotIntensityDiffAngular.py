import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# Configuration
# ============================================================

CSV_FILE = "./mergedData/result.csv"

DOA_COLUMNS = [
    f"doa_{i}"
    for i in range(120)
]

PEAK_PROMINENCE = 0.01       # <-- use your actual value
PEAK_DISTANCE = 5            # <-- use your actual value


# ============================================================
# Peak difference
# ============================================================

def calculate_peak_difference(row):
    """
    Find the two highest LOCAL peaks in doa_0 ... doa_119.

    Returns:
        highest_peak - second_highest_peak

    If fewer than two peaks are found:
        NaN
    """

    values = row[
        DOA_COLUMNS
    ].to_numpy(dtype=float)

    # Check for completely empty DOA data
    if np.all(np.isnan(values)):
        return np.nan

    values = values.copy()

    # NaN cannot be a peak
    values[np.isnan(values)] = -np.inf

    # Find local peaks
    peaks, properties = find_peaks(
        values,
        prominence=PEAK_PROMINENCE,
        distance=PEAK_DISTANCE,
    )

    # Need at least two peaks
    if len(peaks) < 2:
        return np.nan

    # Get peak intensities
    peak_values = values[peaks]

    # Highest first
    sorted_indices = np.argsort(peak_values)[::-1]

    highest = peak_values[sorted_indices[0]]
    second_highest = peak_values[sorted_indices[1]]

    return highest - second_highest


# ============================================================
# Angular separation
# ============================================================

def calculate_angle_separation(row):
    """
    Calculate angular separation as the absolute difference
    between angle_0 and angle_1.

    Example:
        angle_0 = -45
        angle_1 = 54

        separation = 99 degrees
    """

    angle_0 = row["angle_0"]
    angle_1 = row["angle_1"]

    if pd.isna(angle_0) or pd.isna(angle_1):
        return np.nan

    return abs(angle_0 - angle_1)


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(CSV_FILE)


# ============================================================
# Calculate quantities
# ============================================================

df["peak_difference"] = df.apply(
    calculate_peak_difference,
    axis=1,
)

df["angle_separation"] = df.apply(
    calculate_angle_separation,
    axis=1,
)


# ============================================================
# Keep only valid points
# ============================================================

plot_df = df[
    df["peak_difference"].notna()
    & df["angle_separation"].notna()
].copy()


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    plot_df["peak_difference"],
    plot_df["angle_separation"],
    alpha=0.6,
)

plt.xlabel("Peak Difference")
plt.ylabel("Angle of Separation (degrees)")

plt.title(
    "Peak Difference vs Angle of Separation"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()
plt.show()