from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing the 25 CSV files
DATA_DIR = Path("./mergedData/")

# Output directory
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

# DOA columns to use
DOA_COLUMNS = [f"doa_{i}" for i in range(1, 120)]

# Peak detection parameters
#
# Adjust these if necessary.
#
# prominence:
#   Minimum prominence for a point to be considered a peak.
#   Set to 0 if you want essentially every local maximum.
#
# distance:
#   Minimum distance between two detected peaks, in DOA bins.
#
PEAK_PROMINENCE = 0.0
PEAK_DISTANCE = 1

# ============================================================
# FUNCTIONS
# ============================================================

def calculate_peak_difference(row):
    """
    Find the two highest LOCAL peaks in doa_1 ... doa_119.

    Returns:
        intensity_difference = highest_peak - second_highest_peak

    If fewer than two peaks are found, returns NaN.
    """

    values = row[DOA_COLUMNS].to_numpy(dtype=float)

    # Find local maxima
    peaks, properties = find_peaks(
        values,
        prominence=PEAK_PROMINENCE,
        distance=PEAK_DISTANCE
    )

    # Need at least two peaks
    if len(peaks) < 2:
        return np.nan

    # Intensities at detected peaks
    peak_values = values[peaks]

    # Sort peaks by intensity, highest first
    sorted_indices = np.argsort(peak_values)[::-1]

    highest = peak_values[sorted_indices[0]]
    second_highest = peak_values[sorted_indices[1]]

    return highest - second_highest


def process_file(csv_path):
    """
    Process one CSV file and return a dataframe containing:

        row_index
        peak_difference
    """

    df = pd.read_csv(csv_path)

    # Check that all required DOA columns exist
    missing = [col for col in DOA_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing DOA columns: {missing}"
        )

    # Calculate difference for every row
    differences = df.apply(calculate_peak_difference, axis=1)

    result = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "peak_difference": differences
    })

    return result


# ============================================================
# PROCESS ALL FILES
# ============================================================

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {DATA_DIR}"
    )

print(f"Found {len(csv_files)} CSV files")

all_results = {}

for csv_path in csv_files:

    print(f"Processing: {csv_path.name}")

    result = process_file(csv_path)

    all_results[csv_path.name] = result

    # Save individual processed result
    output_file = OUTPUT_DIR / f"{csv_path.stem}_peak_difference.csv"
    result.to_csv(output_file, index=False)


# ============================================================
# GROUP FILES INTO EXPERIMENTS
# ============================================================

experiments = {
    "AA": [],
    "AB": [],
    "AC": [],
    "AD": [],
    "AE": [],
}

for filename, result in all_results.items():

    # Example:
    # 18_08_AA_01.csv
    # 18_08_AB_05.csv

    parts = Path(filename).stem.split("_")

    # Expected:
    # ["18", "08", "AA", "01"]
    experiment = parts[2]

    if experiment in experiments:
        experiments[experiment].append(result)

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(15, 8))

# Fixed colour for each experiment
experiment_colors = {
    "AA": "tab:blue",
    "AB": "tab:orange",
    "AC": "tab:green",
    "AD": "tab:red",
    "AE": "tab:purple",
}

for experiment, results in experiments.items():

    color = experiment_colors[experiment]

    for i, result in enumerate(results):

        # Select only row indices 10 through 30, inclusive
        plot_data = result[
            (result["row_index"] >= 10) &
            (result["row_index"] <= 30)
        ]

        ax.plot(
            plot_data["row_index"],
            plot_data["peak_difference"],
            color=color,
            linewidth=1.5,
            alpha=0.8,

            # Only show experiment once in legend
            label=experiment if i == 0 else "_nolegend_"
        )


ax.set_xlabel("Row index")
ax.set_ylabel(
    "Intensity difference between two highest DOA peaks"
)

ax.set_title(
    "DOA Peak Intensity Difference by Experiment"
)

ax.grid(True, alpha=0.3)
ax.legend(title="Experiment")

plt.tight_layout()

plot_file = OUTPUT_DIR / "doa_peak_difference.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")

plt.show()