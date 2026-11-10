from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("./mergedData/")
RESULT_FILE = Path("./result.csv")

OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_FILE = OUTPUT_DIR / "doa_selected_peak_difference.pdf"

# result.csv contains doa_0 ... doa_119
DOA_COLUMNS = [
    f"doa_{i}"
    for i in range(120)
]

PEAK_PROMINENCE = 0.0
PEAK_DISTANCE = 1

# Exactly 3 selected points per source file
POINTS_PER_FILE = 3

FILENAME_FONT_SIZE = 5


# ============================================================
# PREFIX -> COLOUR
# ============================================================

PREFIX_COLORS = {
    "AA": "blue",
    "AB": "orange",
    "AC": "green",
    "AD": "purple",
    "AE": "red",
    "AF": "brown",
    "AG": "pink",
    "AH": "gray",
    "AI": "olive",
}

legend_map = {
    "AA": 0.00,
    "AB": 0.05,
    "AC": 0.10,
    "AD": 0.15,
    "AE": 0.20,
    "AF": 0.25,
    "AG": 0.30,
    "AH": 0.35,
    "AI": 0.40,
}

def get_prefix_color(prefix):
    prefix = str(prefix).strip()
    experiment = prefix[-2:]

    if experiment not in PREFIX_COLORS:
        raise ValueError(
            f"Unknown experiment code {experiment!r} "
            f"from prefix {prefix!r}."
        )

    return PREFIX_COLORS[experiment]


# ============================================================
# CALCULATE PEAK DIFFERENCE
# ============================================================

def calculate_peak_difference(row):
    """
    Find the two highest LOCAL peaks in doa_0 ... doa_119.

    Returns:
        highest_peak - second_highest_peak

    If fewer than two peaks are found:
        NaN
    """

    values = row[DOA_COLUMNS].to_numpy(dtype=float)

    if np.all(np.isnan(values)):
        return np.nan

    values = values.copy()
    values[np.isnan(values)] = -np.inf

    peaks, _ = find_peaks(
        values,
        prominence=PEAK_PROMINENCE,
        distance=PEAK_DISTANCE,
    )

    if len(peaks) < 2:
        return np.nan

    peak_values = values[peaks]
    sorted_indices = np.argsort(peak_values)[::-1]

    highest = peak_values[sorted_indices[0]]
    second_highest = peak_values[sorted_indices[1]]

    return highest - second_highest


# ============================================================
# LOAD RESULT.CSV
# ============================================================

if not RESULT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n\n{RESULT_FILE}"
    )

print("=" * 70)
print("DOA SELECTED-POINT PEAK-DIFFERENCE PLOT")
print("=" * 70)
print(f"Reading:\n{RESULT_FILE}")

df = pd.read_csv(RESULT_FILE)

required_columns = [
    "filename",
    "prefix",
    *DOA_COLUMNS,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns in result.csv:\n\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# GET ALL SOURCE FILES
# ============================================================

source_files = (
    df["filename"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .tolist()
)

print(f"\nUnique source files: {len(source_files)}")

print(f"Using ALL {len(source_files)} source files.")


# ============================================================
# BUILD PLOT DATA
# ============================================================

plot_records = []

for source_file in source_files:

    file_data = df[
        df["filename"].astype(str) == source_file
    ].copy()

    if len(file_data) != POINTS_PER_FILE:
        print(
            f"\nWARNING: {source_file}: "
            f"{len(file_data)} rows found, "
            f"expected {POINTS_PER_FILE}."
        )

    # Preserve result.csv row order.
    file_data = file_data.reset_index(drop=True)

    # Only the first 3 rows are the three selected points.
    # If there are more rows, do not silently plot them.
    file_data = file_data.iloc[:POINTS_PER_FILE]

    for point_number, (_, row) in enumerate(
        file_data.iterrows(),
        start=1,
    ):
        difference = calculate_peak_difference(row)

        plot_records.append({
            "source_file": source_file,
            "prefix": str(row["prefix"]).strip(),
            "point": point_number,
            "peak_difference": difference,
        })


plot_df = pd.DataFrame(plot_records)


# ============================================================
# PRINT DATA SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PLOT DATA")
print("=" * 70)

for source_file in source_files:

    file_data = plot_df[
        plot_df["source_file"] == source_file
    ].sort_values("point")

    print(f"\n{source_file}")

    for _, row in file_data.iterrows():

        difference = row["peak_difference"]

        if pd.isna(difference):
            value_text = "NaN"
        else:
            value_text = f"{difference:.6f}"

        print(
            f"    Point {int(row['point'])}: {value_text}"
        )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(17, 9))


# ============================================================
# PLOT EACH SOURCE FILE
# ============================================================

for source_file in source_files:

    file_data = plot_df[
        plot_df["source_file"] == source_file
    ].sort_values("point").copy()

    if file_data.empty:
        continue

    # IMPORTANT:
    # Plot the three points individually using point = 1, 2, 3.
    valid_data = file_data[
        file_data["peak_difference"].notna()
    ].copy()

    if valid_data.empty:
        continue

    prefix = str(file_data["prefix"].iloc[0]).strip()
    color = get_prefix_color(prefix)

    x = valid_data["point"].to_numpy(dtype=float)
    y = valid_data["peak_difference"].to_numpy(dtype=float)

    # Explicitly plot the 3 selected points and connect them.
    ax.plot(
        x,
        y,
        color=color,
        linewidth=1.2,
        marker="o",
        markersize=5,
        alpha=0.8,
    )

    # # Filename at the first plotted point.
    # first_point = valid_data.iloc[0]

    # ax.annotate(
    #     source_file,
    #     xy=(
    #         first_point["point"],
    #         first_point["peak_difference"],
    #     ),
    #     xytext=(-5, 0),
    #     textcoords="offset points",
    #     fontsize=FILENAME_FONT_SIZE,
    #     color=color,
    #     va="center",
    #     ha="right",
    #     alpha=0.9,
    # )

    # # Filename at the last plotted point.
    # last_point = valid_data.iloc[-1]

    # ax.annotate(
    #     source_file,
    #     xy=(
    #         last_point["point"],
    #         last_point["peak_difference"],
    #     ),
    #     xytext=(5, 0),
    #     textcoords="offset points",
    #     fontsize=FILENAME_FONT_SIZE,
    #     color=color,
    #     va="center",
    #     ha="left",
    #     alpha=0.9,
    # )


# ============================================================
# AXIS SETTINGS
# ============================================================

ax.set_xlabel(
    "Selected point",
    fontsize=12,
)

ax.set_ylabel(
    "Intensity difference between two highest DOA peaks",
    fontsize=12,
)

ax.set_title(
    "DOA Peak Intensity Difference "
    "for Three Selected Points per File",
    fontsize=14,
)

# Exactly 1, 2, 3 on the x-axis.
ax.set_xlim(0.5, 3.5)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["1", "2", "3"])

ax.grid(True, alpha=0.3)


# ============================================================
# EXPERIMENT LEGEND
# ============================================================

legend_handles = []

for experiment, color in PREFIX_COLORS.items():

    handle, = ax.plot(
        [],
        [],
        color=color,
        linewidth=2,
        marker="o",
        markersize=4,
        label=f"{legend_map[experiment]:.2f}",
    )

    legend_handles.append(handle)

ax.legend(
    handles=legend_handles,
    title="Absolute amplitude difference",
    loc="upper left",
)


# ============================================================
# DISPLAY
# ============================================================

fig.tight_layout()

plt.show()


# ============================================================
# SAVE FINAL VIEW
# ============================================================

fig.savefig(
    PLOT_FILE,
    dpi=300,
    bbox_inches="tight",
    format="pdf",
)

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)
print(f"Plot saved to:\n{PLOT_FILE}")
print("=" * 70)
