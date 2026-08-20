from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from scipy.signal import find_peaks


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("./mergedData/")

RESULT_FILE = DATA_DIR / "result.csv"

OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_FILE = OUTPUT_DIR / "doa_selected_peak_difference.png"


# ------------------------------------------------------------
# DOA columns
#
# Keep the original behavior:
# doa_1 ... doa_119
# ------------------------------------------------------------

DOA_COLUMNS = [
    f"doa_{i}"
    for i in range(1, 120)
]


# ------------------------------------------------------------
# Peak detection
# ------------------------------------------------------------

PEAK_PROMINENCE = 0.0
PEAK_DISTANCE = 1


# ------------------------------------------------------------
# Five selected points per source file
# ------------------------------------------------------------

POINTS_PER_FILE = 5


# ------------------------------------------------------------
# Filename font size
# ------------------------------------------------------------

FILENAME_FONT_SIZE = 5


# ============================================================
# EXPERIMENT COLOURS
# ============================================================
#
# One fixed colour per experiment type.
#
# Every source file belonging to the same experiment gets
# the same colour.
#
# Example:
#
#   16_08_AF_01.csv -> AF -> tab:brown
#   16_08_AF_02.csv -> AF -> tab:brown
#   16_08_AF_03.csv -> AF -> tab:brown
#
# etc.
# ============================================================

EXPERIMENT_COLORS = {
    "AA": "tab:blue",
    "AB": "tab:orange",
    "AC": "tab:green",
    "AD": "tab:red",
    "AE": "tab:purple",
    "AF": "tab:brown",
    "AG": "tab:pink",
    "AH": "tab:gray",
    "AI": "olive",
}


# ============================================================
# CALCULATE PEAK DIFFERENCE
# ============================================================

def calculate_peak_difference(row):
    """
    Find the two highest LOCAL peaks in doa_1 ... doa_119.

    Returns:
        highest_peak - second_highest_peak

    If fewer than two peaks are found:
        NaN
    """

    values = row[
        DOA_COLUMNS
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Check for completely empty DOA data.
    # --------------------------------------------------------

    if np.all(np.isnan(values)):
        return np.nan

    values = values.copy()

    # --------------------------------------------------------
    # NaN cannot be a peak.
    # --------------------------------------------------------

    values[
        np.isnan(values)
    ] = -np.inf

    # --------------------------------------------------------
    # Find local peaks.
    # --------------------------------------------------------

    peaks, properties = find_peaks(
        values,
        prominence=PEAK_PROMINENCE,
        distance=PEAK_DISTANCE,
    )

    # --------------------------------------------------------
    # Need at least two peaks.
    # --------------------------------------------------------

    if len(peaks) < 2:
        return np.nan

    # --------------------------------------------------------
    # Get peak intensities.
    # --------------------------------------------------------

    peak_values = values[peaks]

    # Highest first.
    sorted_indices = np.argsort(
        peak_values
    )[::-1]

    highest = peak_values[
        sorted_indices[0]
    ]

    second_highest = peak_values[
        sorted_indices[1]
    ]

    return (
        highest - second_highest
    )


# ============================================================
# GET EXPERIMENT NAME
# ============================================================

def get_experiment_name(source_file):
    """
    Extract experiment name from filename.

    Example:

        16_08_AF_03.csv

    returns:

        AF
    """

    stem = Path(
        source_file
    ).stem

    parts = stem.split("_")

    if len(parts) >= 3:

        experiment = parts[2]

        if experiment in EXPERIMENT_COLORS:
            return experiment

    return "OTHER"


# ============================================================
# LOAD RESULT.CSV
# ============================================================

if not RESULT_FILE.exists():

    raise FileNotFoundError(
        f"Could not find:\n\n{RESULT_FILE}"
    )


print("=" * 70)

print(
    "DOA SELECTED-POINT PEAK-DIFFERENCE PLOT"
)

print("=" * 70)

print(
    f"Reading:\n{RESULT_FILE}"
)


df = pd.read_csv(
    RESULT_FILE
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "source_file",
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
# GET SOURCE FILES
# ============================================================
#
# IMPORTANT:
#
# We now use ALL source files.
#
# The previous version used only the last 25 files.
# That restriction has been removed.
# ============================================================

source_files = (
    df["source_file"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .tolist()
)


print(
    f"\nUnique source files: "
    f"{len(source_files)}"
)


# ============================================================
# CHECK EXPERIMENT TYPES
# ============================================================

experiments_found = []

for source_file in source_files:

    experiment = get_experiment_name(
        source_file
    )

    if experiment not in experiments_found:
        experiments_found.append(
            experiment
        )


print(
    "\nExperiment types found:"
)

for experiment in experiments_found:

    print(
        f"    {experiment}"
    )


# ------------------------------------------------------------
# Warn about unexpected experiment types.
# ------------------------------------------------------------

unexpected_experiments = [
    experiment
    for experiment in experiments_found
    if experiment not in EXPERIMENT_COLORS
]


if unexpected_experiments:

    print(
        "\nWARNING:"
    )

    print(
        "The following experiment types do not "
        "have a defined colour:"
    )

    for experiment in unexpected_experiments:

        print(
            f"    {experiment}"
        )


# ============================================================
# BUILD PLOT DATA
# ============================================================

plot_records = []


for source_file in source_files:

    file_data = df[
        df["source_file"].astype(str)
        == source_file
    ].copy()


    # --------------------------------------------------------
    # We expect exactly five rows per source file.
    # --------------------------------------------------------

    if len(file_data) != POINTS_PER_FILE:

        print(
            "\nWARNING:"
        )

        print(
            f"{source_file}: "
            f"{len(file_data)} rows found, "
            f"expected {POINTS_PER_FILE}."
        )


    # --------------------------------------------------------
    # Preserve result.csv order.
    #
    # first row  -> point 1
    # second row -> point 2
    # third row  -> point 3
    # fourth row -> point 4
    # fifth row  -> point 5
    #
    # These five points form ONE LINE for the source file.
    # --------------------------------------------------------

    file_data = file_data.reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Calculate DOA peak difference for every row.
    # --------------------------------------------------------

    for point_number, (_, row) in enumerate(
        file_data.iterrows(),
        start=1,
    ):

        difference = (
            calculate_peak_difference(row)
        )


        plot_records.append({

            "source_file": source_file,

            "experiment": get_experiment_name(
                source_file
            ),

            "point": point_number,

            "peak_difference": difference,

        })


plot_df = pd.DataFrame(
    plot_records
)


# ============================================================
# PRINT DATA SUMMARY
# ============================================================

print("\n")

print("=" * 70)

print("PLOT DATA")

print("=" * 70)


for source_file in source_files:

    file_data = plot_df[
        plot_df["source_file"]
        == source_file
    ]


    print(
        f"\n{source_file}"
    )


    experiment = get_experiment_name(
        source_file
    )


    print(
        f"    Experiment: {experiment}"
    )


    for _, row in file_data.iterrows():

        difference = row[
            "peak_difference"
        ]


        if pd.isna(difference):

            value_text = "NaN"

        else:

            value_text = (
                f"{difference:.6f}"
            )


        print(
            f"    Point "
            f"{int(row['point'])}: "
            f"{value_text}"
        )


# ============================================================
# CREATE FIGURE
# ============================================================

fig = plt.figure(
    figsize=(17, 12)
)


# ============================================================
# MAIN PLOT AXES
# ============================================================

ax = fig.add_axes([
    0.08,
    0.10,
    0.66,
    0.82,
])


# ============================================================
# CHECKBOX AXES
# ============================================================

check_ax = fig.add_axes([
    0.77,
    0.18,
    0.20,
    0.70,
])


# ============================================================
# BUTTON AXES
# ============================================================

select_all_ax = fig.add_axes([
    0.77,
    0.09,
    0.095,
    0.055,
])


deselect_all_ax = fig.add_axes([
    0.875,
    0.09,
    0.095,
    0.055,
])


# ============================================================
# STORE LINE OBJECTS
# ============================================================

line_objects = {}


# ------------------------------------------------------------
# Store annotation objects too.
# ------------------------------------------------------------

left_labels = {}

right_labels = {}


# ============================================================
# PLOT EACH SOURCE FILE
# ============================================================
#
# ONE SOURCE FILE = ONE LINE
#
# The five rows belonging to the source file become:
#
#     point 1
#     point 2
#     point 3
#     point 4
#     point 5
#
# The line colour is determined by the experiment type.
# ============================================================

for source_file in source_files:

    file_data = plot_df[
        plot_df["source_file"]
        == source_file
    ].copy()


    if file_data.empty:
        continue


    # --------------------------------------------------------
    # Get experiment type.
    # --------------------------------------------------------

    experiment = (
        file_data["experiment"].iloc[0]
    )


    # --------------------------------------------------------
    # Get fixed experiment colour.
    # --------------------------------------------------------

    color = EXPERIMENT_COLORS.get(
        experiment,
        "black",
    )


    # --------------------------------------------------------
    # Only valid Y values.
    # --------------------------------------------------------

    valid_data = file_data[
        file_data["peak_difference"].notna()
    ].copy()


    if valid_data.empty:
        continue


    # --------------------------------------------------------
    # Plot ONE line for this source file.
    # --------------------------------------------------------

    line, = ax.plot(

        valid_data["point"],

        valid_data["peak_difference"],

        color=color,

        linewidth=1.2,

        marker="o",

        markersize=4,

        alpha=0.8,
    )


    line_objects[
        source_file
    ] = line


    # ========================================================
    # LEFT FILENAME LABEL
    # ========================================================

    first_point = valid_data.iloc[0]


    left_label = ax.annotate(

        source_file,

        xy=(

            first_point["point"],

            first_point["peak_difference"],

        ),

        xytext=(

            -5,

            0,

        ),

        textcoords="offset points",

        fontsize=FILENAME_FONT_SIZE,

        color=color,

        va="center",

        ha="right",

        alpha=0.9,
    )


    left_labels[
        source_file
    ] = left_label


    # ========================================================
    # RIGHT FILENAME LABEL
    # ========================================================

    last_point = valid_data.iloc[-1]


    right_label = ax.annotate(

        source_file,

        xy=(

            last_point["point"],

            last_point["peak_difference"],

        ),

        xytext=(

            5,

            0,

        ),

        textcoords="offset points",

        fontsize=FILENAME_FONT_SIZE,

        color=color,

        va="center",

        ha="left",

        alpha=0.9,
    )


    right_labels[
        source_file
    ] = right_label


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
    "for Five Selected Points per File",
    fontsize=14,
)


# ------------------------------------------------------------
# X-axis = exactly 1 through 5
# ------------------------------------------------------------

ax.set_xlim(
    0.4,
    5.6,
)


ax.set_xticks([
    1,
    2,
    3,
    4,
    5,
])


ax.set_xticklabels([
    "1",
    "2",
    "3",
    "4",
    "5",
])


# ============================================================
# GRID
# ============================================================

ax.grid(
    True,
    alpha=0.3,
)


# ============================================================
# EXPERIMENT LEGEND
# ============================================================
#
# The legend contains ONE entry per experiment type.
#
# It does NOT create separate legend entries for individual
# source files.
# ============================================================

legend_handles = []


for experiment, color in (
    EXPERIMENT_COLORS.items()
):

    handle, = ax.plot(

        [],

        [],

        color=color,

        linewidth=2,

        marker="o",

        markersize=4,

        label=experiment,
    )


    legend_handles.append(
        handle
    )


ax.legend(
    handles=legend_handles,
    title="Experiment",
    loc="upper left",
)


# ============================================================
# CHECKBOX LABELS
# ============================================================
#
# One checkbox per source CSV.
#
# Selecting/deselecting a CSV therefore shows/hides its
# complete five-point line.
# ============================================================

checkbox_labels = [
    Path(source_file).stem
    for source_file in source_files
]


# ------------------------------------------------------------
# Initially everything is visible.
# ------------------------------------------------------------

initial_visibility = [
    True
    for _ in source_files
]


check = CheckButtons(

    check_ax,

    checkbox_labels,

    initial_visibility,
)


# ------------------------------------------------------------
# Make checkbox labels small.
# ------------------------------------------------------------

for text in check.labels:

    text.set_fontsize(6)


check_ax.set_title(
    "Select files",
    fontsize=10,
    pad=10,
)


# ============================================================
# CHECKBOX TOGGLE FUNCTION
# ============================================================

def toggle_line(label):

    # --------------------------------------------------------
    # The checkbox label does not contain ".csv",
    # so convert it back to the full filename.
    # --------------------------------------------------------

    matching_source_file = None


    for source_file in source_files:

        if Path(
            source_file
        ).stem == label:

            matching_source_file = (
                source_file
            )

            break


    if matching_source_file is None:
        return


    # --------------------------------------------------------
    # Toggle line.
    # --------------------------------------------------------

    line = line_objects.get(
        matching_source_file
    )


    if line is not None:

        new_visibility = (
            not line.get_visible()
        )


        line.set_visible(
            new_visibility
        )


        # ----------------------------------------------------
        # Toggle filename labels too.
        # ----------------------------------------------------

        if matching_source_file in left_labels:

            left_labels[
                matching_source_file
            ].set_visible(
                new_visibility
            )


        if matching_source_file in right_labels:

            right_labels[
                matching_source_file
            ].set_visible(
                new_visibility
            )


        fig.canvas.draw_idle()


check.on_clicked(
    toggle_line
)


# ============================================================
# SELECT ALL
# ============================================================

def select_all(event):

    # --------------------------------------------------------
    # Turn every currently hidden line ON.
    # --------------------------------------------------------

    for source_file in source_files:

        line = line_objects.get(
            source_file
        )


        if line is None:
            continue


        if not line.get_visible():

            line.set_visible(
                True
            )


            if source_file in left_labels:

                left_labels[
                    source_file
                ].set_visible(
                    True
                )


            if source_file in right_labels:

                right_labels[
                    source_file
                ].set_visible(
                    True
                )


    # --------------------------------------------------------
    # Synchronize checkbox states.
    # --------------------------------------------------------

    for index, source_file in enumerate(
        source_files
    ):

        line = line_objects.get(
            source_file
        )


        if line is None:
            continue


        current_state = (
            check.get_status()[
                index
            ]
        )


        if not current_state:

            check.set_active(
                index
            )


    fig.canvas.draw_idle()


select_all_button = Button(
    select_all_ax,
    "Select All",
)


select_all_button.on_clicked(
    select_all
)


# ============================================================
# DESELECT ALL
# ============================================================

def deselect_all(event):

    # --------------------------------------------------------
    # Turn every line OFF.
    # --------------------------------------------------------

    for source_file in source_files:

        line = line_objects.get(
            source_file
        )


        if line is None:
            continue


        line.set_visible(
            False
        )


        if source_file in left_labels:

            left_labels[
                source_file
            ].set_visible(
                False
            )


        if source_file in right_labels:

            right_labels[
                source_file
            ].set_visible(
                False
            )


    # --------------------------------------------------------
    # Synchronize checkbox states.
    # --------------------------------------------------------

    for index, source_file in enumerate(
        source_files
    ):

        line = line_objects.get(
            source_file
        )


        if line is None:
            continue


        current_state = (
            check.get_status()[
                index
            ]
        )


        if current_state:

            check.set_active(
                index
            )


    fig.canvas.draw_idle()


deselect_all_button = Button(
    deselect_all_ax,
    "Deselect All",
)


deselect_all_button.on_clicked(
    deselect_all
)


# ============================================================
# SAVE BUTTON / NOTE
# ============================================================

save_note_ax = fig.add_axes([
    0.77,
    0.01,
    0.20,
    0.05,
])


save_note_ax.axis(
    "off"
)


save_note_ax.text(

    0.5,

    0.5,

    "Toggle files to compare",

    ha="center",

    va="center",

    fontsize=8,
)


# ============================================================
# LAYOUT
# ============================================================

plt.show()


# ============================================================
# SAVE FINAL VIEW
# ============================================================
#
# IMPORTANT:
#
# plt.show() blocks until the window is closed.
#
# When the user closes the window, save the final state
# of the plot.
# ============================================================

fig.savefig(

    PLOT_FILE,

    dpi=300,

    bbox_inches="tight",
)


print("\n")

print("=" * 70)

print("DONE")

print("=" * 70)


print(
    f"Plot saved to:\n{PLOT_FILE}"
)

print("=" * 70)