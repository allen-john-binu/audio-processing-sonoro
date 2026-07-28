import numpy as np
import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import scipy.signal as signal

# Negative angle = robot is on the left
# Positive angle = robot is on the right

ANGLE_START = 2
ANGLE_END = -11

# --------------------------------------------------
# HARDCODED SAVE FOLDER
# Change this to wherever you want plots saved
# --------------------------------------------------
SAVE_FOLDER = pathlib.Path("./saved_plots")

# Create folder automatically if it doesn't exist
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 1. Load CSV + filename argument
# -----------------------------
if len(sys.argv) < 3:
    print(
        "Usage: python3 plot_bump_stats.py "
        "<path_to_csv> <save_filename>"
    )
    print(
        "Example: python3 plot_bump_stats.py "
        "data.csv run_01"
    )
    sys.exit(1)

csv_path = pathlib.Path(sys.argv[1])

# Base filename passed as argument
save_filename = sys.argv[2]

if not csv_path.exists():
    print(f"Error: file not found — {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)


# -----------------------------
# 2. Read data
# -----------------------------
angle_cols = df.columns[ANGLE_START:ANGLE_END]
angles = np.array([float(a) for a in angle_cols])

spl_array = (
    df["dB_SPL"]
    .astype(str)
    .str.strip("[]")
    .astype(float)
    .values
)

intensity_matrix = (
    df.iloc[:, ANGLE_START:ANGLE_END]
    .values
    .astype(float)
)


# -----------------------------
# 3. Angle spacing fix
# -----------------------------
angle_step = angles[1] - angles[0]

y_min = angles[0] - angle_step / 2
y_max = angles[-1] + angle_step / 2


# -----------------------------
# 4. Figure setup
# -----------------------------
fig, (ax, ax_diff) = plt.subplots(
    1,
    2,
    figsize=(16, 8),
    gridspec_kw={"width_ratios": [3, 1]}
)

# Extra bottom space for slider + save button
plt.subplots_adjust(
    bottom=0.22,
    wspace=0.3
)

threshold_init = spl_array.min()


# -----------------------------
# 5. Filtering function
# -----------------------------
def filter_data(threshold):

    mask = spl_array >= threshold

    # Filter rows completely
    filtered_intensity = intensity_matrix[mask]
    filtered_spl = spl_array[mask]

    peak_angles_1 = []
    peak_angles_2 = []
    intensity_diffs = []

    for row in filtered_intensity:

        peaks, _ = signal.find_peaks(row)

        if len(peaks) >= 2:

            # Take top 2 peaks by intensity
            top2 = peaks[
                np.argsort(row[peaks])[-2:]
            ]

            p1, p2 = top2

            a1 = angles[p1]
            a2 = angles[p2]

            i1 = row[p1]
            i2 = row[p2]

            peak_angles_1.append(a1)
            peak_angles_2.append(a2)

            intensity_diffs.append(
                abs(i1 - i2)
            )

        elif len(peaks) == 1:

            # Only one peak
            p1 = peaks[0]

            a1 = angles[p1]

            peak_angles_1.append(a1)
            peak_angles_2.append(np.nan)

            intensity_diffs.append(np.nan)

        else:

            # No peaks
            peak_angles_1.append(np.nan)
            peak_angles_2.append(np.nan)

            intensity_diffs.append(np.nan)

    new_time = np.arange(
        len(filtered_intensity)
    )

    return (
        filtered_intensity,
        np.array(peak_angles_1),
        np.array(peak_angles_2),
        np.array(intensity_diffs),
        new_time,
        filtered_spl
    )


# -----------------------------
# 6. Initial data
# -----------------------------
(
    filtered_intensity,
    peak1,
    peak2,
    intensity_diffs,
    time_steps,
    filtered_spl
) = filter_data(threshold_init)


# -----------------------------
# 7. Initial difference plot
# -----------------------------

# Replace NaNs with 0 for plotting
intensity_plot = np.nan_to_num(
    intensity_diffs,
    nan=0.0
)

# Red = NaN / only one or no peak
# Blue = valid two-peak difference
colors = [
    "red" if np.isnan(v) else "blue"
    for v in intensity_diffs
]

scatter_diff = ax_diff.scatter(
    time_steps,
    intensity_plot,
    c=colors,
    s=20
)

ax_diff.set_title(
    "Peak Intensity Difference"
)

ax_diff.set_xlabel(
    "Time Step"
)

ax_diff.set_ylabel(
    "|Peak1 - Peak2|"
)

ax_diff.set_xlim(
    -0.5,
    len(time_steps) - 0.5
)


# -----------------------------
# 8. Heatmap
# -----------------------------
im = ax.imshow(
    filtered_intensity.T,
    aspect="auto",
    cmap="inferno",
    origin="lower",
    extent=[
        -0.5,
        len(time_steps) - 0.5,
        y_min,
        y_max
    ]
)


# -----------------------------
# 9. Peak scatter plots
# -----------------------------
scatter1 = ax.scatter(
    time_steps,
    peak1,
    color="cyan",
    s=20,
    label="Peak 1"
)

scatter2 = ax.scatter(
    time_steps,
    peak2,
    color="lime",
    s=20,
    label="Peak 2"
)


# -----------------------------
# 10. Labels
# -----------------------------
ax.set_xlabel(
    "Time Step (Filtered)"
)

ax.set_ylabel(
    "Angle (degrees)"
)

ax.set_title(
    "DOA with SPL Filtering (Compressed Time)"
)

ax.legend()

cbar = plt.colorbar(
    im,
    ax=ax
)

cbar.set_label(
    "Intensity"
)


# -----------------------------
# 11. SPL Slider
# -----------------------------
ax_slider = plt.axes(
    [0.15, 0.08, 0.55, 0.03]
)

slider = Slider(
    ax_slider,
    "SPL Threshold",
    spl_array.min(),
    spl_array.max(),
    valinit=threshold_init
)


# -----------------------------
# 12. Update function
# -----------------------------
def update(val):

    threshold = slider.val

    (
        filtered_intensity,
        peak1,
        peak2,
        intensity_diffs,
        time_steps,
        filtered_spl
    ) = filter_data(threshold)

    # --------------------------------
    # Update peak scatter plots
    # --------------------------------
    scatter1.set_offsets(
        np.c_[time_steps, peak1]
    )

    scatter2.set_offsets(
        np.c_[time_steps, peak2]
    )

    # --------------------------------
    # Update heatmap
    # --------------------------------
    im.set_data(
        filtered_intensity.T
    )

    im.set_extent([
        -0.5,
        len(time_steps) - 0.5,
        y_min,
        y_max
    ])

    ax.set_xlim(
        -0.5,
        len(time_steps) - 0.5
    )

    # --------------------------------
    # Update intensity difference
    # --------------------------------
    intensity_plot = np.nan_to_num(
        intensity_diffs,
        nan=0.0
    )

    colors = [
        "red" if np.isnan(v) else "blue"
        for v in intensity_diffs
    ]

    scatter_diff.set_offsets(
        np.c_[time_steps, intensity_plot]
    )

    scatter_diff.set_color(
        colors
    )

    # --------------------------------
    # Rescale difference plot
    # --------------------------------
    ax_diff.set_xlim(
        -0.5,
        len(time_steps) - 0.5
    )

    ax_diff.relim()
    ax_diff.autoscale_view()

    fig.canvas.draw_idle()


slider.on_changed(update)


# -----------------------------
# 13. Find next save number
# -----------------------------
def get_next_save_number():

    """
    Find the next available filename number.

    Example:
        run_01_01.png exists
        run_01_02.png exists

    Returns:
        3

    So next file becomes:
        run_01_03.png
    """

    counter = 1

    while True:

        output_path = SAVE_FOLDER / (
            f"{save_filename}_{counter:02d}.png"
        )

        if not output_path.exists():
            return counter

        counter += 1


# Initialize counter based on existing files
save_counter = get_next_save_number()


# -----------------------------
# 14. Save function
# -----------------------------
def save_plot(event):

    global save_counter

    # Build filename:
    #
    # run_01_01.png
    # run_01_02.png
    # run_01_03.png
    #
    output_path = SAVE_FOLDER / (
        f"{save_filename}_{save_counter:02d}.png"
    )

    # Get current threshold
    current_threshold = slider.val

    print(
        f"Saving plot with SPL threshold: "
        f"{current_threshold:.2f}"
    )

    # Save current figure
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Saved: {output_path}"
    )

    # Increment for next click
    save_counter += 1


# -----------------------------
# 15. Save button
# -----------------------------
ax_save = plt.axes(
    [0.76, 0.06, 0.12, 0.06]
)

save_button = Button(
    ax_save,
    "Save Plot"
)

save_button.on_clicked(
    save_plot
)


# -----------------------------
# 16. Show
# -----------------------------
plt.show()