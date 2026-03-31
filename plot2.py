import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.signal as signal

#negative angle robot is on the left, positive angle robot is on the right. 

# -----------------------------
# 1. Load CSV
# -----------------------------
file_path = "hi.csv"
df = pd.read_csv(file_path)

angle_cols = df.columns[2:]
angles = np.array([float(a) for a in angle_cols])

spl_array = df["dB_SPL"].astype(str).str.strip("[]").astype(float).values
intensity_matrix = df.iloc[:, 2:].values.astype(float)

# -----------------------------
# 2. Angle spacing fix
# -----------------------------
angle_step = angles[1] - angles[0]
y_min = angles[0] - angle_step / 2
y_max = angles[-1] + angle_step / 2

# -----------------------------
# 3. Figure setup
# -----------------------------
fig, (ax, ax_diff) = plt.subplots(
    1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [3, 1]}
)
plt.subplots_adjust(bottom=0.2, wspace=0.3)

threshold_init = spl_array.min()

# -----------------------------
# 4. Filtering function (REAL filtering)
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
            top2 = peaks[np.argsort(row[peaks])[-2:]]
            p1, p2 = top2

            a1, a2 = angles[p1], angles[p2]
            i1, i2 = row[p1], row[p2]

            peak_angles_1.append(a1)
            peak_angles_2.append(a2)

            intensity_diffs.append(abs(i1 - i2))

        elif len(peaks) == 1:
            # Only one peak
            p1 = peaks[0]

            a1 = angles[p1]
            i1 = row[p1]

            peak_angles_1.append(a1)
            peak_angles_2.append(np.nan)

            intensity_diffs.append(np.nan)

        else:
            # No peaks
            peak_angles_1.append(np.nan)
            peak_angles_2.append(np.nan)
            intensity_diffs.append(np.nan)

    new_time = np.arange(len(filtered_intensity))

    return (
        filtered_intensity,
        np.array(peak_angles_1),
        np.array(peak_angles_2),
        np.array(intensity_diffs),
        new_time,
        filtered_spl
    )

# Initial data
filtered_intensity, peak1, peak2, intensity_diffs, time_steps, filtered_spl = filter_data(threshold_init)

# -----------------------------
# 5. Initial plot
# -----------------------------
# Replace NaNs with 0 for plotting
intensity_plot = np.nan_to_num(intensity_diffs, nan=0.0)

# Create color array (red for NaN, default for valid)
colors = ["red" if np.isnan(v) else "blue" for v in intensity_diffs]

scatter_diff = ax_diff.scatter(time_steps, intensity_plot, c=colors, s=20)

ax_diff.set_title("Peak Intensity Difference")
ax_diff.set_xlabel("Time Step")
ax_diff.set_ylabel("|Peak1 - Peak2|")

ax_diff.set_xlim(-0.5, len(time_steps) - 0.5)

im = ax.imshow(
    filtered_intensity.T,
    aspect="auto",
    cmap="inferno",
    origin="lower",
    extent=[-0.5, len(time_steps) - 0.5, y_min, y_max]
)

# -----------------------------
# 6. Peak scatter plots
# -----------------------------
scatter1 = ax.scatter(time_steps, peak1, color="cyan", s=20, label="Peak 1")
scatter2 = ax.scatter(time_steps, peak2, color="lime", s=20, label="Peak 2")

# -----------------------------
# 8. Labels
# -----------------------------
ax.set_xlabel("Time Step (Filtered)")
ax.set_ylabel("Angle (degrees)")
ax.set_title("DOA with SPL Filtering (Compressed Time)")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Intensity")

# -----------------------------
# 6. Slider
# -----------------------------
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])

slider = Slider(
    ax_slider,
    "SPL Threshold",
    spl_array.min(),
    spl_array.max(),
    valinit=threshold_init
)

# -----------------------------
# 7. Update function
# -----------------------------
def update(val):
    threshold = slider.val

    filtered_intensity, peak1, peak2, intensity_diffs, time_steps, filtered_spl = filter_data(threshold)

    # Update peaks
    scatter1.set_offsets(np.c_[time_steps, peak1])
    scatter2.set_offsets(np.c_[time_steps, peak2])

    # Update heatmap
    im.set_data(filtered_intensity.T)
    im.set_extent([-0.5, len(time_steps) - 0.5, y_min, y_max])

    # Update scatter

    # Update peaks
    scatter1.set_offsets(np.c_[time_steps, peak1])
    scatter2.set_offsets(np.c_[time_steps, peak2])

    ax.set_xlim(-0.5, len(time_steps) - 0.5)

        # Update intensity difference line
    # Replace NaNs with 0
    intensity_plot = np.nan_to_num(intensity_diffs, nan=0.0)

    # Update colors (red for NaN)
    colors = ["red" if np.isnan(v) else "blue" for v in intensity_diffs]

    # Update scatter positions
    scatter_diff.set_offsets(np.c_[time_steps, intensity_plot])

    # Update colors
    scatter_diff.set_color(colors)

    # Rescale axis
    ax_diff.set_xlim(-0.5, len(time_steps) - 0.5)
    ax_diff.relim()
    ax_diff.autoscale_view()

    fig.canvas.draw_idle()

slider.on_changed(update)

# -----------------------------
# 8. Show
# -----------------------------
plt.show()