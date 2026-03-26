import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------
# 1. Load CSV
# -----------------------------
file_path = "dataFromReal/23_03_data_2b.csv"
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
fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(bottom=0.2)

threshold_init = spl_array.min()

# -----------------------------
# 4. Filtering function (REAL filtering)
# -----------------------------
def filter_data(threshold):
    mask = spl_array >= threshold

    # Filter rows completely
    filtered_intensity = intensity_matrix[mask]
    filtered_spl = spl_array[mask]

    # Recompute peaks on filtered data
    peak_indices = np.argmax(filtered_intensity, axis=1)
    peak_angles = angles[peak_indices]

    # New time axis (compressed)
    new_time = np.arange(len(filtered_intensity))

    return filtered_intensity, peak_angles, new_time, filtered_spl


# Initial data
filtered_intensity, peak_angles, time_steps, filtered_spl = filter_data(threshold_init)

# -----------------------------
# 5. Initial plot
# -----------------------------
im = ax.imshow(
    filtered_intensity.T,
    aspect="auto",
    cmap="inferno",
    origin="lower",
    extent=[-0.5, len(time_steps) - 0.5, y_min, y_max]
)

scatter = ax.scatter(
    time_steps,
    peak_angles,
    color="cyan",
    s=20,
    edgecolors="white",
    linewidths=0.5
)

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

    filtered_intensity, peak_angles, time_steps, filtered_spl = filter_data(threshold)

    # Update heatmap
    im.set_data(filtered_intensity.T)
    im.set_extent([-0.5, len(time_steps) - 0.5, y_min, y_max])

    # Update scatter
    scatter.set_offsets(np.c_[time_steps, peak_angles])

    ax.set_xlim(-0.5, len(time_steps) - 0.5)

    fig.canvas.draw_idle()

slider.on_changed(update)

# -----------------------------
# 8. Show
# -----------------------------
plt.show()