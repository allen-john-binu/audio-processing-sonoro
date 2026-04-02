import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# -----------------------------
# 1. Load CSV
# -----------------------------
file_path = "hi.csv"
df = pd.read_csv(file_path)

# -----------------------------
# 2. Parse data
# -----------------------------
angle_cols = df.columns[2:]
angles = np.array([float(a) for a in angle_cols])

spl_array = df["dB_SPL"].astype(str).str.strip("[]").astype(float).values
intensity_matrix = df.iloc[:, 2:].values.astype(float)

num_steps = intensity_matrix.shape[0]

# -----------------------------
# 3. Normalization
# -----------------------------
int_min, int_max = intensity_matrix.min(), intensity_matrix.max()
spl_min, spl_max = spl_array.min(), spl_array.max()

# -----------------------------
# 4. Peak detection
# -----------------------------
peak_indices = np.argmax(intensity_matrix, axis=1)
peak_angles = angles[peak_indices]
time_steps = np.arange(num_steps)

# -----------------------------
# 5. Angle spacing fix (IMPORTANT)
# -----------------------------
angle_step = angles[1] - angles[0]
y_min = angles[0] - angle_step / 2
y_max = angles[-1] + angle_step / 2

# -----------------------------
# 6. Plot layout
# -----------------------------
fig = plt.figure(figsize=(18, 10))

gs = gridspec.GridSpec(
    nrows=2,
    ncols=2,
    height_ratios=[12, 1],
    width_ratios=[20, 1],
    hspace=0.05,
    wspace=0.2
)

# -----------------------------
# 7. HEATMAP (TRANSPOSED CORRECTLY)
# -----------------------------
ax_main = fig.add_subplot(gs[0, 0])

im = ax_main.imshow(
    intensity_matrix.T,   # transpose → (angle, time)
    aspect="auto",
    cmap="inferno",
    vmin=int_min,
    vmax=int_max,
    origin="lower",
    extent=[
        -0.5, num_steps - 0.5,   # X = time (centered)
        y_min, y_max             # Y = angle (correct spacing)
    ]
)

# ax_main.set_xlabel("Time Step")
ax_main.set_ylabel("Angle (degrees)")
ax_main.set_title("DOA Intensity + Peak (Correct Alignment)")

# -----------------------------
# 8. SCATTER PEAKS (IMPORTANT FIX)
# -----------------------------
ax_main.scatter(
    time_steps,        # X = time
    peak_angles,       # Y = angle
    color="cyan",
    s=20,
    edgecolors="white",
    linewidths=0.5,
    label="Peak DOA"
)

ax_main.legend()

# -----------------------------
# 9. INTENSITY COLORBAR
# -----------------------------
cax1 = fig.add_subplot(gs[0, 1])
cb1 = fig.colorbar(im, cax=cax1)
cb1.set_label("Intensity")

# -----------------------------
# 10. SPL STRIP (BOTTOM, ALIGNED)
# -----------------------------
ax_spl = fig.add_subplot(gs[1, 0], sharex=ax_main)

spl_image = spl_array.reshape(1, -1)

im_spl = ax_spl.imshow(
    spl_image,
    aspect="auto",
    cmap="viridis",
    vmin=spl_min,
    vmax=spl_max,
    extent=[-0.5, num_steps - 0.5, 0, 1]
)

ax_spl.set_yticks([])
ax_spl.set_xlabel("Time Step")

# -----------------------------
# 11. SPL COLORBAR
# -----------------------------
cax2 = fig.add_subplot(gs[1, 1])
cb2 = fig.colorbar(im_spl, cax=cax2)
cb2.set_label("dB SPL")

# -----------------------------
# 12. CLEANUP
# -----------------------------
plt.setp(ax_main.get_xticklabels(), visible=False)

plt.tight_layout()
plt.show()