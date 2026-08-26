import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 15,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def doa_to_row(
    doa_estimate,
    global_min=0.0000000005,
    global_max=0.0000068100,
    NORMALIZING_FACTOR=0.1546,
):
    """
    Replicates the normalization performed in the ring attractor script.
    """
    denom = global_max - global_min
    norm_arr = []

    for v in doa_estimate:
        if v == 0.0:
            norm_arr.append(0.0)
        else:
            if denom == 0:
                norm = 0.0
            else:
                norm = (v - global_min) / denom

            norm_arr.append(norm * NORMALIZING_FACTOR)

    return np.array(norm_arr)


# ============================================================
# Load CSV
# ============================================================
df = pd.read_csv("hi.csv")

# Angle values from the header: -90, -87, ..., 90
angles = np.array(df.columns[2:], dtype=float)

# Single row of intensity values
intensities = df.iloc[277, 2:].astype(float).values


# ============================================================
# Normalize intensity
# ============================================================
min_val = 0.0000000005
max_val = 0.0000068100

if max_val == min_val:
    normalized = np.zeros_like(intensities)
else:
    normalized = (intensities - min_val) / (max_val - min_val)


# ============================================================
# Create 120-element ring input
# -180° to 177° with 3° resolution
# ============================================================
full_array_doa = np.zeros(120)

for angle, value in zip(angles, intensities):
    idx = int((angle + 180) // 3)
    full_array_doa[idx] = value

ring_input = doa_to_row(full_array_doa)

full_angles = np.arange(-180, 180, 3)


# ============================================================
# Combined figure: 3 plots horizontally
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))


# (a) Original DOA intensity
axes[0].plot(angles, intensities, marker="o", color="blue")
axes[0].set_xlabel("Angle of arrival (°)")
axes[0].set_ylabel("Sound Intensity")
axes[0].set_ylim(0, 0.000007)

# (b) Min-max normalized intensity
axes[1].plot(angles, normalized, marker="o", color="red")
axes[1].set_xlabel("Angle (°)")
axes[1].set_ylabel("Normalized Sound Intensity")

# (c) Final ring attractor input
axes[2].plot(full_angles, ring_input, marker="o", color="green")
axes[2].set_xlabel("Angle (°)")
axes[2].set_ylabel("External Input to Ring Attractor")


# Improve spacing
fig.tight_layout()


# ============================================================
# Save as vector PDF
# ============================================================
fig.savefig(
    "doa_normalization_pipeline.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()