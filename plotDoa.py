import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
intensities = df.iloc[0, 2:].astype(float).values


# ============================================================
# Plot 1: Original intensity vs angle
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(angles, intensities, marker="o",  color="blue")
plt.xlabel("angle")
plt.ylabel("intensity")
plt.tight_layout()


# ============================================================
# Plot 2: Min-max normalized intensity vs angle
# ============================================================
min_val = 0.0000000005
max_val = 0.0000068100

if max_val == min_val:
    normalized = np.zeros_like(intensities)
else:
    normalized = (intensities - min_val) / (max_val - min_val)

plt.figure(figsize=(8, 5))
plt.plot(angles, normalized, marker="o", color="red")
plt.xlabel("angle")
plt.ylabel("intensity")
plt.tight_layout()


# ============================================================
# Plot 3:
# Create a 120-element array representing
# angles from -180° to 177° with 3° resolution
# ============================================================
full_array_doa = np.zeros(120)

for angle, value in zip(angles, intensities):
    idx = int((angle + 180) // 3)
    full_array_doa[idx] = value

ring_input = doa_to_row(full_array_doa)

# x-axis as actual angles
full_angles = np.arange(-180, 180, 3)

plt.figure(figsize=(10, 5))
plt.plot(full_angles, ring_input, marker="o", color="green")
plt.xlabel("angle")
plt.ylabel("intensity")
plt.tight_layout()

plt.show()