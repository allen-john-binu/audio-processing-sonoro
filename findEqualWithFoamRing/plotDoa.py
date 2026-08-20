#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import find_peaks

# -------------------------------------------------------
# Load CSV
# -------------------------------------------------------

# filename = "./datacollection/dataFromSound/doa_log_20251119_alternating1.csv"      # <-- change if needed

filename = "./datacollection/dataFromSound/doa_log_20251119_fast_RCA_speakerChange.csv"

df = pd.read_csv(filename)

# -------------------------------------------------------
# Extract angle columns
# -------------------------------------------------------

metadata = ["timestamp", "elapsed_s", "db_rms","doa_deg", "peak_response"]

resp_cols = [c for c in df.columns if c not in metadata]

angles = np.array([float(c) for c in resp_cols])

responses = df[resp_cols].to_numpy()

# Normalize each frame independently
responses = responses / np.maximum(
    responses.max(axis=1, keepdims=True),
    1e-12
)

db = df["db_rms"].to_numpy()

# -------------------------------------------------------
# Peak detection parameters
# -------------------------------------------------------

PROMINENCE = 0.05      # minimum prominence
DISTANCE = 5           # minimum separation (samples)

# -------------------------------------------------------
# Detect peaks for every frame
# -------------------------------------------------------

peak1_angles = np.full(len(responses), np.nan)
peak2_angles = np.full(len(responses), np.nan)
single_angles = np.full(len(responses), np.nan)

for i, y in enumerate(responses):

    peaks, properties = find_peaks(
        y,
        prominence=PROMINENCE,
        distance=DISTANCE
    )

    if len(peaks) == 0:
        continue

    # elif len(peaks) == :
    else:
        single_angles[i] = angles[peaks[0]]

    # else:
    #     # Sort by peak height
    #     order = np.argsort(y[peaks])[::-1]

    #     p1 = peaks[order[0]]
    #     p2 = peaks[order[1]]

    #     peak1_angles[i] = angles[p1]
    #     peak2_angles[i] = angles[p2]

# -------------------------------------------------------
# Figure
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))
plt.subplots_adjust(bottom=0.18)

threshold = db.min()
mask = db >= threshold

# Heatmap
heat = ax.imshow(
    responses[mask].T,
    origin="lower",
    aspect="auto",
    extent=[0, mask.sum(), angles.min(), angles.max()],
    cmap="viridis",
    interpolation="nearest",
    vmin=0,
    vmax=1,
)

# Highest peak
blue_scatter = ax.scatter(
    np.arange(mask.sum()),
    peak1_angles[mask],
    color="blue",
    s=20,
    label="Highest Peak",
    zorder=5,
)

# Second highest peak
green_scatter = ax.scatter(
    np.arange(mask.sum()),
    peak2_angles[mask],
    color="lime",
    s=20,
    label="Second Highest Peak",
    zorder=5,
)

# Only one peak
red_scatter = ax.scatter(
    np.arange(mask.sum()),
    single_angles[mask],
    color="red",
    s=20,
    label="Only Peak",
    zorder=5,
)

ax.set_xlabel("Filtered Frame Index")
ax.set_ylabel("Angle (deg)")
ax.set_title("Spatial Response with Peak Detection")

cbar = plt.colorbar(heat)
cbar.set_label("Normalized Spatial Response")

ax.legend(loc="upper right")

# -------------------------------------------------------
# Slider
# -------------------------------------------------------

slider_ax = plt.axes([0.15, 0.06, 0.70, 0.03])

slider = Slider(
    slider_ax,
    "dB RMS",
    np.floor(db.min()),
    np.ceil(db.max()),
    valinit=threshold,
)

# -------------------------------------------------------
# Update callback
# -------------------------------------------------------

def update(val):

    threshold = slider.val

    mask = db >= threshold

    if mask.sum() == 0:
        return

    # Update heatmap
    heat.set_data(responses[mask].T)
    heat.set_extent([0, mask.sum(), angles.min(), angles.max()])

    x = np.arange(mask.sum())

    blue_scatter.set_offsets(
        np.column_stack((x, peak1_angles[mask]))
    )

    green_scatter.set_offsets(
        np.column_stack((x, peak2_angles[mask]))
    )

    red_scatter.set_offsets(
        np.column_stack((x, single_angles[mask]))
    )

    ax.set_xlim(0, mask.sum())

    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()