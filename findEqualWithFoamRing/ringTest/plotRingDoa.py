#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------------------------------------------------------
# Load CSV
# -------------------------------------------------------

filename = "test.csv"      # <-- change

df = pd.read_csv(filename)

# -------------------------------------------------------
# Extract angle columns
# -------------------------------------------------------

metadata = ["timestamp", "db_rms", "doa_deg", "peak_response"]

resp_cols = [c for c in df.columns if c not in metadata]

# Angles come directly from the headers
angles = np.array([float(c) for c in resp_cols])

responses = df[resp_cols].to_numpy()

# Normalize each frame independently
responses = responses / np.maximum(
    responses.max(axis=1, keepdims=True),
    1e-12
)

doa = df["doa_deg"].to_numpy()
db = df["db_rms"].to_numpy()

# -------------------------------------------------------
# Figure
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(14,6))
plt.subplots_adjust(bottom=0.18)

threshold = db.min()
mask = db >= threshold

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

scatter = ax.scatter(
    np.arange(mask.sum()),
    doa[mask],
    c="red",
    s=12,
)

ax.set_xlabel("Filtered Frame Index")
ax.set_ylabel("Angle (deg)")
ax.set_title("Spatial Response (Filtered by dB RMS)")

cbar = plt.colorbar(heat)
cbar.set_label("Normalized Spatial Response")

# -------------------------------------------------------
# Slider
# -------------------------------------------------------

slider_ax = plt.axes([0.15,0.06,0.70,0.03])

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

    heat.set_data(responses[mask].T)
    heat.set_extent([0, mask.sum(), angles.min(), angles.max()])

    scatter.set_offsets(
        np.column_stack((
            np.arange(mask.sum()),
            doa[mask]
        ))
    )

    ax.set_xlim(0, mask.sum())

    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()