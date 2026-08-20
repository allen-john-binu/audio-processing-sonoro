#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import find_peaks

# ============================================================
# Configuration
# ============================================================

filename = "./datacollection/dataFromSound/doa_log_20251119_fast_UCA.csv"

PROMINENCE = 0.05
DISTANCE = 5

# Only keep frames with db_rms >= this value
DB_RMS_THRESHOLD = -33.0

# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(filename)

metadata = [
    "timestamp",
    "elapsed_s",
    "db_rms",
    "peak_response",
]

# Everything else is assumed to be an angle response column
resp_cols = [c for c in df.columns if c not in metadata]

angles = np.array([float(c) for c in resp_cols])
responses = df[resp_cols].to_numpy(dtype=float)

elapsed = df["elapsed_s"].to_numpy()
db = df["db_rms"].to_numpy()

# ============================================================
# Filter frames based on db_rms threshold
# ============================================================

valid_mask = db >= DB_RMS_THRESHOLD

print(
    f"Keeping {valid_mask.sum()} / {len(valid_mask)} frames "
    f"with dB RMS >= {DB_RMS_THRESHOLD:.2f} dB"
)

responses = responses[valid_mask]
elapsed = elapsed[valid_mask]
db = db[valid_mask]

# Keep the original CSV row index if useful
original_indices = df.index.to_numpy()[valid_mask]

if len(responses) == 0:
    raise ValueError(
        f"No frames satisfy db_rms >= {DB_RMS_THRESHOLD:.2f} dB"
    )

# ============================================================
# Normalize each frame independently
# ============================================================

responses = responses / np.maximum(
    responses.max(axis=1, keepdims=True),
    1e-12
)

# ============================================================
# Peak detection for every frame
# ============================================================

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

    elif len(peaks) == 1:
        single_angles[i] = angles[peaks[0]]

    else:
        # Sort peaks according to response height
        order = np.argsort(y[peaks])[::-1]

        p1 = peaks[order[0]]
        p2 = peaks[order[1]]

        peak1_angles[i] = angles[p1]
        peak2_angles[i] = angles[p2]

# ============================================================
# Create figure
# ============================================================

fig = plt.figure(figsize=(15, 10))

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[2, 2],
    hspace=0.35
)

ax_heat = fig.add_subplot(gs[0])
ax_response = fig.add_subplot(gs[1])

plt.subplots_adjust(
    bottom=0.15,
    left=0.08,
    right=0.95
)

# ============================================================
# 1. Spatial response heatmap
# ============================================================

heat = ax_heat.imshow(
    responses.T,
    origin="lower",
    aspect="auto",
    extent=[
        0,
        len(responses) - 1,
        angles.min(),
        angles.max()
    ],
    cmap="viridis",
    interpolation="nearest",
    vmin=0,
    vmax=1,
)

ax_heat.set_xlabel("Filtered Timestep")
ax_heat.set_ylabel("Angle (deg)")
ax_heat.set_title(
    f"Spatial Response — dB RMS >= {DB_RMS_THRESHOLD:.1f} dB"
)

cbar = fig.colorbar(
    heat,
    ax=ax_heat
)

cbar.set_label("Normalized Spatial Response")

# Selected timestep marker

selected_frame = 0

heat_line = ax_heat.axvline(
    selected_frame,
    color="white",
    linewidth=2,
    linestyle="--",
)

# ============================================================
# 2. Selected frame spatial response
# ============================================================

y0 = responses[selected_frame]

response_line, = ax_response.plot(
    angles,
    y0,
    linewidth=2,
    label="Spatial response"
)

# Peak markers

p1_marker, = ax_response.plot(
    [],
    [],
    "o",
    color="blue",
    markersize=8,
    label="Highest peak"
)

p2_marker, = ax_response.plot(
    [],
    [],
    "o",
    color="lime",
    markersize=8,
    label="Second highest peak"
)

single_marker, = ax_response.plot(
    [],
    [],
    "o",
    color="red",
    markersize=8,
    label="Only peak"
)

ax_response.set_xlim(
    angles.min(),
    angles.max()
)

ax_response.set_ylim(
    0,
    1.05
)

ax_response.set_xlabel("Angle (deg)")
ax_response.set_ylabel("Normalized response")

ax_response.grid(True, alpha=0.3)

ax_response.legend(
    loc="upper right"
)

# ============================================================
# Timestep information
# ============================================================

title_text = fig.suptitle(
    "",
    fontsize=14
)

# ============================================================
# Slider
# ============================================================

slider_ax = plt.axes(
    [0.15, 0.055, 0.70, 0.035]
)

slider = Slider(
    slider_ax,
    "Timestep",
    0,
    len(responses) - 1,
    valinit=0,
    valstep=1,
)

# ============================================================
# Update function
# ============================================================

def update(val):

    timestep = int(slider.val)

    y = responses[timestep]

    # --------------------------------------------------------
    # Update timestep marker
    # --------------------------------------------------------

    heat_line.set_xdata([timestep, timestep])

    # --------------------------------------------------------
    # Update selected spatial response
    # --------------------------------------------------------

    response_line.set_ydata(y)

    # --------------------------------------------------------
    # Detect peaks for selected frame
    # --------------------------------------------------------

    peaks, properties = find_peaks(
        y,
        prominence=PROMINENCE,
        distance=DISTANCE
    )

    # Clear peak markers

    p1_marker.set_data([], [])
    p2_marker.set_data([], [])
    single_marker.set_data([], [])

    if len(peaks) == 1:

        p = peaks[0]

        single_marker.set_data(
            [angles[p]],
            [y[p]]
        )

    elif len(peaks) >= 2:

        # Sort by response height
        order = np.argsort(y[peaks])[::-1]

        p1 = peaks[order[0]]
        p2 = peaks[order[1]]

        p1_marker.set_data(
            [angles[p1]],
            [y[p1]]
        )

        p2_marker.set_data(
            [angles[p2]],
            [y[p2]]
        )

    # --------------------------------------------------------
    # Update title
    # --------------------------------------------------------

    title_text.set_text(
        f"Filtered timestep: {timestep}   |   "
        f"Original row: {original_indices[timestep]}   |   "
        f"Time: {elapsed[timestep]:.3f} s   |   "
        f"dB RMS: {db[timestep]:.2f} dB   |   "
        f"Threshold: {DB_RMS_THRESHOLD:.2f} dB"
    )

    # --------------------------------------------------------
    # Redraw
    # --------------------------------------------------------

    fig.canvas.draw_idle()


# ============================================================
# Connect slider
# ============================================================

slider.on_changed(update)

# Initialize

update(0)

# ============================================================
# Show
# ============================================================

plt.show()