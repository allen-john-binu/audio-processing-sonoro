import numpy as np
import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import scipy.signal as signal

# -----------------------------
# Config
# -----------------------------
ANGLE_START = 2

FILE_KEYS = [
    "doa_1_5R",
    "doa_1_6R",
    "doa_1_7R",
    "doa_1_8R",
    "doa_1_9R",
    "doa_1_1",
]

# FILE_KEYS = [
#     "doa_1_1",
#     "doa_1_5L",
#     "doa_1_6L",
#     "doa_1_7L",
#     "doa_1_8L",
#     "doa_1_9L",
# ]

PANEL_LAYOUT = (2, 3)   # rows x cols of panels
HEATMAP_ROW  = 0        # within each panel: heatmap on top
DIFF_ROW     = 1        # intensity diff scatter on bottom


# -----------------------------
# 1. Load folder
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python3 plot_doa_compare.py <folder_path>")
    sys.exit(1)

folder = pathlib.Path(sys.argv[1])
if not folder.is_dir():
    print(f"Error: not a directory — {folder}")
    sys.exit(1)

# Resolve each file; warn if missing
datasets = {}
for key in FILE_KEYS:
    matches = sorted(folder.glob(f"{key}.csv"))
    if not matches:
        print(f"Warning: {key}.csv not found in {folder}, skipping.")
    else:
        datasets[key] = matches[0]

if not datasets:
    print("No matching files found. Exiting.")
    sys.exit(1)

# -----------------------------
# 2. Parse all CSVs
# -----------------------------
file_data = {}   # key -> dict with angles, spl_array, intensity_matrix, angle geometry

for key, path in datasets.items():
    df = pd.read_csv(path)
    angle_cols = df.columns[ANGLE_START:]
    angles = np.array([float(a) for a in angle_cols])
    spl_array = df["dB_SPL"].astype(str).str.strip("[]").astype(float).values
    intensity_matrix = df.iloc[:, ANGLE_START:].values.astype(float)

    angle_step = angles[1] - angles[0]
    y_min = angles[0]  - angle_step / 2
    y_max = angles[-1] + angle_step / 2

    file_data[key] = dict(
        angles=angles,
        spl_array=spl_array,
        intensity_matrix=intensity_matrix,
        y_min=y_min,
        y_max=y_max,
        path=path,
    )

# Global SPL range across all files
global_spl_min = min(d["spl_array"].min() for d in file_data.values())
global_spl_max = max(d["spl_array"].max() for d in file_data.values())


# -----------------------------
# 3. Filtering / peak detection
# -----------------------------
def filter_data(key, threshold):
    d = file_data[key]
    spl_array       = d["spl_array"]
    intensity_matrix = d["intensity_matrix"]
    angles          = d["angles"]

    mask = spl_array >= threshold
    filtered_intensity = intensity_matrix[mask]

    peak_angles_1  = []
    peak_angles_2  = []
    intensity_diffs = []

    for row in filtered_intensity:
        peaks, _ = signal.find_peaks(row)

        if len(peaks) >= 2:
            top2 = peaks[np.argsort(row[peaks])[-2:]]
            p1, p2 = top2
            peak_angles_1.append(angles[p1])
            peak_angles_2.append(angles[p2])
            intensity_diffs.append(abs(row[p1] - row[p2]))

        elif len(peaks) == 1:
            peak_angles_1.append(angles[peaks[0]])
            peak_angles_2.append(np.nan)
            intensity_diffs.append(np.nan)

        else:
            peak_angles_1.append(np.nan)
            peak_angles_2.append(np.nan)
            intensity_diffs.append(np.nan)

    time_steps = np.arange(len(filtered_intensity))

    return (
        filtered_intensity,
        np.array(peak_angles_1),
        np.array(peak_angles_2),
        np.array(intensity_diffs),
        time_steps,
    )


# -----------------------------
# 4. Figure & axes layout
#
#  Internal grid per panel:
#    row 0 → heatmap  (height ratio 3)
#    row 1 → diff     (height ratio 1)
#
#  Panels arranged in PANEL_LAYOUT (2 rows x 3 cols)
#  with a small gap between panels.
# -----------------------------
n_panel_rows, n_panel_cols = PANEL_LAYOUT
n_ax_rows = n_panel_rows * 2   # heatmap + diff per panel row

fig = plt.figure(figsize=(18, 11))
fig.suptitle("DOA Comparison — SPL Filtered", fontsize=14, y=0.98)

# Outer grid: panel rows × panel cols, with spacing
outer = gridspec.GridSpec(
    n_panel_rows, n_panel_cols,
    figure=fig,
    hspace=0.45,
    wspace=0.30,
    top=0.93, bottom=0.12,
    left=0.06, right=0.97
)

# For each panel, an inner 2-row gridspec
axes_heat = {}   # key -> ax
axes_diff = {}   # key -> ax

keys_ordered = [k for k in FILE_KEYS if k in file_data]

for idx, key in enumerate(keys_ordered):
    pr = idx // n_panel_cols   # panel row
    pc = idx %  n_panel_cols   # panel col

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer[pr, pc],
        hspace=0.08,
        height_ratios=[3, 1]
    )

    ax_h = fig.add_subplot(inner[0])
    ax_d = fig.add_subplot(inner[1])

    axes_heat[key] = ax_h
    axes_diff[key] = ax_d

    ax_h.set_title(key, fontsize=9, pad=3)
    ax_h.set_ylabel("Angle (°)", fontsize=7)
    ax_h.tick_params(axis="both", labelsize=7)
    ax_h.tick_params(labelbottom=False)   # share x with diff panel

    ax_d.set_ylabel("|ΔPeak|", fontsize=7)
    ax_d.set_xlabel("Time Step", fontsize=7)
    ax_d.tick_params(axis="both", labelsize=7)


# -----------------------------
# 5. Initial draw
# -----------------------------
plot_objects = {}   # key -> dict of artists

for key in keys_ordered:
    d = file_data[key]
    filtered_intensity, peak1, peak2, intensity_diffs, time_steps = filter_data(key, global_spl_min)

    ax_h = axes_heat[key]
    ax_d = axes_diff[key]

    # --- Heatmap ---
    n_t = max(len(time_steps), 1)
    im = ax_h.imshow(
        filtered_intensity.T,
        aspect="auto",
        cmap="inferno",
        origin="lower",
        extent=[-0.5, n_t - 0.5, d["y_min"], d["y_max"]]
    )
    ax_h.set_xlim(-0.5, n_t - 0.5)
    ax_h.set_ylim(d["y_min"], d["y_max"])

    cbar = fig.colorbar(im, ax=ax_h, pad=0.01, fraction=0.046)
    cbar.set_label("Intensity", fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    # --- Peak overlays ---
    sc1 = ax_h.scatter(time_steps, peak1, color="cyan",  s=8, label="Peak 1", zorder=3)
    sc2 = ax_h.scatter(time_steps, peak2, color="lime",  s=8, label="Peak 2", zorder=3)

    # --- Intensity diff scatter ---
    intensity_plot = np.nan_to_num(intensity_diffs, nan=0.0)
    colors = ["red" if np.isnan(v) else "steelblue" for v in intensity_diffs]
    sc_diff = ax_d.scatter(time_steps, intensity_plot, c=colors, s=8, zorder=3)
    ax_d.set_xlim(-0.5, n_t - 0.5)

    plot_objects[key] = dict(im=im, sc1=sc1, sc2=sc2, sc_diff=sc_diff)

# Compute initial global diff max and sync y-axes
def sync_diff_ylim(keys):
    """Set all diff axes to the same y range based on current data."""
    all_max = []
    for key in keys:
        ax_d = axes_diff[key]
        y_data = [pt[1] for pt in plot_objects[key]["sc_diff"].get_offsets()]
        if y_data:
            all_max.append(max(y_data))
    global_max = max(all_max) if all_max else 1.0
    for key in keys:
        axes_diff[key].set_ylim(0, global_max * 1.1)

sync_diff_ylim(keys_ordered)


# -----------------------------
# 6. Slider
# -----------------------------
ax_slider = fig.add_axes([0.20, 0.03, 0.60, 0.025])
slider = Slider(
    ax_slider,
    "Global SPL Threshold",
    global_spl_min,
    global_spl_max,
    valinit=global_spl_min,
    color="steelblue"
)
slider.label.set_fontsize(9)


# -----------------------------
# 7. Update callback
# -----------------------------
def update(val):
    threshold = slider.val

    for key in keys_ordered:
        d = file_data[key]
        filtered_intensity, peak1, peak2, intensity_diffs, time_steps = filter_data(key, threshold)

        n_t = max(len(time_steps), 1)
        objs = plot_objects[key]

        # Heatmap
        objs["im"].set_data(filtered_intensity.T)
        objs["im"].set_extent([-0.5, n_t - 0.5, d["y_min"], d["y_max"]])
        objs["im"].autoscale()          # per-file color scale
        axes_heat[key].set_xlim(-0.5, n_t - 0.5)

        # Peak overlays
        objs["sc1"].set_offsets(np.c_[time_steps, peak1])
        objs["sc2"].set_offsets(np.c_[time_steps, peak2])

        # Diff scatter
        intensity_plot = np.nan_to_num(intensity_diffs, nan=0.0)
        colors = ["red" if np.isnan(v) else "steelblue" for v in intensity_diffs]
        objs["sc_diff"].set_offsets(np.c_[time_steps, intensity_plot])
        objs["sc_diff"].set_color(colors)
        axes_diff[key].set_xlim(-0.5, n_t - 0.5)

    # Sync all diff y-axes to global max after update
    sync_diff_ylim(keys_ordered)

    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()