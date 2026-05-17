"""
plot_trajectories.py
--------------------
Plots the 2-D trajectories of the robot and two targets from viccon CSV files.
Colour encodes the row index (timestep) so progression through time is visible.
Targets are shown as tight scatter clouds (they don't move).

Output: expA/expA1_trajectories.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = "expA"
ROBOT_CSV   = os.path.join(BASE_DIR, "expA1_robot.csv")
TARGET1_CSV = os.path.join(BASE_DIR, "expA1_target1.csv")
TARGET2_CSV = os.path.join(BASE_DIR, "expA1_target2.csv")

# ── style ────────────────────────────────────────────────────────────────────
CMAP_ROBOT   = cm.plasma          # colourmap for robot trajectory
CMAP_T1      = cm.cool            # colourmap for target 1 cloud
CMAP_T2      = cm.autumn          # colourmap for target 2 cloud
BG_COLOR     = "#0d0d0d"
GRID_COLOR   = "#2a2a2a"
TEXT_COLOR   = "#e8e8e8"

plt.rcParams.update({
    "font.family":      "monospace",
    "text.color":       TEXT_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
})

# ── load data ────────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)          # index = timestep
    df.index.name = "timestep"
    return df

print("Loading CSVs …")
robot   = load(ROBOT_CSV)
target1 = load(TARGET1_CSV)
target2 = load(TARGET2_CSV)

# normalise index to [0, 1] for each dataset independently
def norm_idx(df):
    idx = df.index.to_numpy().astype(float)
    return (idx - idx.min()) / (idx.max() - idx.min() + 1e-9)

r_t  = norm_idx(robot)
t1_t = norm_idx(target1)
t2_t = norm_idx(target2)

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- robot trajectory: line + scatter ---------------------------------------
# thin connecting line first so dots sit on top
ax.plot(robot["x"], robot["y"],
        color="#ffffff", alpha=0.08, linewidth=0.6, zorder=1)

sc_robot = ax.scatter(
    robot["x"], robot["y"],
    c=r_t, cmap=CMAP_ROBOT,
    s=6, alpha=0.85, zorder=2,
    label="Robot"
)

# start / end markers
ax.scatter(*robot[["x","y"]].iloc[0],
           marker="o", s=120, color=CMAP_ROBOT(0.0),
           edgecolors="white", linewidths=0.8, zorder=5)
ax.scatter(*robot[["x","y"]].iloc[-1],
           marker="*", s=180, color=CMAP_ROBOT(1.0),
           edgecolors="white", linewidths=0.8, zorder=5)

# --- target 1 cloud ---------------------------------------------------------
sc_t1 = ax.scatter(
    target1["x"], target1["y"],
    c=t1_t, cmap=CMAP_T1,
    s=12, alpha=0.55, zorder=3,
    label="Target 1"
)
# centroid marker
ax.scatter(target1["x"].mean(), target1["y"].mean(),
           marker="D", s=120, color=CMAP_T1(0.5),
           edgecolors="white", linewidths=1.0, zorder=6)

# --- target 2 cloud ---------------------------------------------------------
sc_t2 = ax.scatter(
    target2["x"], target2["y"],
    c=t2_t, cmap=CMAP_T2,
    s=12, alpha=0.55, zorder=3,
    label="Target 2"
)
ax.scatter(target2["x"].mean(), target2["y"].mean(),
           marker="D", s=120, color=CMAP_T2(0.5),
           edgecolors="white", linewidths=1.0, zorder=6)

# ── colourbar (robot timestep) ───────────────────────────────────────────────
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.12)
cb = plt.colorbar(sc_robot, cax=cax)
cb.set_label("Timestep index (robot)", fontsize=9, labelpad=8)
cb.ax.yaxis.set_tick_params(color=TEXT_COLOR)
cb.outline.set_edgecolor(GRID_COLOR)
n = len(robot)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.set_ticklabels([f"{int(v*n)}" for v in [0, 0.25, 0.5, 0.75, 1.0]])

# ── legend with proxy patches ────────────────────────────────────────────────
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=CMAP_ROBOT(0.5), markeredgecolor="white",
           markersize=8, label="Robot trajectory"),
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=CMAP_T1(0.5), markeredgecolor="white",
           markersize=8, label="Target 1 cloud"),
    Line2D([0], [0], marker="D", color="none",
           markerfacecolor=CMAP_T1(0.5), markeredgecolor="white",
           markersize=7, label="Target 1 centroid"),
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=CMAP_T2(0.5), markeredgecolor="white",
           markersize=8, label="Target 2 cloud"),
    Line2D([0], [0], marker="D", color="none",
           markerfacecolor=CMAP_T2(0.5), markeredgecolor="white",
           markersize=7, label="Target 2 centroid"),
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=CMAP_ROBOT(0.0), markeredgecolor="white",
           markersize=8, label="Robot start"),
    Line2D([0], [0], marker="*", color="none",
           markerfacecolor=CMAP_ROBOT(1.0), markeredgecolor="white",
           markersize=10, label="Robot end"),
]

# ── cosmetics ────────────────────────────────────────────────────────────────
ax.set_xlabel("X  (mm)", fontsize=10)
ax.set_ylabel("Y  (mm)", fontsize=10)
ax.set_title("Viccon Trajectories — Robot & Targets\n(colour = timestep index)",
             fontsize=13, pad=14)
ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--")
ax.set_aspect("equal")

plt.tight_layout()
plt.show()