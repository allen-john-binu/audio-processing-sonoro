import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATA_DIR = Path("k03_45")

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------
csv_files = sorted(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise RuntimeError(f"No csv files found in {DATA_DIR}")

# --------------------------------------------------
# GLOBAL AXIS LIMITS
# --------------------------------------------------
all_data = []
xmin =  math.inf
xmax = -math.inf
ymin =  math.inf
ymax = -math.inf

for file in csv_files:
    df = pd.read_csv(file)

    trial_a = df[df["trial"] == "A"].copy()
    trial_b = df[df["trial"] == "B"].copy()

    if len(trial_a) == 0 or len(trial_b) == 0:
        print(f"Skipping {file.name} — missing trial A or B")
        continue

    all_data.append((file.stem, trial_a, trial_b))

    xmin = min(xmin, trial_a["x"].min(), trial_b["x"].min())
    xmax = max(xmax, trial_a["x"].max(), trial_b["x"].max())
    ymin = min(ymin, trial_a["y"].min(), trial_b["y"].min())
    ymax = max(ymax, trial_a["y"].max(), trial_b["y"].max())

# --------------------------------------------------
# SUBPLOTS
# --------------------------------------------------
n    = len(all_data)
cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

if n == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# --------------------------------------------------
# PLOT
# --------------------------------------------------
for ax, (label, trial_a, trial_b) in zip(axes, all_data):

    # Trial A path
    ax.plot(
        trial_a["x"], trial_a["y"],
        color="tab:blue", linewidth=3, label="Trial A"
    )

    # Trial B path
    ax.plot(
        trial_b["x"], trial_b["y"],
        color="tab:red", linewidth=3, label="Trial B"
    )

    # Start A (absolute)
    ax.scatter(
        trial_a["x"].iloc[0], trial_a["y"].iloc[0],
        color="tab:blue", s=100, marker="o", zorder=5,
        label="Start A"
    )

    # Start B (absolute — different from Start A due to soft reset)
    ax.scatter(
        trial_b["x"].iloc[0], trial_b["y"].iloc[0],
        color="tab:red", s=100, marker="o", zorder=5,
        label="Start B"
    )

    # End A
    ax.scatter(
        trial_a["x"].iloc[-1], trial_a["y"].iloc[-1],
        color="tab:blue", s=120, marker="x", linewidths=2.5, zorder=5,
        label="End A"
    )

    # End B
    ax.scatter(
        trial_b["x"].iloc[-1], trial_b["y"].iloc[-1],
        color="tab:red", s=120, marker="x", linewidths=2.5, zorder=5,
        label="End B"
    )

    ax.set_title(label)
    ax.set_xlim(xmin - 20, xmax + 20)
    ax.set_ylim(ymin - 20, ymax + 20)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True)
    ax.legend(fontsize=7)

# --------------------------------------------------
# REMOVE UNUSED AXES
# --------------------------------------------------
for i in range(len(all_data), len(axes)):
    fig.delaxes(axes[i])

# --------------------------------------------------
# FINALIZE
# --------------------------------------------------
fig.suptitle(DATA_DIR.name, fontsize=16)
plt.tight_layout()
plt.show()