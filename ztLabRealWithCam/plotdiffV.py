"""
Plot robot_x / robot_y trajectories for all files matching "16_07_E*.csv"
on a SINGLE plot. Each individual csv file is drawn as one line, and the
line color is determined by its batch (EA, EB, EC, ED) - so all 5 files
belonging to e.g. EA share the same color, all EB files share another
color, etc. Raw (non-normalized) robot_x/robot_y values are plotted.

Expected filename pattern:  16_07_<batch>_<index>.csv
    e.g. 16_07_EA_01.csv -> batch = "EA"

Usage:
    python plot_grouped_trajectories.py

Edit INPUT_DIR / OUTPUT_DIR / FILE_PREFIX below to match your setup.
"""

import os
import re
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never try to open a display / show a window
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 24,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ------------------------------------------------------------------ #
# Config - edit these to match your folders
INPUT_DIR = "./mergedData/"
OUTPUT_DIR = "./mergedData/BatchPlots/"

# Only files whose name starts with this are included, e.g. "16_07_E"
FILE_PREFIX = "16_07_E"

# Output plot filename (saved inside OUTPUT_DIR)
OUTPUT_NAME = "16_07_E_all_batches.pdf"

# Color assigned to each batch.
BATCH_COLORS = {
    "EA": "blue",
    "EB": "orange",
    "EC": "green",
    "ED": "red",
}

# {
#     "AA": "blue",
#     "AB": "orange",
#     "AD": "green",
#     "AD": "purple",
#     "AE": "red",
#     "CA": "blue",
#     "CB": "orange",
#     "CC": "green",
#     "CD": "red",
#     "DA": "blue",
#     "DB": "orange",
#     "DC": "green",
#     "DD": "red",
# }

# Velocity corresponding to each batch (used in legend)
BATCH_V = {
    "EA": 0.3,
    "EB": 0.4,
    "EC": 0.5,
    "ED": 0.6,
}
# ------------------------------------------------------------------ #

# Matches e.g. "16_07_EA_01.csv" -> batch="EA", index="01"
FILENAME_RE = re.compile(r"^16_07_(?P<batch>[A-Z]+)_(?P<index>\d{2})\.csv$")


def collect_files(input_dir: str, file_prefix: str):
    """Group matching files by batch -> list of filepaths."""
    batches = {}
    pattern = os.path.join(input_dir, f"{file_prefix}*.csv")
    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        m = FILENAME_RE.match(filename)
        if not m:
            print(f"Skipping file that doesn't match naming pattern: {filename}")
            continue
        batch = m.group("batch")
        if int(m.group("index")) > 5:
            continue
        batches.setdefault(batch, []).append(filepath)
    return batches


def plot_all(batches: dict, output_dir: str, output_name: str):
    if not batches:
        print("No matching files found.")
        return

    fig, ax = plt.subplots(figsize=(16, 16))

    for batch in sorted(batches.keys()):
        color = BATCH_COLORS.get(batch)
        if color is None:
            print(f"Warning: no color defined for batch '{batch}', using default cycle.")
            continue
        first_line_for_batch = True
        for filepath in batches[batch]:
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                print(f"Could not read {filepath}: {e}")
                continue

            if "robot_x" not in df.columns or "robot_y" not in df.columns:
                print(f"robot_x/robot_y not found in {filepath}, skipping.")
                continue

            # Only label the first line of each batch so the legend shows
            # one entry per batch instead of one per file.
            if first_line_for_batch:
                if batch in BATCH_V:
                    label = f"ν = {BATCH_V[batch]}"
                else:
                    label = batch
                first_line_for_batch = False
            else:
                label = None

            ax.plot(
                df["robot_x"],
                df["robot_y"],
                color=color,
                linewidth=1,
                alpha=0.8,
                label=label,
            )

    ax.set_title("Trajectories for different shape of the connectivity control")
    ax.set_xlabel("robot_x",fontsize=30)
    ax.set_ylabel("robot_y",fontsize=30)
    ax.legend(title="ν, shape of the connectivity control", fontsize=24, loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    fig.savefig(out_path, dpi=150,
            format="pdf",
            bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    batches = collect_files(INPUT_DIR, FILE_PREFIX)
    plot_all(batches, OUTPUT_DIR, OUTPUT_NAME)


if __name__ == "__main__":
    main()