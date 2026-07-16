"""
Plot robot_x / robot_y paths from CSV files, grouped by "type" (date_group prefix)
and split into two batches: files 01-05 and files 06-10 (if present).

Expected filename pattern:  <prefix>_<index>.csv
    e.g. 09_07_AA_01.csv  -> prefix = "09_07_AA", index = 1
         16_07_EA_05.csv  -> prefix = "16_07_EA", index = 5   (no 06-10 -> batch 2 skipped)

Each file's robot_x and robot_y columns are min-max normalized independently
(per file) before plotting, and all files belonging to the same batch are
overlaid on a single plot.

Usage:
    python plot_robot_paths.py

Edit INPUT_DIR / OUTPUT_DIR below to match your setup.
"""

import os
import re
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never try to open a display / show a window
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Config - edit these to match your folders
INPUT_DIR = "./mergedData/"
OUTPUT_DIR = "./mergedData/BatchPlots/"
# ------------------------------------------------------------------ #

FILENAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{2})\.csv$")


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a pandas Series to [0, 1]. Handles constant columns safely."""
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return series * 0.0  # constant column -> flat at 0
    return (series - lo) / (hi - lo)


def group_files(input_dir: str):
    """Group csv files by prefix -> {index: filepath}."""
    groups = {}
    for filepath in glob.glob(os.path.join(input_dir, "*.csv")):
        filename = os.path.basename(filepath)
        m = FILENAME_RE.match(filename)
        if not m:
            print(f"Skipping file that doesn't match naming pattern: {filename}")
            continue
        prefix = m.group("prefix")
        index = int(m.group("index"))
        groups.setdefault(prefix, {})[index] = filepath
    return groups


def plot_batch(prefix: str, batch_name: str, filepaths: dict, output_dir: str):
    """Overlay normalized robot_x/robot_y paths for all files in this batch and save."""
    if not filepaths:
        return  # nothing to plot for this batch

    fig, ax = plt.subplots(figsize=(8, 8))

    for index in sorted(filepaths.keys()):
        filepath = filepaths[index]
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Could not read {filepath}: {e}")
            continue

        if "robot_x" not in df.columns or "robot_y" not in df.columns:
            print(f"robot_x/robot_y not found in {filepath}, skipping.")
            continue

        # x_norm = min_max_normalize(df["robot_x"])
        # y_norm = min_max_normalize(df["robot_y"])        
        x_norm = df["robot_x"]
        y_norm = df["robot_y"]

        ax.plot(x_norm, y_norm, marker="o", markersize=2, linewidth=1,
                label=os.path.basename(filepath))

    ax.set_title(f"{prefix} - {batch_name}")
    ax.set_xlabel("robot_x (normalized)")
    ax.set_ylabel("robot_y (normalized)")
    # ax.set_xlim(-0.05, 1.05)
    # ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}_{batch_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    groups = group_files(INPUT_DIR)

    for prefix, files_by_index in sorted(groups.items()):
        batch1 = {i: fp for i, fp in files_by_index.items() if 1 <= i <= 5}
        batch2 = {i: fp for i, fp in files_by_index.items() if 6 <= i <= 10}

        plot_batch(prefix, "batch1", batch1, OUTPUT_DIR)
        plot_batch(prefix, "batch2", batch2, OUTPUT_DIR)


if __name__ == "__main__":
    main()