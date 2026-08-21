"""
Plot the raw robot trajectory (solid line) together with its reconstructed
two-segment line map A -> B -> C (dotted line), for batch 1 (files 01-05)
only. All files belonging to the same prefix are overlaid on a single plot,
one distinct colour per file, shared between its raw trajectory and its
line map.

This script is self-contained: it recomputes the line map itself (same TLS
/ breakpoint-search logic as line_map_reconstruction.py) rather than reading
line_map_reconstruction.csv.

Expected filename pattern:  <prefix>_<index>.csv
    e.g. 09_07_AA_01.csv  -> prefix = "09_07_AA", index = 1

Usage:
    python plot_line_map_overlay.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never try to open a display / show a window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ #
# Config - edit these to match your setup
INPUT_DIR = "../mergedData/"
OUTPUT_DIR = "./BatchPlots/"

# Minimum number of points required in each of the two fitted segments.
MIN_SEGMENT_POINTS = 2

# Fixed colour per file position within a batch (up to 5 files: 01-05).
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
# ------------------------------------------------------------------ #

FILENAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{2})\.csv$")


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


def tls_line_fit(points: np.ndarray):
    """
    Total-least-squares (PCA) line fit for a set of 2D points.
    Returns (mean_point, direction_unit_vector, sum_sq_perp_error).
    """
    mean = points.mean(axis=0)
    centered = points - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]
    if Vt.shape[0] > 1:
        normal = Vt[1]
    else:
        normal = np.array([-direction[1], direction[0]])
    perp_residuals = centered @ normal
    sse = float(np.sum(perp_residuals ** 2))
    return mean, direction, sse


def line_intersection(p1, d1, p2, d2):
    """Intersection point of two lines given as point + direction. None if parallel."""
    A = np.column_stack([d1, -d2])
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return None
    b = p2 - p1
    t, _s = np.linalg.solve(A, b)
    return p1 + t * d1


def fit_two_segment_line_map(points: np.ndarray, min_seg: int = MIN_SEGMENT_POINTS):
    """
    Search over breakpoint index k, fit a TLS line to points[:k] and
    points[k:], and return the k that minimizes total perpendicular
    fitting error, together with the two fitted lines (point + direction).
    """
    n = len(points)
    if n < 2 * min_seg:
        return None

    best_k = None
    best_error = None
    for k in range(min_seg, n - min_seg + 1):
        _, _, sse1 = tls_line_fit(points[:k])
        _, _, sse2 = tls_line_fit(points[k:])
        total = sse1 + sse2
        if best_error is None or total < best_error:
            best_error = total
            best_k = k

    mean1, dir1, _ = tls_line_fit(points[:best_k])
    mean2, dir2, _ = tls_line_fit(points[best_k:])
    return best_k, mean1, dir1, mean2, dir2


def compute_abc(filepath: str):
    """Return (A, B, C) as 2-element np.arrays, or None if fitting failed."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Could not read {filepath}: {e}")
        return None

    if "robot_x" not in df.columns or "robot_y" not in df.columns:
        print(f"robot_x/robot_y not found in {filepath}, skipping.")
        return None

    traj = df[["robot_x", "robot_y"]].dropna().to_numpy()
    if len(traj) < 2 * MIN_SEGMENT_POINTS:
        print(f"{os.path.basename(filepath)}: not enough valid points, skipping.")
        return None

    A = traj[0]
    C = traj[-1]

    fit = fit_two_segment_line_map(traj)
    if fit is None:
        return None
    best_k, mean1, dir1, mean2, dir2 = fit

    B = line_intersection(mean1, dir1, mean2, dir2)
    if B is None:
        print(f"{os.path.basename(filepath)}: fitted lines nearly parallel, using raw breakpoint for B.")
        B = traj[best_k]

    return traj, A, B, C


def plot_batch1_overlay(prefix: str, filepaths: dict, output_dir: str):
    """Overlay raw trajectory + line map for all batch-1 files of a prefix."""
    if not filepaths:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    file_handles = []
    b_points = []

    for color, index in zip(COLORS, sorted(filepaths.keys())):
        filepath = filepaths[index]
        result = compute_abc(filepath)
        if result is None:
            continue
        traj, A, B, C = result
        b_points.append(B)

        # raw trajectory - solid line
        ax.plot(traj[:, 0], traj[:, 1], linestyle="-", linewidth=1,
                marker="o", markersize=1.5, color=color, alpha=0.8)

        # reconstructed line map A -> B -> C - dotted line, same colour
        line_map = np.array([A, B, C])
        ax.plot(line_map[:, 0], line_map[:, 1], linestyle=":", linewidth=2,
                color=color)
        ax.plot(line_map[:, 0], line_map[:, 1], linestyle="None",
                marker="o", markersize=5, markeredgecolor="black",
                markerfacecolor=color, zorder=5)

        file_handles.append(
            Line2D([0], [0], color=color, lw=2,
                   label=os.path.basename(filepath))
        )

    if not file_handles:
        print(f"{prefix}: no valid batch-1 files to plot, skipping figure.")
        plt.close(fig)
        return

    # cross mark at the average B (turning point) across this batch's files -
    # visual sanity check that the individual B estimates agree with each other
    avg_B = np.mean(np.array(b_points), axis=0)
    ax.plot(avg_B[0], avg_B[1], marker="x", markersize=14, markeredgewidth=3,
            color="black", linestyle="None", zorder=6)

    ax.set_title(f"{prefix} - batch1: raw trajectory vs. line map")
    ax.set_xlabel("robot_x")
    ax.set_ylabel("robot_y")
    ax.set_aspect("equal", adjustable="box")

    # legend 1: colour -> file
    legend_files = ax.legend(handles=file_handles, fontsize=7,
                              loc="upper left", title="File")
    ax.add_artist(legend_files)

    # legend 2: linestyle -> meaning (raw vs. line map)
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", lw=1.5, label="Raw trajectory"),
        Line2D([0], [0], color="black", linestyle=":", lw=2, label="Line map (A\u2192B\u2192C)"),
        Line2D([0], [0], color="black", marker="x", markersize=10, markeredgewidth=3,
               linestyle="None", label="Average B"),
    ]
    ax.legend(handles=style_handles, fontsize=7, loc="lower right")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}_batch1_trajectories.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    groups = group_files(INPUT_DIR)

    for prefix, files_by_index in sorted(groups.items()):
        batch1 = {i: fp for i, fp in files_by_index.items() if 1 <= i <= 8}
        plot_batch1_overlay(prefix, batch1, OUTPUT_DIR)


if __name__ == "__main__":
    main()