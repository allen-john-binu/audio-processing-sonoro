"""
Reconstruct robot trajectories (batch 1 only, files 01-05) as a fixed
A -> B -> C two-segment line map (Total Least Squares / PCA line fitting
with breakpoint search), and pair each file's reconstruction with the
mode of its noisy speaker (L/R) position data.

Expected filename pattern:  <prefix>_<index>.csv
    e.g. 09_07_AA_01.csv  -> prefix = "09_07_AA", index = 1

Only batch 1 (index 01-05) is processed. Batch 2 (06-10) is ignored entirely.

Line map model
---------------
    A = raw trajectory start point (first row's robot_x/robot_y, as-is)
    C = raw trajectory end point   (last row's robot_x/robot_y, as-is)
    B = estimated turning/corner point

To find B, the ordered trajectory points are split at a breakpoint index k.
A TLS (PCA) line is fit to the points before k and another to the points
from k onward. We search over all valid k and keep the one that minimizes
the *total* perpendicular fitting error (sum over both segments). B is then
the intersection of the two fitted lines (not necessarily a raw data point).

Speaker points
--------------
robot_x/y and speakerL/R_x/y are continuous, noisy (Qualisys) floats, so an
exact statistical mode would rarely find repeats. Each speaker's x and y
values are independently rounded to ROUND_DECIMALS and then the mode of the
rounded values is used as the "most common" point.

Output
------
One combined CSV (OUTPUT_CSV), one row per input file, with columns:
    prefix, index, filename,
    Ax, Ay, Bx, By, Cx, Cy, breakpoint_row_index,
    speakerL_x_mode, speakerL_y_mode,
    speakerR_x_mode, speakerR_y_mode

Usage:
    python line_map_reconstruction.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Config - edit these to match your setup
INPUT_DIR = "../mergedData/"
OUTPUT_CSV = "./line_map_reconstruction.csv"

# Decimal precision used before computing the "mode" of the noisy
# speaker (x, y) positions. Qualisys data is continuous, so raw values
# essentially never repeat exactly -> we round first, then take the
# most frequent rounded value. Adjust to match your units:
#   e.g. if x/y are in METERS, 3 decimals ~= 1 mm resolution.
#   if x/y are in MILLIMETERS, use 0 or 1 instead.
ROUND_DECIMALS = 3

# Minimum number of points required in each of the two fitted segments.
MIN_SEGMENT_POINTS = 2
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
    # SVD of centered points: first right-singular vector = best-fit direction
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]
    if Vt.shape[0] > 1:
        normal = Vt[1]
    else:
        # degenerate case (single distinct point): pick an arbitrary
        # perpendicular so downstream code doesn't crash
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
        return None  # not enough points to split into two segments

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


def most_common_point(df: pd.DataFrame, col_x: str, col_y: str, decimals: int):
    """Mode of rounded (x, y) values, taken independently per axis."""
    xr = df[col_x].dropna().round(decimals)
    yr = df[col_y].dropna().round(decimals)
    x_mode = xr.mode()
    y_mode = yr.mode()
    x_val = float(x_mode.iloc[0]) if not x_mode.empty else np.nan
    y_val = float(y_mode.iloc[0]) if not y_mode.empty else np.nan
    return x_val, y_val


def process_file(filepath: str):
    """Build one output row for a single trajectory file, or None on failure."""
    filename = os.path.basename(filepath)
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Could not read {filepath}: {e}")
        return None

    required_cols = ["robot_x", "robot_y",
                      "speakerL_x", "speakerL_y",
                      "speakerR_x", "speakerR_y"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"{filename}: missing columns {missing}, skipping.")
        return None

    traj = df[["robot_x", "robot_y"]].dropna().to_numpy()
    if len(traj) < 2 * MIN_SEGMENT_POINTS:
        print(f"{filename}: not enough valid robot_x/y points, skipping.")
        return None

    # A, C = raw first / last trajectory points (as-is, no smoothing)
    A = traj[0]
    C = traj[-1]

    fit = fit_two_segment_line_map(traj)
    if fit is None:
        print(f"{filename}: could not fit two-segment line map, skipping.")
        return None
    best_k, mean1, dir1, mean2, dir2 = fit

    B = line_intersection(mean1, dir1, mean2, dir2)
    if B is None:
        # fitted lines are (near) parallel -> fall back to the raw
        # trajectory point at the optimal breakpoint index
        print(f"{filename}: fitted lines nearly parallel, using raw breakpoint for B.")
        B = traj[best_k]

    speakerL_x, speakerL_y = most_common_point(df, "speakerL_x", "speakerL_y", ROUND_DECIMALS)
    speakerR_x, speakerR_y = most_common_point(df, "speakerR_x", "speakerR_y", ROUND_DECIMALS)

    m = FILENAME_RE.match(filename)
    prefix, index = m.group("prefix"), int(m.group("index"))

    return {
        "prefix": prefix,
        "index": index,
        "filename": filename,
        "Ax": A[0], "Ay": A[1],
        "Bx": B[0], "By": B[1],
        "Cx": C[0], "Cy": C[1],
        "breakpoint_row_index": best_k,
        "speakerL_x_mode": speakerL_x, "speakerL_y_mode": speakerL_y,
        "speakerR_x_mode": speakerR_x, "speakerR_y_mode": speakerR_y,
    }


def main():
    groups = group_files(INPUT_DIR)
    rows = []

    for prefix, files_by_index in sorted(groups.items()):
        batch1 = {i: fp for i, fp in files_by_index.items() if 1 <= i <= 5}
        for index in sorted(batch1.keys()):
            row = process_file(batch1[index])
            if row is not None:
                rows.append(row)

    if not rows:
        print("No valid batch-1 files processed; nothing to write.")
        return

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()