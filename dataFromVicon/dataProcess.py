"""
compute_angle.py
----------------
Batch version — walks all experiment folders and processes every run.

Directory convention (auto-discovered):
  <base_dir>/
    expA/         expBB/        expBC/  ...
      expA1_robot.csv
      expA1_target1.csv
      expA1_target2.csv
      expA1.csv          ← OUTPUT (timestamp, angle)
      expA2_robot.csv
      ...

Sign convention:
  + (positive)  →  target1 is to the LEFT  of target2 from the robot's perspective
  - (negative)  →  target1 is to the RIGHT of target2 from the robot's perspective

Usage:
  python compute_angle.py                  # auto-discovers from current directory
  python compute_angle.py /path/to/data    # explicit base directory
  python compute_angle.py --dry-run        # show what would be processed, no writes
  python compute_angle.py --overwrite      # reprocess runs that already have output
  python compute_angle.py --eps 10 --min-samples 5   # tune DBSCAN
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


# ── DBSCAN defaults (override via CLI flags) ──────────────────────────────────
DEFAULT_EPS         = 5.0
DEFAULT_MIN_SAMPLES = 10


# ── helpers ───────────────────────────────────────────────────────────────────
def dominant_position(csv_path: str, eps: float, min_samples: int):
    """Return (x, y) centroid of the largest DBSCAN cluster in a target CSV."""
    df     = pd.read_csv(csv_path)
    coords = df[["x", "y"]].values

    db     = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique) == 0:
        raise ValueError(
            f"DBSCAN found no clusters in {csv_path}. "
            f"Try increasing --eps (current={eps}) or decreasing --min-samples (current={min_samples})."
        )
    dominant_label = unique[np.argmax(counts)]
    cluster_pts    = coords[labels == dominant_label]
    return cluster_pts.mean(axis=0)   # (cx, cy)


def signed_angle_deg(rx, ry, x1, y1, x2, y2):
    """
    Signed angle from vector (robot→target1) to vector (robot→target2).
    Positive  →  target1 is to the LEFT  of target2 (CCW).
    Range: (-180°, +180°].
    """
    v1    = np.array([x1 - rx, y1 - ry], dtype=float)
    v2    = np.array([x2 - rx, y2 - ry], dtype=float)
    cross = v2[0] * v1[1] - v2[1] * v1[0]   # z of (v2 × v1)
    dot   = np.dot(v1, v2)
    return np.degrees(np.arctan2(cross, dot))


def process_run(robot_path, t1_path, t2_path, out_path, eps, min_samples, dry_run):
    """Cluster targets → compute angles → write output CSV. Returns True on success."""
    try:
        (x1, y1) = dominant_position(t1_path, eps, min_samples)
        (x2, y2) = dominant_position(t2_path, eps, min_samples)

        robot          = pd.read_csv(robot_path)
        robot["angle"] = robot.apply(
            lambda row: signed_angle_deg(row["x"], row["y"], x1, y1, x2, y2),
            axis=1,
        )

        # add fixed target positions to every row
        robot["x1"] = x1
        robot["y1"] = y1
        robot["x2"] = x2
        robot["y2"] = y2

        # rename robot coordinates for clarity
        robot["robot_x"] = robot["x"]
        robot["robot_y"] = robot["y"]

        if not dry_run:
            robot[
                [
                    "timestamp",
                    "angle",
                    "x1", "y1",
                    "x2", "y2",
                    "robot_x", "robot_y",
                ]
            ].to_csv(out_path, index=False)
        return True

    except Exception as exc:
        print(f"      [ERROR] {exc}")
        return False


# ── discovery ─────────────────────────────────────────────────────────────────
def discover_runs(base_dir):
    """
    Yield tuples (folder_name, prefix, robot_path, t1_path, t2_path, out_path)
    for every complete run found under base_dir, sorted by folder then run number.
    A run is complete when all three source CSVs exist.
    """
    runs = []
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        robot_files = [f for f in os.listdir(folder_path) if f.endswith("_robot.csv")]

        def run_number(fname):
            m = re.search(r"(\d+)", fname)
            return int(m.group()) if m else 0

        for robot_file in sorted(robot_files, key=run_number):
            prefix   = robot_file[: -len("_robot.csv")]
            t1_path  = os.path.join(folder_path, f"{prefix}_target1.csv")
            t2_path  = os.path.join(folder_path, f"{prefix}_target2.csv")
            out_path = os.path.join(folder_path, f"{prefix}.csv")

            if not os.path.isfile(t1_path) or not os.path.isfile(t2_path):
                print(f"  [SKIP] {prefix}: missing target file(s) — skipping")
                continue

            runs.append((folder, prefix,
                         os.path.join(folder_path, robot_file),
                         t1_path, t2_path, out_path))
    return runs


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Batch-compute signed bearing angle for all viccon runs."
    )
    parser.add_argument(
        "base_dir", nargs="?", default=".",
        help="Root directory containing experiment folders (default: current dir)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List runs without writing any output"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reprocess runs that already have an output CSV"
    )
    parser.add_argument("--eps",         type=float, default=DEFAULT_EPS,
                        help=f"DBSCAN neighbourhood radius (default: {DEFAULT_EPS})")
    parser.add_argument("--min-samples", type=int,   default=DEFAULT_MIN_SAMPLES,
                        help=f"DBSCAN min_samples (default: {DEFAULT_MIN_SAMPLES})")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    print(f"Base directory : {base_dir}")
    print(f"DBSCAN params  : eps={args.eps}, min_samples={args.min_samples}")
    print(f"Dry run        : {args.dry_run}")
    print(f"Overwrite      : {args.overwrite}\n")

    runs = discover_runs(base_dir)
    if not runs:
        print("No runs found. Check that the base directory contains experiment sub-folders.")
        sys.exit(1)

    folders = sorted({r[0] for r in runs})
    print(f"Discovered {len(runs)} run(s) across {len(folders)} folder(s).\n")

    ok = skipped = errors = 0
    current_folder = None

    for folder, prefix, robot_path, t1_path, t2_path, out_path in runs:
        if folder != current_folder:
            current_folder = folder
            print(f"── {folder} " + "─" * max(0, 52 - len(folder)))

        already_exists = os.path.isfile(out_path)

        if already_exists and not args.overwrite and not args.dry_run:
            print(f"  [EXISTS]  {prefix}.csv  (use --overwrite to redo)")
            skipped += 1
            continue

        tag = "[DRY] " if args.dry_run else ""
        print(f"  {tag}{prefix}  →  {prefix}.csv", end="", flush=True)

        success = process_run(
            robot_path, t1_path, t2_path, out_path,
            args.eps, args.min_samples, args.dry_run
        )

        if success:
            print("  ✓")
            ok += 1
        else:
            print("  ✗")
            errors += 1

    print()
    if args.dry_run:
        print(f"Dry run complete — {len(runs)} run(s) would be processed.")
    else:
        print(f"Done.  ✓ {ok} written   skip {skipped}   ✗ {errors} error(s)")


if __name__ == "__main__":
    main()