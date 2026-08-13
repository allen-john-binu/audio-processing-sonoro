#!/usr/bin/env python3
"""
Plot a heatmap of bump angle distributions over time.

Input:
    CSV produced by the Ring Attractor script.

Expected columns:
    timestamp, bump1, bump2, ..., bump50

Usage:
    python plot_bump_heatmap.py bumps.csv

Output:
    bumps_heatmap.png
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

ANGLE_MIN = -180
ANGLE_MAX = 180
BIN_SIZE = 5

ANGLE_BINS = np.arange(ANGLE_MIN, ANGLE_MAX + BIN_SIZE, BIN_SIZE)


# ----------------------------------------------------------------------
def plot_heatmap(csv_file):
    df = pd.read_csv(csv_file)

    # ------------------------------------------------------------------
    # Check required columns
    # ------------------------------------------------------------------
    required = ["timestamp"] + [f"bump{i}" for i in range(1, 51)]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    timestamps = df["timestamp"].to_numpy()

    bump_cols = [f"bump{i}" for i in range(1, 51)]
    bump_angles = df[bump_cols].to_numpy()

    n_times = len(timestamps)
    n_bins = len(ANGLE_BINS) - 1

    # ------------------------------------------------------------------
    # Build heatmap
    # heatmap[t, b] = number of bump angles in bin b at time t
    # ------------------------------------------------------------------
    heatmap = np.zeros((n_times, n_bins), dtype=int)

    for t in range(n_times):
        counts, _ = np.histogram(
            bump_angles[t],
            bins=ANGLE_BINS
        )
        heatmap[t] = counts

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 6))

    x_edges = np.arange(n_times + 1)
    y_edges = ANGLE_BINS

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        heatmap.T,
        shading="flat",
        cmap="viridis"
    )

    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Number of bump angles")

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Bump Angle Distribution Over Time")

    ax.set_ylim([-180, 180])
    ax.set_yticks(np.arange(-180, 181, 30))

    ax.set_xlim([0, n_times])

    plt.tight_layout()

    output_file = (
        os.path.splitext(csv_file)[0]
        + "_heatmap.png"
    )

    plt.savefig(output_file, dpi=300)
    print(f"Saved: {output_file}")

    plt.show()


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot bump angle heatmap."
    )

    parser.add_argument(
        "csv_file",
        help="Output CSV from Ring Attractor"
    )

    args = parser.parse_args()

    plot_heatmap(args.csv_file)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()