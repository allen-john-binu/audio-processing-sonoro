#!/usr/bin/env python3

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

ANGLE_START = 2

# Colors assigned dynamically to each group
COLOR_CYCLE = [
    "tab:blue",
    "tab:green",
    "tab:red",
    "tab:orange",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


def compute_intensity_difference(csv_path):
    """
    Returns ONLY timesteps where >=2 peaks exist.
    Time is compressed after filtering.
    """

    df = pd.read_csv(csv_path)
    intensity_matrix = df.iloc[:, ANGLE_START:].values.astype(float)

    diffs = []

    for row in intensity_matrix:
        peaks, _ = signal.find_peaks(row)

        if len(peaks) < 2:
            continue

        top2 = peaks[np.argsort(row[peaks])[-2:]]
        diffs.append(abs(row[top2[0]] - row[top2[1]]))

    return np.asarray(diffs)


def get_group(filename):
    """
    30_07_01_F_01.csv
        ->
    30_07_01_F
    """
    return "_".join(filename.stem.split("_")[:-1])


def main():

    if len(sys.argv) != 2:
        print("Usage: python3 plot_intensity_groups.py <folder>")
        sys.exit(1)

    folder = pathlib.Path(sys.argv[1])

    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        print("No CSV files found.")
        sys.exit(1)

    # Assign colors to groups automatically
    groups = sorted({get_group(f) for f in csv_files})
    color_map = {
        group: COLOR_CYCLE[i % len(COLOR_CYCLE)]
        for i, group in enumerate(groups)
    }

    plt.figure(figsize=(14, 7))

    for csv_file in csv_files:

        diffs = compute_intensity_difference(csv_file)

        plt.plot(
            np.arange(len(diffs)),
            diffs,
            color=color_map[get_group(csv_file)],
            linewidth=2,
            label=csv_file.stem,
        )

    plt.xlabel("Filtered Time Step")
    plt.ylabel("|Peak1 - Peak2|")
    plt.title("Peak Intensity Difference")

    plt.grid(alpha=0.3)

    plt.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()