#!/usr/bin/env python3

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from scipy.signal import find_peaks


# ============================================================
# CONFIGURATION
# ============================================================

# Experiment folder
EXPERIMENT_FOLDER = Path("./mergedData/")

# ------------------------------------------------------------
# Select files by filename prefix
# ------------------------------------------------------------
#
# Example:
#
#     FILE_PREFIX = "19_08"
#
# This loads every CSV whose filename starts with "19_08".
#
FILE_PREFIX = "16_08"


# Robot trajectory columns
X_COLUMN = "robot_x"
Y_COLUMN = "robot_y"


# DOA columns
DOA_COLUMNS = [f"doa_{i}" for i in range(120)]


# ============================================================
# PLOT SETTINGS
# ============================================================

# Thickness of the trajectory
TRAJECTORY_LINE_WIDTH = 2.0

# Transparency of trajectory
TRAJECTORY_LINE_ALPHA = 0.4

# Colour map for >=2 peak points
COLORMAP = "viridis"

# Maximum difference represented by the colour scale.
#
# Any difference >= this value receives the maximum
# viridis colour.
#
MAX_COLOR_DIFFERENCE = 0.20

# Colour for 0 or 1 peak
SPECIAL_PEAK_COLOR = "black"


# ============================================================
# NATURAL SORTING
# ============================================================

def natural_sort_key(path):
    """
    Natural filename ordering.

    Example:

        19_08_2.csv
        19_08_10.csv

    rather than:

        19_08_10.csv
        19_08_2.csv
    """

    name = path.name.lower()

    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", name)
    ]


# ============================================================
# DATA LOADING
# ============================================================

def load_experiment(csv_file):
    """
    Load one experiment CSV.

    Returns only rows with valid X/Y trajectory coordinates.
    """

    df = pd.read_csv(csv_file)

    required_columns = [
        X_COLUMN,
        Y_COLUMN,
        *DOA_COLUMNS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{csv_file.name} is missing required columns: "
            f"{missing}"
        )

    if len(df) == 0:
        raise ValueError(
            f"{csv_file.name} contains no rows."
        )

    # --------------------------------------------------------
    # Convert trajectory columns to numeric
    # --------------------------------------------------------

    df[X_COLUMN] = pd.to_numeric(
        df[X_COLUMN],
        errors="coerce",
    )

    df[Y_COLUMN] = pd.to_numeric(
        df[Y_COLUMN],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Convert DOA columns to numeric
    # --------------------------------------------------------

    for column in DOA_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Keep rows that can be plotted
    # --------------------------------------------------------

    plot_df = df.dropna(
        subset=[
            X_COLUMN,
            Y_COLUMN,
        ]
    ).copy()

    if len(plot_df) == 0:
        raise ValueError(
            f"{csv_file.name} has no valid "
            f"{X_COLUMN}/{Y_COLUMN} points."
        )

    return plot_df


# ============================================================
# DOA PEAK ANALYSIS
# ============================================================

def analyze_doa_row(row):
    """
    Analyze doa_0 ... doa_119 for one trajectory point.

    For 0 peaks:
        difference = None

    For 1 peak:
        difference = None

    For >=2 peaks:
        take the two highest peaks by intensity and calculate:

            abs(peak1 - peak2)
    """

    doa_values = row[
        DOA_COLUMNS
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Handle NaNs
    # --------------------------------------------------------

    if pd.isna(doa_values).all():

        return {
            "number_of_peaks": 0,
            "difference": None,
        }

    finite_mask = ~pd.isna(
        doa_values
    )

    if not finite_mask.all():

        doa_values = doa_values.copy()

        # NaNs cannot become peaks
        doa_values[
            ~finite_mask
        ] = float("-inf")

    # --------------------------------------------------------
    # Find peaks
    #
    # Same default find_peaks() settings as the original
    # script.
    # --------------------------------------------------------

    peaks, properties = find_peaks(
        doa_values
    )

    if len(peaks) == 0:

        return {
            "number_of_peaks": 0,
            "difference": None,
        }

    # --------------------------------------------------------
    # Sort peaks by intensity
    # Highest first
    # --------------------------------------------------------

    peak_values_all = (
        doa_values[peaks]
    )

    order = sorted(
        range(len(peaks)),
        key=lambda i: peak_values_all[i],
        reverse=True,
    )

    sorted_peaks = [
        int(peaks[i])
        for i in order
    ]

    sorted_values = [
        float(peak_values_all[i])
        for i in order
    ]

    # --------------------------------------------------------
    # Exactly one peak
    # --------------------------------------------------------

    if len(sorted_peaks) == 1:

        return {
            "number_of_peaks": 1,
            "difference": None,
        }

    # --------------------------------------------------------
    # Two or more peaks
    #
    # Take the two highest peaks.
    # --------------------------------------------------------

    peak1_value = sorted_values[0]
    peak2_value = sorted_values[1]

    difference = abs(
        peak1_value - peak2_value
    )

    return {
        "number_of_peaks": len(sorted_peaks),
        "difference": difference,
    }


# ============================================================
# ANALYZE ONE EXPERIMENT
# ============================================================

def analyze_experiment(csv_file):
    """
    Load one file and calculate the DOA information
    for every trajectory point.
    """

    plot_df = load_experiment(
        csv_file
    )

    doa_analysis = [
        analyze_doa_row(row)
        for _, row in plot_df.iterrows()
    ]

    differences = [
        result["difference"]
        for result in doa_analysis
    ]

    return (
        plot_df,
        doa_analysis,
        differences,
    )


# ============================================================
# BUILD COLOURED TRAJECTORY SEGMENTS
# ============================================================

def build_segments(
    x,
    y,
    doa_analysis,
    differences,
):
    """
    Build one line segment for every pair of consecutive
    trajectory points.

    IMPORTANT:

        P0 -> P1 uses D0
        P1 -> P2 uses D1
        P2 -> P3 uses D2
        ...

    The final difference DN is not used because there is
    no segment after PN.

    Returns:
        segments
        values
        special_colors
    """

    segments = []
    values = []
    special_colors = []

    # --------------------------------------------------------
    # There are N points and therefore N-1 segments.
    # --------------------------------------------------------

    for i in range(
        len(x) - 1
    ):

        # ----------------------------------------------------
        # Segment:
        #
        # P_i -> P_(i+1)
        #
        # is coloured using D_i.
        # ----------------------------------------------------

        segment = [
            [
                x[i],
                y[i],
            ],
            [
                x[i + 1],
                y[i + 1],
            ],
        ]

        segments.append(
            segment
        )

        result = doa_analysis[i]

        number_of_peaks = (
            result["number_of_peaks"]
        )

        difference = (
            differences[i]
        )

        # ----------------------------------------------------
        # 2+ peaks:
        #
        # Use continuous colour scale.
        # ----------------------------------------------------

        if (
            number_of_peaks >= 2
            and difference is not None
        ):

            values.append(
                difference
            )

            special_colors.append(
                None
            )

        # ----------------------------------------------------
        # 0 or 1 peak:
        #
        # Black segment.
        # ----------------------------------------------------

        else:

            values.append(
                None
            )

            special_colors.append(
                SPECIAL_PEAK_COLOR
            )

    return (
        segments,
        values,
        special_colors,
    )


# ============================================================
# PLOT ALL SELECTED TRAJECTORIES
# ============================================================

def plot_selected_trajectories(
    csv_files
):
    """
    Plot all matching trajectories on the same axes.

    The trajectory itself is colour-coded rather than using
    individual point markers.
    """

    # --------------------------------------------------------
    # Analyse all files first
    # --------------------------------------------------------
    #
    # This allows us to use one common colour normalization
    # for every trajectory.
    #

    experiments = []

    for csv_file in csv_files:

        print(
            f"Loading: {csv_file.name}"
        )

        try:

            (
                plot_df,
                doa_analysis,
                differences,
            ) = analyze_experiment(
                csv_file
            )

            experiments.append({
                "file": csv_file,
                "plot_df": plot_df,
                "doa_analysis": doa_analysis,
                "differences": differences,
            })

        except Exception as exc:

            print(
                f"WARNING: Could not load "
                f"{csv_file.name}"
            )

            print(
                f"         {exc}"
            )

    if not experiments:

        print(
            "\nNo valid files could be loaded."
        )

        return

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Common normalization
    # --------------------------------------------------------

    color_norm = Normalize(
        vmin=0.0,
        vmax=MAX_COLOR_DIFFERENCE,
        clip=True,
    )

    cmap = plt.get_cmap(
        COLORMAP
    )

    # --------------------------------------------------------
    # Keep track of whether any continuous-colour segment
    # exists.
    # --------------------------------------------------------

    has_coloured_segments = False

    # --------------------------------------------------------
    # Process each trajectory
    # --------------------------------------------------------

    for experiment in experiments:

        csv_file = experiment[
            "file"
        ]

        plot_df = experiment[
            "plot_df"
        ]

        doa_analysis = experiment[
            "doa_analysis"
        ]

        differences = experiment[
            "differences"
        ]

        x = plot_df[
            X_COLUMN
        ].to_numpy()

        y = plot_df[
            Y_COLUMN
        ].to_numpy()

        # ----------------------------------------------------
        # Need at least two points for a trajectory segment.
        # ----------------------------------------------------

        if len(x) < 2:
            continue

        # ----------------------------------------------------
        # Build segments
        # ----------------------------------------------------

        (
            segments,
            values,
            special_colors,
        ) = build_segments(
            x,
            y,
            doa_analysis,
            differences,
        )

        # ----------------------------------------------------
        # Separate coloured and black segments.
        # ----------------------------------------------------

        coloured_segments = []
        coloured_values = []

        black_segments = []

        for (
            segment,
            value,
            special_color,
        ) in zip(
            segments,
            values,
            special_colors,
        ):

            if value is not None:

                coloured_segments.append(
                    segment
                )

                coloured_values.append(
                    value
                )

            else:

                black_segments.append(
                    segment
                )

        # ----------------------------------------------------
        # Draw continuous-colour segments
        # ----------------------------------------------------

        if coloured_segments:

            has_coloured_segments = True

            coloured_collection = (
                LineCollection(
                    coloured_segments,
                    cmap=cmap,
                    norm=color_norm,
                    linewidths=TRAJECTORY_LINE_WIDTH,
                    alpha=TRAJECTORY_LINE_ALPHA,
                    capstyle="round",
                    joinstyle="round",
                )
            )

            coloured_collection.set_array(
                np.asarray(
                    coloured_values,
                    dtype=float,
                )
            )

            ax.add_collection(
                coloured_collection
            )

        # ----------------------------------------------------
        # Draw black segments
        # ----------------------------------------------------

        if black_segments:

            black_collection = (
                LineCollection(
                    black_segments,
                    colors=SPECIAL_PEAK_COLOR,
                    linewidths=TRAJECTORY_LINE_WIDTH,
                    alpha=TRAJECTORY_LINE_ALPHA,
                    capstyle="round",
                    joinstyle="round",
                )
            )

            ax.add_collection(
                black_collection
            )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    if has_coloured_segments:

        scalar_mappable = ScalarMappable(
            norm=color_norm,
            cmap=cmap,
        )

        scalar_mappable.set_array(
            np.array([])
        )

        colorbar = fig.colorbar(
            scalar_mappable,
            ax=ax,
            pad=0.02,
        )

        colorbar.set_label(
            "Peak intensity difference Δ"
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        X_COLUMN
    )

    ax.set_ylabel(
        Y_COLUMN
    )

    ax.set_title(
        f"Trajectories: files starting with "
        f"'{FILE_PREFIX}'\n"
        f"{len(experiments)} file(s)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    # --------------------------------------------------------
    # Make sure all LineCollection objects are included in
    # the automatic axis limits.
    # --------------------------------------------------------

    ax.autoscale()

    # --------------------------------------------------------
    # NO LEGEND
    # --------------------------------------------------------

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    folder = (
        EXPERIMENT_FOLDER
        .expanduser()
        .resolve()
    )

    # --------------------------------------------------------
    # Check folder
    # --------------------------------------------------------

    if not folder.exists():

        print(
            f"ERROR: folder does not exist:\n"
            f"{folder}"
        )

        sys.exit(1)

    if not folder.is_dir():

        print(
            f"ERROR: not a directory:\n"
            f"{folder}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Find all CSV files starting with FILE_PREFIX
    # --------------------------------------------------------

    csv_files = [
        path
        for path in folder.glob("*.csv")
        if path.name.startswith(
            FILE_PREFIX
        )
    ]

    csv_files.sort(
        key=natural_sort_key
    )

    # --------------------------------------------------------
    # Check matches
    # --------------------------------------------------------

    if not csv_files:

        print(
            f"No CSV files found starting with "
            f"'{FILE_PREFIX}' in:"
        )

        print(
            f"{folder}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print("=" * 60)

    print(
        "Trajectory Plotter"
    )

    print("=" * 60)

    print(
        f"Folder:       {folder}"
    )

    print(
        f"File prefix:  {FILE_PREFIX}"
    )

    print(
        f"Files found:  {len(csv_files)}"
    )

    print("=" * 60)

    print(
        "\nMatching files:"
    )

    for csv_file in csv_files:

        print(
            f"  {csv_file.name}"
        )

    print("=" * 60)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_selected_trajectories(
        csv_files
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()