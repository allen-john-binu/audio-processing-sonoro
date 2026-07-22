import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PLOT PARAMETER STUDY RESULTS
# =============================================================================

def plot_parameter_study(csv_path, output_path=None):
    """
    Plot four parameter-study metrics as a 2x2 grid of heatmaps.

    Expected CSV columns:
        normalizing_factor
        h_b
        v
        beta
        avg_number_of_bumps
        avg_total_active_neurons
        avg_bump_width
        avg_largest_bump_width
    """

    # -------------------------------------------------------------------------
    # Load results
    # -------------------------------------------------------------------------

    df = pd.read_csv(csv_path)

    required_columns = [
        "normalizing_factor",
        "h_b",
        "v",
        "beta",
        "avg_number_of_bumps",
        "avg_total_active_neurons",
        "avg_bump_width",
        "avg_largest_bump_width",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("=" * 70)
    print("PARAMETER STUDY PLOT")
    print("=" * 70)

    print(f"Input CSV          : {csv_path}")
    print(f"Rows               : {len(df)}")

    print(
        "Normalizing factors:",
        sorted(df["normalizing_factor"].unique())
    )

    print(
        "h_b values         :",
        sorted(df["h_b"].unique())
    )

    print(
        "v values           :",
        sorted(df["v"].unique())
    )

    print(
        "beta values        :",
        sorted(df["beta"].unique())
    )

    # -------------------------------------------------------------------------
    # Check parameter grid
    # -------------------------------------------------------------------------

    n_nf = df["normalizing_factor"].nunique()
    n_hb = df["h_b"].nunique()

    expected_rows = n_nf * n_hb

    if len(df) != expected_rows:
        print()
        print(
            f"[WARNING] Found {len(df)} rows, but "
            f"{n_nf} × {n_hb} = {expected_rows} "
            f"parameter combinations are expected."
        )

        print(
            "This may indicate missing combinations, "
            "duplicate rows, or multiple v/beta values."
        )

    # -------------------------------------------------------------------------
    # Metrics to plot
    # -------------------------------------------------------------------------

    metrics = [
        (
            "avg_number_of_bumps",
            "Average Number of Bumps"
        ),
        (
            "avg_total_active_neurons",
            "Average Total Active Neurons"
        ),
        (
            "avg_bump_width",
            "Average Bump Width"
        ),
        (
            "avg_largest_bump_width",
            "Average Largest Bump Width"
        ),
    ]

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 13),
        constrained_layout=True
    )

    axes = axes.flatten()

    # -------------------------------------------------------------------------
    # Plot each metric
    # -------------------------------------------------------------------------

    for ax, (metric, title) in zip(
        axes,
        metrics
    ):

        # Create matrix:
        #
        # rows    = h_b
        # columns = normalizing_factor
        #
        heatmap = df.pivot(
            index="h_b",
            columns="normalizing_factor",
            values=metric
        )

        # Sort both axes numerically ascending.
        heatmap = heatmap.sort_index(
            ascending=True
        )

        heatmap = heatmap.sort_index(
            axis=1,
            ascending=True
        )

        x_values = heatmap.columns.to_numpy()
        y_values = heatmap.index.to_numpy()

        data = heatmap.to_numpy()

        # -------------------------------------------------------------
        # Heatmap
        # -------------------------------------------------------------

        image = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            interpolation="nearest"
        )

        # -------------------------------------------------------------
        # Colorbar
        # -------------------------------------------------------------

        cbar = fig.colorbar(
            image,
            ax=ax
        )

        cbar.set_label(
            title,
            fontsize=11
        )

        # -------------------------------------------------------------
        # Axis tick positions
        # -------------------------------------------------------------

        ax.set_xticks(
            np.arange(
                len(x_values)
            )
        )

        ax.set_yticks(
            np.arange(
                len(y_values)
            )
        )

        # -------------------------------------------------------------
        # Axis tick labels
        # -------------------------------------------------------------

        ax.set_xticklabels(
            [
                f"{x:.2f}"
                for x in x_values
            ],
            rotation=45,
            ha="right"
        )

        ax.set_yticklabels(
            [
                f"{y:.2f}"
                for y in y_values
            ]
        )

        # -------------------------------------------------------------
        # Labels
        # -------------------------------------------------------------

        ax.set_xlabel(
            "NORMALIZING_FACTOR",
            fontsize=12
        )

        ax.set_ylabel(
            "h_b",
            fontsize=12
        )

        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold"
        )

    # -------------------------------------------------------------------------
    # Main title
    # -------------------------------------------------------------------------

    unique_v = df["v"].unique()
    unique_beta = df["beta"].unique()

    if (
        len(unique_v) == 1
        and len(unique_beta) == 1
    ):

        main_title = (
            "Ring Attractor Parameter Study\n"
            f"v = {unique_v[0]}, "
            f"beta = {unique_beta[0]}"
        )

    else:

        main_title = (
            "Ring Attractor Parameter Study"
        )

    fig.suptitle(
        main_title,
        fontsize=18,
        fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Save if requested
    # -------------------------------------------------------------------------

    if output_path:

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        print()
        print(
            f"Saved plot → {output_path}"
        )

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    plt.show()


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Plot ring attractor parameter-study "
            "metrics as four heatmaps."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to parameter_study_results.csv"
        )
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output image path, "
            "e.g. parameter_study_heatmaps.png"
        )
    )

    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    args = parse_args()

    plot_parameter_study(
        csv_path=args.input,
        output_path=args.output
    )