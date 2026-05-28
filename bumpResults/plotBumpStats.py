#!/usr/bin/env python3

"""
plot_bump_stats.py
──────────────────────────────────────────────────────────────────────────────

Interactive + Batch visualization tool for bump-angle statistics.

FEATURES
────────
1. Interactive mode
   - Opens matplotlib GUI
   - Includes dB_SPL slider
   - Dynamically filters rows

2. Batch mode
   - Processes ALL CSV files inside a folder
   - Automatically applies optimal thresholds
   - Saves PNG files
   - Does NOT open GUI windows

USAGE
─────

Interactive mode:
    python3 plot_bump_stats.py file.csv

Batch mode:
    python3 plot_bump_stats.py folder/ --batch

OUTPUT FILES
────────────
Batch mode saves:

    original.csv
        ->
    original_stats.png

EXAMPLE:
    expA_exA1_bumps.csv
        ->
    expA_exA1_bumps_stats.png

──────────────────────────────────────────────────────────────────────────────
"""

import sys
import pathlib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from matplotlib.widgets import Slider


# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

THRESHOLDS_CSV = "../dataFromReal/experiment_thresholds.csv"


# ────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def lookup_optimal_threshold(experiment: str) -> float | None:
    """
    Look up experiment threshold from thresholds CSV.
    """

    try:

        tdf = pd.read_csv(THRESHOLDS_CSV)

        match = tdf[tdf["experiment"] == experiment]

        if match.empty:

            print(
                f"[thresholds] "
                f"No threshold for '{experiment}'"
            )

            return None

        value = float(
            match.iloc[0]["global_threshold_dB"]
        )

        print(
            f"[thresholds] "
            f"{experiment} -> {value} dB_SPL"
        )

        return value

    except FileNotFoundError:

        print(
            f"[thresholds] "
            f"File not found: {THRESHOLDS_CSV}"
        )

        return None

    except Exception as e:

        print(
            f"[thresholds] Error: {e}"
        )

        return None


def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV and validate columns.
    """

    df = pd.read_csv(path)

    missing = {"dB_SPL"} - set(df.columns)

    if missing:

        raise ValueError(
            f"CSV missing columns: {missing}"
        )

    return df


def bump_columns(df: pd.DataFrame) -> list[str]:
    """
    Return:
        bump1, bump2, bump3, ...
    in numerical order.
    """

    cols = [
        c for c in df.columns
        if c.startswith("bump") and c[4:].isdigit()
    ]

    cols.sort(
        key=lambda c: int(c[4:])
    )

    return cols


# ────────────────────────────────────────────────────────────────────────────
# STATISTICS COMPUTATION
# ────────────────────────────────────────────────────────────────────────────

def compute_stats(
    df: pd.DataFrame,
    b_cols: list[str]
) -> dict:
    """
    Compute statistics for each timestamp row.

    OUTPUTS
    ───────
    mean_all:
        Mean of all bump values

    pos_sd:
        Standard deviation of positive values

    neg_sd:
        Standard deviation of negative values

    count_metric:
        (#positive - #negative)
    """

    n = len(df)

    mean_all = np.full(n, np.nan)

    pos_sd = np.full(n, np.nan)

    neg_sd = np.full(n, np.nan)

    count_metric = np.full(n, np.nan)

    bump_vals = df[b_cols].values

    for i in range(n):

        row = bump_vals[i]

        # Remove NaNs
        row = row[~np.isnan(row)]

        if len(row) == 0:
            continue

        # ─────────────────────────────────────────────
        # Mean of all bump values
        # ─────────────────────────────────────────────

        mean_all[i] = np.mean(row)

        # ─────────────────────────────────────────────
        # Positive / Negative split
        # ─────────────────────────────────────────────

        pos = row[row > 0]

        neg = row[row < 0]

        # ─────────────────────────────────────────────
        # Standard deviation
        # ─────────────────────────────────────────────

        if len(pos) > 0:

            pos_sd[i] = np.std(
                pos,
                ddof=0
            )

        if len(neg) > 0:

            neg_sd[i] = np.std(
                neg,
                ddof=0
            )

        # ─────────────────────────────────────────────
        # Count metric
        # ─────────────────────────────────────────────

        count_metric[i] = (
            len(pos) - len(neg)
        )

    return dict(
        mean_all=mean_all,
        pos_sd=pos_sd,
        neg_sd=neg_sd,
        count_metric=count_metric,
    )


# ────────────────────────────────────────────────────────────────────────────
# PLOT CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

POS_COLOR = "#f59e0b"   # amber
NEG_COLOR = "#1e40af"   # deep blue


# ────────────────────────────────────────────────────────────────────────────
# DRAW PLOTS
# ────────────────────────────────────────────────────────────────────────────

def draw_plots(
    df_filtered: pd.DataFrame,
    b_cols: list[str],
    ax_mean,
    ax_sd,
    ax_count,
    fig,
    filename: str
):
    """
    Draw all plot panels.
    """

    # Clear previous plots
    for ax in (ax_mean, ax_sd, ax_count):
        ax.cla()

    n_timesteps = len(df_filtered)

    xs = np.arange(n_timesteps)

    # ─────────────────────────────────────────────
    # Empty data case
    # ─────────────────────────────────────────────

    if n_timesteps == 0:

        ax_mean.set_ylabel("Mean")

        ax_sd.set_ylabel("Std Dev")

        ax_count.set_ylabel("Count")

        ax_count.set_xlabel("Timestamp index")

        for ax in (ax_mean, ax_sd):
            ax.set_xticklabels([])

        fig.canvas.draw_idle()

        return

    stats = compute_stats(
        df_filtered,
        b_cols
    )

    # ─────────────────────────────────────────────
    # PANEL 1 — MEAN
    # ─────────────────────────────────────────────

    ax_mean.plot(
        xs,
        stats["mean_all"],
        color="#111111",
        linewidth=1.2,
        label="mean"
    )

    ax_mean.set_ylabel("Mean")

    ax_mean.set_xlim(0, n_timesteps)

    ax_mean.set_xticklabels([])

    ax_mean.legend(
        fontsize=7,
        loc="upper right",
        framealpha=0.6
    )

    # ─────────────────────────────────────────────
    # PANEL 2 — STANDARD DEVIATION
    # ─────────────────────────────────────────────

    ax_sd.plot(
        xs,
        stats["pos_sd"],
        color=POS_COLOR,
        linewidth=0.9,
        label="positive"
    )

    ax_sd.plot(
        xs,
        stats["neg_sd"],
        color=NEG_COLOR,
        linewidth=0.9,
        label="negative"
    )

    ax_sd.set_ylabel("Std Dev (°)")

    ax_sd.set_xlim(0, n_timesteps)

    ax_sd.set_xticklabels([])

    ax_sd.legend(
        fontsize=7,
        loc="upper right",
        framealpha=0.6
    )

    # ─────────────────────────────────────────────
    # PANEL 3 — COUNT METRIC
    # ─────────────────────────────────────────────

    ax_count.plot(
        xs,
        stats["count_metric"],
        color="#7c3aed",
        linewidth=1.2,
        label="pos_count - neg_count"
    )

    ax_count.axhline(
        0,
        color="#16a34a",
        linewidth=1.0,
        linestyle="-"
    )

    ax_count.set_ylabel("Count")

    ax_count.set_xlabel("Timestamp index")

    ax_count.set_xlim(0, n_timesteps)

    ax_count.legend(
        fontsize=7,
        loc="upper right",
        framealpha=0.6
    )

    fig.canvas.draw_idle()


# ────────────────────────────────────────────────────────────────────────────
# BATCH MODE
# ────────────────────────────────────────────────────────────────────────────

def batch_process(folder_path: pathlib.Path):
    """
    Process all CSV files inside a folder.

    Saves:
        *_stats.png
    """

    csv_files = sorted(
        folder_path.glob("*.csv")
    )

    if not csv_files:

        print("No CSV files found.")

        return

    print(
        f"Found {len(csv_files)} CSV files"
    )

    for csv_path in csv_files:

        try:

            print(
                f"\nProcessing: {csv_path.name}"
            )

            df = load_csv(str(csv_path))

            b_cols = bump_columns(df)

            if not b_cols:

                print(
                    "No bump columns found. Skipping."
                )

                continue

            filename = csv_path.name

            # Extract experiment name
            experiment = csv_path.stem.split("_")[0]

            optimal_threshold = (
                lookup_optimal_threshold(
                    experiment
                )
            )

            # Fallback if threshold unavailable
            if optimal_threshold is None:

                optimal_threshold = float(
                    df["dB_SPL"].min()
                )

            # Apply filtering
            df_filtered = df[
                df["dB_SPL"] >= optimal_threshold
            ]

            # ─────────────────────────────────────────
            # FIGURE LAYOUT
            # ─────────────────────────────────────────

            fig = plt.figure(figsize=(14, 10))

            fig.subplots_adjust(
                left=0.08,
                right=0.97,
                top=0.91,
                bottom=0.10
            )

            outer = gs.GridSpec(
                nrows=3,
                ncols=1,
                figure=fig,
                height_ratios=[3, 3, 3],
                hspace=0.08,
            )

            ax_mean = fig.add_subplot(outer[0])

            ax_sd = fig.add_subplot(outer[1])

            ax_count = fig.add_subplot(outer[2])

            fig.suptitle(
                f"Bump Angle Statistics — "
                f"{filename} "
                f"(n={len(df_filtered)}) "
                f"| threshold ≥ "
                f"{optimal_threshold:.1f}",
                fontsize=10,
                y=0.97,
            )

            # Draw plots
            draw_plots(
                df_filtered,
                b_cols,
                ax_mean,
                ax_sd,
                ax_count,
                fig,
                filename,
            )

            # Save:
            # file.csv -> file_stats.png
            output_path = csv_path.with_name(
                csv_path.stem + "_stats.png"
            )

            plt.savefig(
                output_path,
                dpi=300
            )

            plt.close(fig)

            print(f"Saved: {output_path}")

        except Exception as e:

            print(
                f"Error processing "
                f"{csv_path.name}: {e}"
            )


# ────────────────────────────────────────────────────────────────────────────
# MAIN PROGRAM
# ────────────────────────────────────────────────────────────────────────────

def main():

    if len(sys.argv) < 2:

        print("\nUsage:")
        print("")

        print("Interactive mode:")
        print(
            "  python3 plot_bump_stats.py file.csv"
        )

        print("")

        print("Batch mode:")
        print(
            "  python3 plot_bump_stats.py folder/ --batch"
        )

        print("")

        sys.exit(1)

    input_path = pathlib.Path(sys.argv[1])

    # Enable batch mode if --batch exists
    batch_mode = "--batch" in sys.argv

    # ─────────────────────────────────────────────
    # BATCH MODE
    # ─────────────────────────────────────────────

    if batch_mode:

        if not input_path.is_dir():

            print(
                "Batch mode requires "
                "a folder path."
            )

            sys.exit(1)

        batch_process(input_path)

        return

    # ─────────────────────────────────────────────
    # INTERACTIVE MODE
    # ─────────────────────────────────────────────

    if not input_path.exists():

        print(
            f"Error: file not found "
            f"— {input_path}"
        )

        sys.exit(1)

    # Load CSV
    df = load_csv(str(input_path))

    # Find bump columns
    b_cols = bump_columns(df)

    if not b_cols:

        print(
            "Error: no bump columns found."
        )

        sys.exit(1)

    filename = input_path.name

    db_min = float(df["dB_SPL"].min())

    db_max = float(df["dB_SPL"].max())

    # ─────────────────────────────────────────────
    # FIGURE LAYOUT
    # ─────────────────────────────────────────────

    fig = plt.figure(figsize=(14, 10))

    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.91,
        bottom=0.10
    )

    outer = gs.GridSpec(
        nrows=4,
        ncols=1,
        figure=fig,
        height_ratios=[3, 3, 3, 0.6],
        hspace=0.08,
    )

    ax_mean = fig.add_subplot(outer[0])

    ax_sd = fig.add_subplot(outer[1])

    ax_count = fig.add_subplot(outer[2])

    # Slider axis
    ax_slider = fig.add_axes(
        [0.10, 0.03, 0.78, 0.025]
    )

    # ─────────────────────────────────────────────
    # REDRAW FUNCTION
    # ─────────────────────────────────────────────

    def redraw(threshold: float):

        df_filtered = df[
            df["dB_SPL"] >= threshold
        ]

        fig.suptitle(
            f"Bump Angle Statistics — "
            f"{filename} "
            f"(n={len(df_filtered)}) "
            f"| dB_SPL ≥ "
            f"{threshold:.1f}",
            fontsize=10,
            y=0.97,
        )

        draw_plots(
            df_filtered,
            b_cols,
            ax_mean,
            ax_sd,
            ax_count,
            fig,
            filename,
        )

    # Initial draw
    redraw(db_min)

    # ─────────────────────────────────────────────
    # SLIDER
    # ─────────────────────────────────────────────

    step = (
        (db_max - db_min) / 200
        if db_max != db_min
        else 0.1
    )

    slider = Slider(
        ax=ax_slider,
        label="dB_SPL ≥",
        valmin=db_min,
        valmax=db_max,
        valinit=db_min,
        valstep=step,
        color="#4a90d9",
    )

    slider.label.set_fontsize(9)

    slider.valtext.set_fontsize(9)

    slider.on_changed(
        lambda val: redraw(val)
    )

    # Show GUI
    plt.show()


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()