#!/usr/bin/env python3

"""
plot_bumps.py
──────────────────────────────────────────────────────────────────────────────

Interactive + Batch visualization tool for bump-angle CSV datasets.

FEATURES
────────
1. Interactive mode
   - Opens a matplotlib window
   - Includes a dB_SPL slider
   - Lets you explore filtering thresholds dynamically

2. Batch mode
   - Processes ALL CSV files inside a folder
   - Automatically applies the optimal threshold
   - Saves PNG images
   - Does NOT open any GUI window

USAGE
─────

Interactive mode (single CSV + slider UI):
    python3 plot_bumps.py ./oldResult/expA/file.csv

Batch mode (folder processing + PNG export):
    python3 plot_bumps.py ./oldResult/expA/ --batch


INPUT CSV REQUIREMENTS
──────────────────────
The CSV files must contain:

    timestamp
    bump1 ... bumpN
    angle
    sample_index
    time_seconds
    left_volume
    right_volume
    dB_SPL

OPTIONAL THRESHOLD LOOKUP
─────────────────────────
The script can automatically load experiment-specific thresholds from:

    ../dataFromReal/experiment_thresholds.csv

Expected columns:
    experiment
    global_threshold_dB
    timestamp

Example:
    experiment,global_threshold_dB,timestamp
    expA,41.5,2026-01-01

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
    Look up the optimal dB threshold for an experiment.

    Parameters
    ----------
    experiment : str
        Example: "expA"

    Returns
    -------
    float | None
        The threshold value if found.
        Otherwise returns None.
    """

    try:
        tdf = pd.read_csv(THRESHOLDS_CSV)

        match = tdf[tdf["experiment"] == experiment]

        if match.empty:
            print(f"[thresholds] No threshold found for '{experiment}'")
            return None

        value = float(match.iloc[0]["global_threshold_dB"])

        print(f"[thresholds] {experiment} -> {value} dB_SPL")

        return value

    except FileNotFoundError:
        print(f"[thresholds] File not found: {THRESHOLDS_CSV}")
        return None

    except Exception as e:
        print(f"[thresholds] Error reading thresholds file: {e}")
        return None


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV and validate required columns.
    """

    df = pd.read_csv(path)

    required = {
        "dB_SPL",
        "angle",
        "left_volume",
        "right_volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def bump_columns(df: pd.DataFrame) -> list[str]:
    """
    Find columns named:
        bump1, bump2, bump3, ...

    Returns them in numerical order.
    """

    cols = [
        c for c in df.columns
        if c.startswith("bump") and c[4:].isdigit()
    ]

    cols.sort(key=lambda c: int(c[4:]))

    return cols


def build_heatmap(
    df: pd.DataFrame,
    b_cols: list[str]
) -> np.ndarray:
    """
    Convert bump-angle columns into a heatmap matrix.

    Returns
    -------
    np.ndarray
        Shape:
            (n_timesteps, n_angle_bins)
    """

    # Angle bins:
    # -180 → +180 in steps of 10°
    bin_edges = np.arange(-180, 181, 10)

    n_bins = len(bin_edges) - 1
    n_timesteps = len(df)

    heatmap = np.zeros((n_timesteps, n_bins))

    for i, (_, row) in enumerate(df[b_cols].iterrows()):

        counts, _ = np.histogram(
            row.dropna().values,
            bins=bin_edges
        )

        heatmap[i] = counts

    return heatmap


# ────────────────────────────────────────────────────────────────────────────
# PLOT DRAWING FUNCTION
# ────────────────────────────────────────────────────────────────────────────

def draw_plots(
    df_filtered: pd.DataFrame,
    b_cols: list[str],
    ax_vol,
    ax_heat,
    ax_angle,
    ax_cb_heat,
    fig,
    filename: str,
    cb_heat_ref: list,
    optimal_threshold: float | None = None,
):
    """
    Draw all visualization panels.

    PANELS
    ------
    1. Volume plot
    2. Bump heatmap
    3. Angle plot
    """

    # Clear old plots
    for ax in (ax_vol, ax_heat, ax_angle, ax_cb_heat):
        ax.cla()

    n_timesteps = len(df_filtered)

    xs = np.arange(n_timesteps)

    # Empty dataset case
    if n_timesteps == 0:

        ax_vol.set_title(
            "No data matches current threshold"
        )

        fig.canvas.draw_idle()

        return

    # ─────────────────────────────────────────────────────────────
    # PANEL 1 — LEFT/RIGHT VOLUME
    # ─────────────────────────────────────────────────────────────

    ax_vol.plot(
        xs,
        df_filtered["left_volume"].values + 0.45,
        linewidth=0.9,
        color="#e53935",
        label="left_volume (+0.45 bias)"
    )

    ax_vol.plot(
        xs,
        df_filtered["right_volume"].values,
        linewidth=0.9,
        color="#43a047",
        label="right_volume"
    )

    ax_vol.set_ylabel("Volume")

    ax_vol.set_xlim(0, n_timesteps)
    ax_vol.set_ylim(0.5, 1.5)

    ax_vol.set_xticklabels([])

    ax_vol.legend(
        fontsize=7,
        loc="upper right",
        framealpha=0.6
    )

    # ─────────────────────────────────────────────────────────────
    # PANEL 2 — HEATMAP
    # ─────────────────────────────────────────────────────────────

    heatmap = build_heatmap(df_filtered, b_cols)

    im_heat = ax_heat.imshow(
        heatmap.T,
        aspect="auto",
        origin="lower",
        extent=[0, n_timesteps, -180, 180],
    )

    fig.colorbar(
        im_heat,
        cax=ax_cb_heat,
        label="Count"
    )

    opt_str = (
        f"{optimal_threshold:.1f} dB_SPL"
        if optimal_threshold is not None
        else "N/A"
    )

    fig.suptitle(
        f"{filename} | n={n_timesteps} | optimal={opt_str}",
        fontsize=10,
        y=0.97,
    )

    ax_heat.set_ylabel("Angle (degrees)")
    ax_heat.set_xlim(0, n_timesteps)

    ax_heat.set_xticklabels([])

    # ─────────────────────────────────────────────────────────────
    # PANEL 3 — ANGLE LINE PLOT
    # ─────────────────────────────────────────────────────────────

    angles = df_filtered["angle"].values

    ax_angle.plot(
        xs,
        angles,
        linewidth=0.9,
        color="#2196f3"
    )

    ax_angle.set_ylabel("Angle (°)")
    ax_angle.set_xlabel("Timestamp index")

    ax_angle.set_xlim(0, n_timesteps)

    # Highlight first angle > 90°
    over90 = np.where(angles > 90)[0]

    if len(over90) > 0:

        first_idx = int(over90[0])

        ax_angle.axvline(
            x=first_idx,
            color="red",
            linestyle=":",
            linewidth=1.5,
            label=f"first angle > 90° (i={first_idx})",
        )

        ax_angle.legend(
            fontsize=7,
            loc="upper right",
            framealpha=0.6
        )

    fig.canvas.draw_idle()


# ────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING MODE
# ────────────────────────────────────────────────────────────────────────────

def batch_process(folder_path: pathlib.Path):
    """
    Process all CSV files inside a folder.

    For each CSV:
        - apply optimal threshold
        - generate figure
        - save PNG image

    NO GUI WINDOW is shown.
    """

    csv_files = sorted(folder_path.glob("*.csv"))

    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} CSV files")

    for csv_path in csv_files:

        try:

            print(f"\nProcessing: {csv_path.name}")

            df = load_csv(str(csv_path))

            b_cols = bump_columns(df)

            if not b_cols:
                print("No bump columns found. Skipping.")
                continue

            filename = csv_path.name

            # Extract experiment name from filename
            # Example:
            #   expA_exA1_bumps.csv
            # -> expA
            experiment = csv_path.stem.split("_")[0]

            optimal_threshold = lookup_optimal_threshold(
                experiment
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

            # ─────────────────────────────────────────────
            # FIGURE LAYOUT
            # ─────────────────────────────────────────────

            fig = plt.figure(figsize=(14, 10))

            fig.subplots_adjust(
                left=0.07,
                right=0.93,
                top=0.88,
                bottom=0.10
            )

            outer = gs.GridSpec(
                nrows=3,
                ncols=2,
                figure=fig,
                width_ratios=[20, 1],
                height_ratios=[2, 5, 2],
                hspace=0.08,
                wspace=0.03,
            )

            ax_cb_heat = fig.add_subplot(outer[1, 1])

            # Hide unused cells
            for row in (0, 2):
                fig.add_subplot(
                    outer[row, 1]
                ).set_visible(False)

            ax_vol = fig.add_subplot(outer[0, 0])

            ax_heat = fig.add_subplot(outer[1, 0])

            ax_angle = fig.add_subplot(outer[2, 0])

            cb_heat_ref = [None]

            # Draw plots
            draw_plots(
                df_filtered,
                b_cols,
                ax_vol,
                ax_heat,
                ax_angle,
                ax_cb_heat,
                fig,
                filename,
                cb_heat_ref,
                optimal_threshold=optimal_threshold,
            )

            # Output image path
            output_path = csv_path.with_suffix(".png")

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

    # Need at least:
    #   python3 plot_bumps.py something
    if len(sys.argv) < 2:

        print("\nUsage:")
        print("")

        print("Interactive mode:")
        print(
            "  python3 plot_bumps.py file.csv"
        )

        print("")

        print("Batch mode:")
        print(
            "  python3 plot_bumps.py folder/ --batch"
        )

        print("")

        sys.exit(1)

    input_path = pathlib.Path(sys.argv[1])

    # Enable batch mode if --batch exists
    batch_mode = "--batch" in sys.argv

    # ─────────────────────────────────────────────────────────────
    # BATCH MODE
    # ─────────────────────────────────────────────────────────────

    if batch_mode:

        if not input_path.is_dir():

            print(
                "Batch mode requires "
                "a folder path."
            )

            sys.exit(1)

        batch_process(input_path)

        return

    # ─────────────────────────────────────────────────────────────
    # INTERACTIVE MODE
    # ─────────────────────────────────────────────────────────────

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

    experiment = input_path.stem.split("_")[0]

    db_min = float(df["dB_SPL"].min())

    db_max = float(df["dB_SPL"].max())

    optimal_threshold = lookup_optimal_threshold(
        experiment
    )

    # ─────────────────────────────────────────────
    # FIGURE LAYOUT
    # ─────────────────────────────────────────────

    fig = plt.figure(figsize=(14, 10))

    fig.subplots_adjust(
        left=0.07,
        right=0.93,
        top=0.88,
        bottom=0.10
    )

    outer = gs.GridSpec(
        nrows=4,
        ncols=2,
        figure=fig,
        width_ratios=[20, 1],
        height_ratios=[2, 5, 2, 0.6],
        hspace=0.08,
        wspace=0.03,
    )

    ax_cb_heat = fig.add_subplot(outer[1, 1])

    # Hide unused cells
    for row in (0, 2, 3):
        fig.add_subplot(
            outer[row, 1]
        ).set_visible(False)

    ax_vol = fig.add_subplot(outer[0, 0])

    ax_heat = fig.add_subplot(outer[1, 0])

    ax_angle = fig.add_subplot(outer[2, 0])

    # Slider axis
    ax_slider = fig.add_axes(
        [0.10, 0.03, 0.78, 0.025]
    )

    cb_heat_ref = [None]

    # ─────────────────────────────────────────────
    # REDRAW FUNCTION
    # ─────────────────────────────────────────────

    def redraw(threshold: float):

        df_filtered = df[
            df["dB_SPL"] >= threshold
        ]

        draw_plots(
            df_filtered,
            b_cols,
            ax_vol,
            ax_heat,
            ax_angle,
            ax_cb_heat,
            fig,
            filename,
            cb_heat_ref,
            optimal_threshold=optimal_threshold,
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

    # Show interactive window
    plt.show()


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
    
    
    


# python3 plotBumpStats.py ./oldResult/expBB      --batch;
# python3 plotBumpStats.py ./oldResult/expBD      --batch;
# python3 plotBumpStats.py ./oldResult/expCB      --batch;
# python3 plotBumpStats.py ./oldResult/expCD      --batch;
# python3 plotBumpStats.py ./oldResult/expDB      --batch;
# python3 plotBumpStats.py ./oldResult/expDD --batch;
# python3 plotBumpStats.py ./oldResult/expBA      --batch;
# python3 plotBumpStats.py ./oldResult/expBC      --batch;
# python3 plotBumpStats.py ./oldResult/expCA      --batch;
# python3 plotBumpStats.py ./oldResult/expCC      --batch;
# python3 plotBumpStats.py ./oldResult/expDA      --batch;
# python3 plotBumpStats.py ./oldResult/expDC --batch;
