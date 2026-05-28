"""
plot_combined.py — Four-line normalised mean plot combining two datasets.

Usage:
    python3 plot_combined.py <bumps_csv> <intensity_csv>

    bumps_csv     : CSV with bump1..bumpN, dB_SPL, angle, etc.  (2nd dataset)
    intensity_csv : CSV with timestamp, dB_SPL, -90.0..-90.0 angle-bin
                    intensity columns                            (1st dataset)

Both CSVs must have the same number of rows (matched by row index).
Filtering is applied via the dB_SPL column of the bumps CSV.

Four lines plotted (each independently min-max normalised to [0, 1]):
    DS1 neg  — mean intensity of angle bins -90.0 … 0.0  (inclusive)
    DS1 pos  — mean intensity of angle bins  0.0 … 90.0  (inclusive)
    DS2 neg  — abs mean of bump values < 0
    DS2 pos  — abs mean of bump values > 0

Colors:
    DS1 neg : crimson  #dc2626
    DS1 pos : amber    #f59e0b
    DS2 neg : teal     #0d9488
    DS2 pos : indigo   #4338ca
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.widgets import Slider

# ── COLORS ─────────────────────────────────────────────────────────────────
C_DS1_NEG = "#dc2626"   # crimson — DS1 negative side
C_DS1_POS = "#f59e0b"   # amber   — DS1 positive side
C_DS2_NEG = "#0d9488"   # teal    — DS2 negative bumps
C_DS2_POS = "#4338ca"   # indigo  — DS2 positive bumps
LW        = 1.0


# ── HELPERS ────────────────────────────────────────────────────────────────

def load_bumps(path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    missing = {"dB_SPL"} - set(df.columns)
    if missing:
        raise ValueError(f"Bumps CSV missing columns: {missing}")
    b_cols = [c for c in df.columns if c.startswith("bump") and c[4:].isdigit()]
    b_cols.sort(key=lambda c: int(c[4:]))
    if not b_cols:
        raise ValueError("Bumps CSV has no bump1, bump2, … columns.")
    return df, b_cols


def load_intensity(path: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Returns (df, neg_cols, pos_cols) where neg_cols are bins -90..0
    and pos_cols are bins 0..90 (0.0 included in both).
    """
    df = pd.read_csv(path)
    # Identify angle-bin columns: numeric column names
    bin_cols = []
    for c in df.columns:
        try:
            float(c)
            bin_cols.append(c)
        except ValueError:
            pass
    if not bin_cols:
        raise ValueError("Intensity CSV has no numeric angle-bin columns.")
    bin_vals = [float(c) for c in bin_cols]
    neg_cols = [c for c, v in zip(bin_cols, bin_vals) if v <= 0.0]
    pos_cols = [c for c, v in zip(bin_cols, bin_vals) if v >= 0.0]
    return df, neg_cols, pos_cols


def minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max scale arr to [0, 1]. Returns zeros if range is zero."""
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def compute_series(df_bumps: pd.DataFrame, b_cols: list[str],
                   df_int: pd.DataFrame,
                   neg_cols: list[str], pos_cols: list[str]
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the four raw (un-normalised) mean series then normalise each
    independently with min-max scaling.
    Returns (ds1_neg, ds1_pos, ds2_neg, ds2_pos) all in [0, 1].
    """
    n = len(df_bumps)

    # ── DS1: per-row (intensity × angle), normalise, abs mean ────────────
    neg_angles  = np.array([float(c) for c in neg_cols])
    pos_angles  = np.array([float(c) for c in pos_cols])
    n_rows      = len(df_int)
    ds1_neg_raw = np.zeros(n_rows)
    ds1_pos_raw = np.zeros(n_rows)

    for i in range(n_rows):
        # negative side
        neg_prod = df_int[neg_cols].values[i].astype(float) * neg_angles
        lo, hi   = neg_prod.min(), neg_prod.max()
        neg_norm = (neg_prod - lo) / (hi - lo) if hi != lo else np.zeros_like(neg_prod)
        ds1_neg_raw[i] = np.abs(np.mean(neg_norm))

        # positive side
        pos_prod = df_int[pos_cols].values[i].astype(float) * pos_angles
        lo, hi   = pos_prod.min(), pos_prod.max()
        pos_norm = (pos_prod - lo) / (hi - lo) if hi != lo else np.zeros_like(pos_prod)
        ds1_pos_raw[i] = np.abs(np.mean(pos_norm))

    # ── DS2: abs mean of pos/neg bumps per row ─────────────────────────────
    bump_vals   = df_bumps[b_cols].values.astype(float)
    ds2_neg_raw = np.full(n, np.nan)
    ds2_pos_raw = np.full(n, np.nan)

    for i in range(n):
        row = bump_vals[i]
        row = row[~np.isnan(row)]
        neg = row[row < 0]
        pos = row[row > 0]
        if len(neg) > 0:
            ds2_neg_raw[i] = np.abs(np.mean(neg))
        if len(pos) > 0:
            ds2_pos_raw[i] = np.abs(np.mean(pos))

    return (minmax(ds1_neg_raw), minmax(ds1_pos_raw),
            minmax(ds2_neg_raw), minmax(ds2_pos_raw))


# ── PLOT BUILDER ───────────────────────────────────────────────────────────

def draw_plot(df_bumps: pd.DataFrame, b_cols: list[str],
              df_int: pd.DataFrame, neg_cols: list[str], pos_cols: list[str],
              ax, fig, filename_bumps: str) -> None:
    ax.cla()

    n = len(df_bumps)
    if n == 0:
        ax.set_ylabel("Normalised mean")
        ax.set_xlabel("Timestamp index")
        fig.canvas.draw_idle()
        return

    xs = np.arange(n)
    ds1_neg, ds1_pos, ds2_neg, ds2_pos = compute_series(
        df_bumps, b_cols, df_int, neg_cols, pos_cols
    )

    ax.plot(xs, ds1_neg, color=C_DS1_NEG, linewidth=LW,
            label="DS1 neg  (intensity -90→0)")
    ax.plot(xs, ds1_pos, color=C_DS1_POS, linewidth=LW,
            label="DS1 pos  (intensity 0→+90)")
    ax.plot(xs, ds2_neg, color=C_DS2_NEG, linewidth=LW,
            label="DS2 neg  (bump |mean| < 0)")
    ax.plot(xs, ds2_pos, color=C_DS2_POS, linewidth=LW,
            label="DS2 pos  (bump |mean| > 0)")

    ax.set_ylabel("Normalised mean [0, 1]")
    ax.set_xlabel("Timestamp index")
    ax.set_xlim(0, n)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

    fig.canvas.draw_idle()


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 plot_combined.py <bumps_csv> <intensity_csv>")
        sys.exit(1)

    bumps_path = pathlib.Path(sys.argv[1])
    int_path   = pathlib.Path(sys.argv[2])

    for p in (bumps_path, int_path):
        if not p.exists():
            print(f"Error: file not found — {p}")
            sys.exit(1)

    df_bumps, b_cols = load_bumps(str(bumps_path))
    df_int, neg_cols, pos_cols = load_intensity(str(int_path))

    if len(df_bumps) != len(df_int):
        raise ValueError(
            f"Row count mismatch: bumps has {len(df_bumps)} rows, "
            f"intensity has {len(df_int)} rows. Both must match."
        )

    filename = bumps_path.name
    db_min   = float(df_bumps["dB_SPL"].min())
    db_max   = float(df_bumps["dB_SPL"].max())

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.15)

    outer    = gs.GridSpec(nrows=2, ncols=1, figure=fig,
                           height_ratios=[6, 0.5], hspace=0.08)
    ax       = fig.add_subplot(outer[0])
    ax_slider = fig.add_axes([0.10, 0.04, 0.78, 0.025])

    def redraw(threshold: float) -> None:
        mask        = df_bumps["dB_SPL"].apply(
                          lambda v: float(str(v).strip("[]")) >= threshold)
        df_b_f      = df_bumps[mask].reset_index(drop=True)
        df_i_f      = df_int[mask].reset_index(drop=True)
        fig.suptitle(
            f"Normalised Mean Comparison — {filename}  "
            f"(n={len(df_b_f)})  |  dB_SPL ≥ {threshold:.1f}",
            fontsize=10, y=0.97,
        )
        draw_plot(df_b_f, b_cols, df_i_f, neg_cols, pos_cols,
                  ax, fig, filename)

    redraw(db_min)

    # ── Slider ─────────────────────────────────────────────────────────────
    step   = (db_max - db_min) / 200 if db_max != db_min else 0.1
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
    slider.on_changed(lambda val: redraw(val))

    plt.show()


if __name__ == "__main__":
    main()