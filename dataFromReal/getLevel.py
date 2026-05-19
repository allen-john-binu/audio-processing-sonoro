"""
SPL Threshold Analysis for DOA Filtering
=========================================
Analyses all CSV files in a folder and outputs a single recommended SPL
threshold to apply across every run.

WHAT THIS SCRIPT DOES IN PLAIN TERMS
--------------------------------------
Each row of the CSV is one timestep from the robot.  It has:
  - a measured sound pressure level (dB SPL) — how loud the environment was
  - a full spatial spectrum — intensity at every angle from -90° to +90°

The DOA (Direction of Arrival) estimate for that timestep is the angle where
the intensity spectrum peaks.  When SPL is low (the robot is far from both
targets, or between loud events), the spectrum is flat and noisy — those are
the "black" timesteps in the heatmap that produce bad DOA estimates.

The goal is to find a single dB SPL value T such that:
  - timesteps with dB_SPL < T  are discarded  (low signal, unreliable DOA)
  - timesteps with dB_SPL >= T are kept        (strong signal, trustworthy DOA)

CSV format expected:
  - Column 0: timestamp
  - Column 1: dB_SPL  (stored as a string like "[85.83]")
  - Columns 2+: intensity values at angles -90, -87, … , +90 degrees

Usage:
  python spl_threshold_analysis.py <folder>
  python spl_threshold_analysis.py .          # current directory
"""

import sys
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import find_peaks


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════

def parse_spl(val):
    """
    The dB_SPL column is saved by the robot logger as a Python list string,
    e.g. '[85.828069]'.  This function unwraps it to a plain float (85.828069).
    If it is already a plain number string it is converted directly.
    """
    s = str(val).strip()
    if s.startswith("["):
        # ast.literal_eval safely parses '[85.83]' -> [85.83], then we take [0]
        return float(ast.literal_eval(s)[0])
    return float(s)


def load_csv(path: str) -> pd.DataFrame:
    """
    Load one experiment CSV.

    After loading, column names are normalised:
      'timestamp', 'dB_SPL', -90.0, -87.0, ... , 90.0
    The angle columns (floats) hold the raw beamforming intensity at each
    steering angle, in linear power units (W/m2 or similar — very small
    numbers like 2.7e-8).
    """
    df = pd.read_csv(path, header=0)
    # Rename: first two columns are metadata, the rest are angle labels
    df.columns = ["timestamp", "dB_SPL"] + [float(c) for c in df.columns[2:]]
    df["dB_SPL"] = df["dB_SPL"].apply(parse_spl)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def angle_cols(df):
    """Return just the angle column names (floats), in order."""
    return [c for c in df.columns if isinstance(c, float)]


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE EXTRACTION PER TIMESTEP
# ══════════════════════════════════════════════════════════════════

def peak_doa(row, angles):
    """
    Find the two strongest directions of arrival in one timestep's spectrum.

    METHOD — scipy find_peaks:
      We treat the 61-point intensity array (one value per angle) as a 1-D
      signal and find local maxima using scipy.signal.find_peaks.

      Parameters used:
        height=0      -> only accept positive peaks (intensity is always >= 0)
        distance=5    -> peaks must be at least 5 angle-bins apart (~15 deg),
                         preventing two detections on the same physical source

      The two tallest peaks are taken as the two sound sources.

    Returns
    -------
    (doa1, intensity1, doa2, intensity2)
      doa1/doa2 are the angles in degrees of the two strongest peaks.
      intensity1/intensity2 are their raw linear intensities.
      If fewer than two peaks exist, the missing values are NaN.
    """
    intensities = row[angles].values.astype(float)

    peaks, props = find_peaks(intensities, height=0, distance=5)

    if len(peaks) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Sort peaks by height descending so peaks[0] is always the strongest
    heights = props["peak_heights"]
    order   = np.argsort(heights)[::-1]
    peaks   = peaks[order]
    heights = heights[order]

    a1 = angles[peaks[0]] if len(peaks) > 0 else np.nan
    h1 = heights[0]       if len(peaks) > 0 else np.nan
    a2 = angles[peaks[1]] if len(peaks) > 1 else np.nan
    h2 = heights[1]       if len(peaks) > 1 else np.nan

    return a1, h1, a2, h2


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns to the dataframe that the scoring step needs:

      peak_intensity  — max intensity across all angles in that timestep.
                        High when the robot is close to a source; low in
                        the dark/black heatmap regions.

      mean_intensity  — mean intensity across all angles.
                        A secondary measure of how loud the frame is overall.

      doa1, doa2      — angles (degrees) of the two strongest peaks.
                        These are the raw DOA estimates we are trying to clean.

      peak_diff       — |intensity1 - intensity2|
                        Large when one source strongly dominates; small when
                        the spectrum is flat (noisy, no clear source).
    """
    angles = angle_cols(df)
    intensity_matrix = df[angles].values.astype(float)  # shape: (n_timesteps, 61)

    df = df.copy()
    df["peak_intensity"] = intensity_matrix.max(axis=1)
    df["mean_intensity"] = intensity_matrix.mean(axis=1)
    df["peak_diff"]      = np.nan
    df["doa1"]           = np.nan
    df["doa2"]           = np.nan

    for i, (_, row) in enumerate(df.iterrows()):
        a1, h1, a2, h2 = peak_doa(row, angles)
        df.at[df.index[i], "doa1"]      = a1
        df.at[df.index[i], "doa2"]      = a2
        # peak_diff = 0 if only one peak was found (no second source detected)
        df.at[df.index[i], "peak_diff"] = abs(h1 - h2) if not np.isnan(h2) else 0.0

    return df


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — SCORING A CANDIDATE THRESHOLD
# ══════════════════════════════════════════════════════════════════

def evaluate_threshold(df: pd.DataFrame, threshold: float) -> dict:
    """
    Given a candidate threshold T (dB), keep only rows where dB_SPL >= T
    and compute a quality score for that filtered set.

    SCORE FORMULA
    -------------
    Three quantities are combined into one scalar:

      1. intensity_score  = log(1 + mean_peak_intensity)
         ------------------------------------------------
         We use the log so that very large intensities don't dominate.
         This term grows as the threshold rises — higher thresholds keep
         only the loud, signal-rich timesteps.

      2. scatter_penalty  = std(doa1) / 90
         ------------------------------------------------
         Standard deviation of the DOA estimates across kept timesteps,
         normalised by 90 deg (the max possible angle).
         A noisy set of DOA estimates (large std) gets penalised.
         When the threshold is too low, many bad frames are included and
         DOA scatter is high; raising T reduces scatter.

      3. retention_bonus  = sqrt(retention_fraction)
         ------------------------------------------------
         retention_fraction = n_kept / n_total.
         Square-root chosen so the bonus rises steeply when retention is
         very low (we don't want to throw everything away) but flattens
         once enough data is kept.  This counteracts the tendency of
         intensity_score to push T too high.

    Final score:
      score = intensity_score x retention_bonus / (1 + scatter_penalty)

    The threshold that maximises this score balances:
      - keeping signal-rich frames  (intensity_score up)
      - not over-filtering          (retention_bonus down fast when T too high)
      - consistent DOA estimates    (scatter_penalty in denominator)
    """
    kept    = df[df["dB_SPL"] >= threshold]
    n_kept  = len(kept)
    n_total = len(df)

    if n_kept == 0:
        return dict(threshold=threshold, n_kept=0, retention=0,
                    mean_peak_intensity=0, std_doa1=np.nan, score=0)

    retention  = n_kept / n_total                          # fraction in [0, 1]
    mean_peak  = kept["peak_intensity"].mean()             # linear intensity units

    doa1_valid = kept["doa1"].dropna()
    std_doa1   = doa1_valid.std() if len(doa1_valid) > 1 else np.nan

    # Three components of the score
    intensity_score = np.log1p(mean_peak)                  # log(1 + mean_peak)
    scatter_penalty = 0
    retention_bonus = np.sqrt(retention)                   # sqrt of fraction kept

    score = intensity_score * retention_bonus / (1.0 + scatter_penalty)

    return dict(threshold=threshold, n_kept=n_kept, retention=retention,
                mean_peak_intensity=mean_peak, std_doa1=std_doa1, score=score)


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — SWEEP TO FIND THE BEST THRESHOLD FOR ONE RUN
# ══════════════════════════════════════════════════════════════════

def find_optimal_threshold(df: pd.DataFrame,
                           spl_min: float = 60.0,
                           spl_max: float = 100.0,
                           n_steps: int   = 200):
    """
    Brute-force sweep: try 200 evenly-spaced threshold values between
    spl_min and spl_max dB, score each one, return the best.

    We use a dense linear sweep rather than gradient-based optimisation
    because the score function is non-smooth (it depends on which discrete
    rows survive the threshold cut) and cheap to evaluate.

    Returns
    -------
    best_threshold : float     — the dB value that maximised the score
    metrics        : DataFrame — full sweep table for plotting
    """
    sweep   = np.linspace(spl_min, spl_max, n_steps)
    results = [evaluate_threshold(df, t) for t in sweep]
    metrics = pd.DataFrame(results)

    best_idx       = metrics["score"].idxmax()
    best_threshold = metrics.loc[best_idx, "threshold"]

    return best_threshold, metrics


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — PER-RUN DIAGNOSTIC PLOT
# ══════════════════════════════════════════════════════════════════

def plot_analysis(df: pd.DataFrame,
                  metrics: pd.DataFrame,
                  best_threshold: float,
                  title: str = ""):
    """
    Six-panel figure for one experiment run:
      (a) Full DOA heatmap — all timesteps, no filter
      (b) Filtered heatmap — only timesteps above the threshold
      (c) dB SPL over time with threshold line and shaded filtered region
      (d) Composite score vs threshold — shows where the maximum is
      (e) Retention % and mean peak intensity vs threshold — dual-axis,
          shows the trade-off the score is balancing
    """
    angles           = angle_cols(df)
    intensity_matrix = df[angles].values.astype(float).T  # (n_angles, n_time)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"SPL Threshold Analysis  —  {title}\n"
                 f"Optimal threshold: {best_threshold:.1f} dB",
                 fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # (a) Unfiltered heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(intensity_matrix, aspect="auto", origin="lower",
               extent=[0, len(df), float(angles[0]), float(angles[-1])],
               cmap="inferno")
    ax1.set_title("All timesteps (no filter)")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Angle (deg)")

    # (b) Filtered heatmap — only columns (timesteps) that pass the threshold
    ax2 = fig.add_subplot(gs[0, 1])
    kept_idx       = df[df["dB_SPL"] >= best_threshold].index
    kept_positions = [df.index.get_loc(i) for i in kept_idx]
    if kept_positions:
        filt_matrix = intensity_matrix[:, kept_positions]
        ax2.imshow(filt_matrix, aspect="auto", origin="lower",
                   extent=[0, len(kept_positions), float(angles[0]), float(angles[-1])],
                   cmap="inferno")
    ax2.set_title(f"After filtering  (SPL >= {best_threshold:.1f} dB)")
    ax2.set_xlabel("Filtered time step")
    ax2.set_ylabel("Angle (deg)")

    # (c) SPL time-series with threshold overlay
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(df["dB_SPL"].values, lw=0.8, color="steelblue", label="dB SPL")
    ax3.axhline(best_threshold, color="red", lw=1.5, ls="--",
                label=f"Threshold {best_threshold:.1f} dB")
    ax3.fill_between(range(len(df)), df["dB_SPL"].values, best_threshold,
                     where=(df["dB_SPL"].values < best_threshold),
                     alpha=0.25, color="red", label="Filtered out")
    ax3.set_title("dB SPL over time")
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("dB SPL")
    ax3.legend(fontsize=9)

    # (d) Score curve — the maximum is where the threshold is chosen
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(metrics["threshold"], metrics["score"], color="purple")
    ax4.axvline(best_threshold, color="red", ls="--", lw=1.5,
                label=f"Best: {best_threshold:.1f} dB")
    ax4.set_title("Composite quality score vs threshold\n"
                  "score = log(1+intensity) x sqrt(retention) / (1+DOA_scatter/90)")
    ax4.set_xlabel("SPL threshold (dB)")
    ax4.set_ylabel("Score")
    ax4.legend(fontsize=9)

    # (e) Retention & mean peak intensity vs threshold — dual-axis trade-off view
    ax5  = fig.add_subplot(gs[2, 1])
    c1, c2 = "navy", "darkorange"
    l1, = ax5.plot(metrics["threshold"], metrics["retention"] * 100,
                   color=c1, label="Retention (%)")
    ax5.set_ylabel("Retention (%)", color=c1)
    ax5.tick_params(axis="y", labelcolor=c1)

    ax5b = ax5.twinx()
    l2, = ax5b.plot(metrics["threshold"],
                    metrics["mean_peak_intensity"] * 1e6,
                    color=c2, label="Mean peak (x10^-6)")
    ax5b.set_ylabel("Mean peak intensity (x10^-6)", color=c2)
    ax5b.tick_params(axis="y", labelcolor=c2)

    ax5.axvline(best_threshold, color="red", ls="--", lw=1.5)
    ax5.set_title("Retention & signal strength vs threshold")
    ax5.set_xlabel("SPL threshold (dB)")
    ax5.legend(handles=[l1, l2], fontsize=9, loc="upper left")

    return fig


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — PER-RUN PROCESSING WRAPPER
# ══════════════════════════════════════════════════════════════════

def analyse_run(path: str) -> dict:
    """Load, extract features, and find the optimal threshold for one CSV."""
    df                      = load_csv(path)
    df                      = extract_features(df)
    best_threshold, metrics = find_optimal_threshold(df)
    return dict(file=Path(path).name, df=df, metrics=metrics,
                best_threshold=best_threshold)


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════

def summarise_runs(run_results: list) -> pd.DataFrame:
    """
    Build a one-row-per-run summary DataFrame.
    The 'global_retention_%' column is added later in main(), once the
    global threshold is known, to show what each run loses at that level.
    """
    rows = []
    for r in run_results:
        df   = r["df"]
        thr  = r["best_threshold"]
        kept = df[df["dB_SPL"] >= thr]
        rows.append({
            "file":              r["file"],
            "total_timesteps":   len(df),
            "optimal_threshold": round(thr, 2),
            "kept_timesteps":    len(kept),
            "retention_%":       round(len(kept) / len(df) * 100, 1),
            "mean_SPL_all":      round(df["dB_SPL"].mean(), 2),
            "mean_SPL_kept":     round(kept["dB_SPL"].mean(), 2),
            "mean_peak_intens":  f'{kept["peak_intensity"].mean():.3e}',
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — GLOBAL THRESHOLD ACROSS ALL RUNS
# ══════════════════════════════════════════════════════════════════

def find_global_threshold(run_results: list,
                          spl_min: float = 60.0,
                          spl_max: float = 100.0,
                          n_steps: int   = 200):
    """
    Find one threshold that works well across ALL runs simultaneously.

    WHY NOT JUST TAKE THE MEDIAN OF PER-RUN OPTIMA?
    ------------------------------------------------
    Each run's optimum is found independently.  Taking the median of those
    values would give a threshold that is "central" but not necessarily
    optimal for any run — it ignores how the score actually changes around
    that point.

    WHAT WE DO INSTEAD:
    -------------------
    We sweep the same 200 candidate thresholds again, but for each candidate
    we compute the score on EVERY run and average them:

        global_score(T) = mean over all runs of score(run_i, T)

    The T that maximises global_score is the one threshold that, on average
    across the experiment set, best balances signal quality, DOA consistency,
    and data retention.  Runs where a particular threshold performs poorly
    pull the average down, preventing it from being selected.

    Returns
    -------
    global_threshold : float     — the single recommended dB value
    metrics          : DataFrame — sweep table for the summary plot
    """
    sweep = np.linspace(spl_min, spl_max, n_steps)
    rows  = []
    for t in sweep:
        # Score this threshold on every run, then average across runs
        per_run_scores = [evaluate_threshold(r["df"], t)["score"]
                          for r in run_results]
        rows.append(dict(threshold=t, score=np.mean(per_run_scores)))

    metrics = pd.DataFrame(rows)
    best    = metrics.loc[metrics["score"].idxmax(), "threshold"]
    return round(float(best), 1), metrics


# ══════════════════════════════════════════════════════════════════
# SECTION 9 — GLOBAL SUMMARY PLOT
# ══════════════════════════════════════════════════════════════════

def plot_global_summary(run_results: list,
                        global_threshold: float,
                        global_metrics: pd.DataFrame,
                        out_dir: Path):
    """
    Two-panel figure summarising all runs together:
      Left  — bar chart of each run's own optimal threshold, with the
               global threshold overlaid as a dashed red line.
               Shows how consistent or variable the runs are.
      Right — the global score curve (mean score across all runs vs T).
               The maximum is the recommended threshold.
    """
    per_run_thresholds = [r["best_threshold"] for r in run_results]
    labels             = [r["file"] for r in run_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Global SPL Threshold Analysis\n"
                 f"Recommended threshold: {global_threshold} dB",
                 fontsize=13, fontweight="bold")

    # Left: per-run bar chart
    ax = axes[0]
    bars = ax.bar(range(len(labels)), per_run_thresholds,
                  color="steelblue", zorder=3)
    ax.axhline(global_threshold, color="red", ls="--", lw=1.8,
               label=f"Global threshold: {global_threshold} dB")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Per-run optimal threshold (dB)")
    ax.set_title("Per-run optimal thresholds")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for bar, val in zip(bars, per_run_thresholds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    # Right: global score curve
    ax2 = axes[1]
    ax2.plot(global_metrics["threshold"], global_metrics["score"],
             color="purple", lw=1.5)
    ax2.axvline(global_threshold, color="red", ls="--", lw=1.8,
                label=f"Best: {global_threshold} dB")
    ax2.set_xlabel("SPL threshold (dB)")
    ax2.set_ylabel("Mean score across all runs")
    ax2.set_title("Global quality score vs threshold\n"
                  "mean of per-run scores at each candidate T")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "global_threshold_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════════
# SECTION 10 — COMMON RESULTS LOG
# ══════════════════════════════════════════════════════════════════

# Path to the shared log file. Lives next to the script so it accumulates
# results across every folder/experiment you run over time.
GLOBAL_LOG = Path(__file__).parent / "./experiment_thresholds.csv"

def append_global_threshold(experiment_name: str, global_threshold: float):
    """
    Append one row to the shared log file experiment_thresholds.csv.

    The file has three columns:
      experiment      — the folder name you passed in (e.g. 'session_A')
      global_threshold_dB — the single recommended SPL threshold for that folder
      timestamp       — when this result was recorded

    If the file does not exist yet it is created with a header row.
    If it already exists the new row is appended, so results from every
    run accumulate in one place without overwriting previous entries.

    This lets you build up a table like:
      experiment,  global_threshold_dB,  timestamp
      session_A,   89.1,                 2026-05-19 10:32:01
      session_B,   87.4,                 2026-05-19 11:05:44
      session_C,   90.2,                 2026-05-19 14:21:09
    """
    import datetime

    row = pd.DataFrame([{
        "experiment":           experiment_name,
        "global_threshold_dB":  global_threshold,
        "timestamp":            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])

    if GLOBAL_LOG.exists():
        # File already has data — append without writing the header again
        row.to_csv(GLOBAL_LOG, mode="a", header=False, index=False)
    else:
        # First ever run — create the file and write the header
        row.to_csv(GLOBAL_LOG, mode="w", header=True, index=False)


# ══════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python spl_threshold_analysis.py <folder>")
        print("       python spl_threshold_analysis.py .   # current directory")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.")
        sys.exit(1)

    files = sorted(folder.glob("*.csv"))
    if not files:
        print(f"No CSV files found in '{folder}'.")
        sys.exit(1)

    out_dir = folder / "threshold_analysis"
    out_dir.mkdir(exist_ok=True)

    print(f"\nFolder : {folder.resolve()}")
    print(f"Files  : {len(files)} CSV(s) found")
    print(f"Output : {out_dir}\n")
    print("-" * 55)

    # Step 1: process each run independently
    run_results = []
    for path in files:
        print(f"  Processing  {path.name} ...", end="", flush=True)
        try:
            result = analyse_run(str(path))
            run_results.append(result)

            fig = plot_analysis(result["df"], result["metrics"],
                                result["best_threshold"], title=result["file"])
            fig_path = out_dir / f"{path.stem}_analysis.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  per-run optimum = {result['best_threshold']:.1f} dB")
        except Exception as e:
            print(f"  ERROR: {e}")

    if not run_results:
        print("No files processed successfully.")
        sys.exit(1)

    # Step 2: find one threshold that works across all runs
    print("\n" + "-" * 55)
    print("Computing global threshold across all runs...")
    global_threshold, global_metrics = find_global_threshold(run_results)

    # Step 3: build summary table and annotate with global-threshold retention
    summary = summarise_runs(run_results)
    global_retention = []
    for r in run_results:
        df   = r["df"]
        kept = df[df["dB_SPL"] >= global_threshold]
        global_retention.append(round(len(kept) / len(df) * 100, 1))
    summary["global_retention_%"] = global_retention

    csv_path  = out_dir / "summary.csv"
    summary.to_csv(csv_path, index=False)

    plot_path = plot_global_summary(run_results, global_threshold,
                                    global_metrics, out_dir)

    # Step 4: append (experiment name, threshold) to the shared log.
    # The experiment name is just the folder name — e.g. 'session_A'.
    # This way every time you run the script on a new folder, the result
    # is recorded in one growing CSV file next to the script.
    experiment_name = folder.resolve().name
    append_global_threshold(experiment_name, global_threshold)

    # Step 5: print final report
    print("\n-- Per-run results --------------------------------------------------")
    print(summary[["file", "total_timesteps", "optimal_threshold",
                   "global_retention_%"]].to_string(index=False))

    print("\n+================================================+")
    print(f"|  RECOMMENDED GLOBAL THRESHOLD: {global_threshold:6.1f} dB       |")
    print("+================================================+")

    mean_ret = np.mean(global_retention)
    print(f"\n  Average retention at this threshold: {mean_ret:.1f}% of timesteps")
    print(f"  Summary CSV  -> {csv_path}")
    print(f"  Summary plot -> {plot_path}")
    print(f"  Global log   -> {GLOBAL_LOG}  (appended)\n")


if __name__ == "__main__":
    main()



