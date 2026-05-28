import numpy as np
from random import seed as py_seed
import csv
import math
import utils
import os
import sys
import argparse
import matplotlib
matplotlib.use("TkAgg")  # interactive backend for slider
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 100
np.random.seed(SEED)
py_seed(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
NORMALIZING_FACTOR = 0.1546
DB_SPL_THRESHOLD   = 50
N_BUMP_RUNS        = 50


# ── Ring Attractor ─────────────────────────────────────────────────────────────
class RA():
    def __init__(self):
        self.Ns               = 120
        self.v                = 0.5
        self.h_0              = 0.051
        self.h_b              = 0.0
        self.v_0              = 60
        self.h_ext            = None
        self.beta             = 4000
        self.sigma_ang        = 2 * np.pi / self.Ns
        self.thetas           = np.linspace(-np.pi, np.pi, self.Ns, endpoint=False)
        self.spins            = np.random.choice([1, 0], size=self.Ns)
        self.pos              = np.zeros(2)
        self.allocentric      = False
        self.heading          = 0
        self.updates_per_step = int(round(self.Ns * 4))


# ── Load and Normalize CSV ─────────────────────────────────────────────────────
def load_and_normalize(input_path):
    """
    Read input CSV, filter by dB threshold, normalize DOA values.

    Returns:
        normalized_array : np.ndarray, shape (n_timesteps, 120)
        timestamps       : list of timestamp strings
        vicon_angles     : np.ndarray, shape (n_timesteps,)
    """
    all_values        = []
    rows_data         = []
    timestamps        = []
    vicon_angles_list = []

    with open(input_path, newline='') as f:
        reader    = csv.reader(f)
        header    = next(reader)

        angle_idx   = header.index("angle")
        doa_headers = header[2:angle_idx]
        csv_angles  = [int(float(a)) for a in doa_headers]

        for row in reader:
            db_spl = float(row[1].strip("[]"))
            if db_spl > DB_SPL_THRESHOLD:
                values     = [float(x) for x in row[2:angle_idx]]
                full_array = [0.0] * 120

                for angle, value in zip(csv_angles, values):
                    idx = (angle + 180) // 3
                    if 0 <= idx < 120:
                        full_array[idx] = value
                        all_values.append(value)

                timestamps.append(row[0])
                rows_data.append(full_array)
                vicon_angles_list.append(float(row[angle_idx]))

    if not rows_data:
        print("  [SKIP] No rows above dB threshold.")
        return None

    global_min = min(all_values)
    global_max = max(all_values)
    print(f"  Timesteps found : {len(rows_data)}")

    # Normalize and store as (n_timesteps, 120) numpy array
    normalized_list = []
    for arr in rows_data:
        norm_arr = []
        for v in arr:
            if v == 0.0:
                norm_arr.append(0.0)
            else:
                norm = (v - global_min) / (global_max - global_min)
                norm_arr.append(norm * NORMALIZING_FACTOR)
        normalized_list.append(norm_arr)

    normalized_array = np.array(normalized_list)   # (n_timesteps, 120)

    return normalized_array, timestamps, np.array(vicon_angles_list)


# ── Run Ring Attractor and Collect All Spins ───────────────────────────────────
def collect_spins(normalized_array):
    """
    Pre-compute and store all spins for every timestep and every bump run.

    Args:
        normalized_array : np.ndarray, shape (n_timesteps, 120)

    Returns:
        spins_data : np.ndarray, shape (n_timesteps, N_BUMP_RUNS, 120)
        thetas_deg : np.ndarray, shape (120,) — neuron angles in degrees
    """
    n_timesteps = normalized_array.shape[0]
    spins_data  = np.zeros((n_timesteps, N_BUMP_RUNS, 120), dtype=np.int8)

    ref_ring   = RA()
    thetas_deg = np.degrees(ref_ring.thetas)   # -180 to +180

    print(f"\n  Running {n_timesteps} timesteps x {N_BUMP_RUNS} runs...")
    print(f"  (all spins stored in memory)\n")

    for t in range(n_timesteps):
        if (t + 1) % max(1, n_timesteps // 10) == 0 or t == 0:
            print(f"    Timestep {t + 1:>4} / {n_timesteps}")

        h_ext = normalized_array[t].tolist()

        for run in range(N_BUMP_RUNS):
            ring       = RA()
            ring.h_ext = h_ext

            for _ in range(ring.updates_per_step):
                i       = np.random.randint(0, ring.Ns)
                delta_H = utils.compute_delta_H(ring, i)
                if delta_H < 0:
                    ring.spins[i] = 1 - ring.spins[i]
                else:
                    p = np.exp(-ring.beta * delta_H)
                    if np.random.rand() < p:
                        ring.spins[i] = 1 - ring.spins[i]

            spins_data[t, run, :] = ring.spins

    print(f"\n  Done. spins_data shape: {spins_data.shape}  (timesteps x runs x neurons)")
    return spins_data, thetas_deg


# ── Interactive Visualizer ─────────────────────────────────────────────────────
def launch_visualizer(spins_data, normalized_array, thetas_deg,
                      timestamps, vicon_angles, filename):
    """
    Launch interactive matplotlib figure.

    Layout:
        Top    (~70%) : spins heatmap
                         X — run index (0 to 49)
                         Y — neuron angle in degrees (-180 at bottom, +180 at top)
                         color — spin value (0=black, 1=white)
        Bottom (~30%) : DOA estimate line plot
                         X — neuron angle in degrees (-180 to +180)
                         Y — normalized DOA value
        Slider        : selects timestep; only im.set_data(), line.set_ydata(),
                         and title are updated — nothing else is recreated
    """
    n_timesteps = spins_data.shape[0]

    # ── Figure and GridSpec ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 12))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.18)

    gs     = GridSpec(2, 1, figure=fig, height_ratios=[7, 3], hspace=0.35)
    ax_top = fig.add_subplot(gs[0])   # spins heatmap
    ax_bot = fig.add_subplot(gs[1])   # DOA line plot

    # ── Top: Spins Heatmap — created ONCE ─────────────────────────────────────
    im = ax_top.imshow(
        spins_data[0].T,              # (120, 50): neurons on Y, runs on X
        aspect        = 'auto',
        cmap          = 'gray',
        origin        = 'lower',      # -180 at bottom, +180 at top
        vmin          = 0,
        vmax          = 1,
        extent        = [-0.5, N_BUMP_RUNS - 0.5, thetas_deg[0], thetas_deg[-1]],
        interpolation = 'nearest'
    )

    cbar = fig.colorbar(im, ax=ax_top, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0 (black)', '1 (white)'])
    cbar.set_label("Spin value", fontsize=11)

    ax_top.set_xlabel("Run index (0 – 49)", fontsize=12)
    ax_top.set_ylabel("Neuron angle (degrees)", fontsize=12)
    ax_top.set_yticks(np.arange(-180, 181, 30))
    ax_top.grid(True, alpha=0.2, linewidth=0.5)

    title = ax_top.set_title("", fontsize=13, fontweight='bold')

    # ── Bottom: DOA Line Plot — created ONCE ──────────────────────────────────
    line, = ax_bot.plot(thetas_deg, normalized_array[0],
                        color='steelblue', linewidth=1.5)

    # Fix Y-limits to global max across all timesteps so scale never jumps
    doa_global_max = normalized_array.max()
    ax_bot.set_xlim(thetas_deg[0], thetas_deg[-1])
    ax_bot.set_ylim(0, doa_global_max * 1.1)
    ax_bot.set_xlabel("Neuron angle (degrees)", fontsize=12)
    ax_bot.set_ylabel("Normalized DOA value", fontsize=12)
    ax_bot.set_title("DOA Estimate Input to Ring Attractor", fontsize=12, fontweight='bold')
    ax_bot.set_xticks(np.arange(-180, 181, 30))
    ax_bot.grid(True, alpha=0.3)

    # ── Helper: update title ───────────────────────────────────────────────────
    def _set_title(t):
        title.set_text(
            f"{filename}  |  Timestep {t} / {n_timesteps - 1}"
            f"    Timestamp: {timestamps[t]}  |  Vicon angle: {vicon_angles[t]:.2f}°"
        )

    _set_title(0)

    # ── Slider ─────────────────────────────────────────────────────────────────
    ax_slider = plt.axes([0.15, 0.06, 0.70, 0.04])
    slider    = Slider(
        ax_slider, 'Timestep',
        0, n_timesteps - 1,
        valinit  = 0,
        valstep  = 1,
        color    = 'steelblue'
    )

    # ── Update — only these 3 things change per slider move ───────────────────
    def on_slider_change(val):
        t = int(slider.val)
        im.set_data(spins_data[t].T)
        line.set_ydata(normalized_array[t])
        _set_title(t)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)

    print("=" * 60)
    print("  Interactive visualizer ready — drag the slider!")
    print("=" * 60)
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Debug: visualize ring attractor spins with a timestep slider."
    )
    parser.add_argument(
        "csv_path",
        help="Path to a single input CSV file (from processData/)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"ERROR: File not found: {args.csv_path}")
        sys.exit(1)

    filename = os.path.basename(args.csv_path)
    print(f"\n{'=' * 60}")
    print(f"  Debug mode: {filename}")
    print(f"{'=' * 60}")

    result = load_and_normalize(args.csv_path)
    if result is None:
        sys.exit(1)

    normalized_array, timestamps, vicon_angles = result
    spins_data, thetas_deg = collect_spins(normalized_array)

    launch_visualizer(
        spins_data, normalized_array, thetas_deg,
        timestamps, vicon_angles, filename
    )


if __name__ == "__main__":
    main()