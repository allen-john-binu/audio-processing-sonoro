import numpy as np
from random import seed as py_seed
import csv
import math
import utils
import os
import sys
import argparse
import glob
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — save only, no display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 100
np.random.seed(SEED)
py_seed(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
NORMALIZING_FACTOR = 0.1546
DB_SPL_THRESHOLD   = 50 # removing the threshold filter 
N_BUMP_RUNS        = 50
PROCESS_DATA_DIR   = "./processData"
BUMP_RESULTS_DIR   = "./bumpResults"


# ── Ring Attractor ─────────────────────────────────────────────────────────────
class RA():
    def __init__(self):
        self.Ns              = 120
        self.v               = 0.5
        self.h_0             = 0.051
        self.h_b             = 0.0122
        self.v_0             = 60
        self.h_ext           = None
        self.beta            = 400
        self.sigma_ang       = 2 * np.pi / self.Ns
        self.thetas          = np.linspace(-np.pi, np.pi, self.Ns, endpoint=False)
        self.spins           = np.random.choice([1, 0], size=self.Ns)
        self.pos             = np.zeros(2)
        self.allocentric     = False
        self.heading         = 0
        self.updates_per_step = int(round(self.Ns * 4))


# ── Per-file pipeline ──────────────────────────────────────────────────────────
def process_file(input_path, output_csv_path, output_plot_path):
    filename = os.path.basename(input_path)
    print(f"\n{'=' * 60}")
    print(f"Processing: {filename}")
    print(f"{'=' * 60}")

    all_values = []
    rows_data  = []
    extra_cols = []   # [angle, sample_index, time_seconds, left_volume, right_volume]

    # ── FIRST PASS: filter + build ring arrays ─────────────────────────────────
    with open(input_path, newline='') as f:
        reader    = csv.reader(f)
        header    = next(reader)
        csv_angles = [int(float(a)) for a in header[2:-5]]

        # Indices for extra columns
        angle_idx        = header.index("angle")
        sample_idx       = header.index("sample_index")
        time_idx         = header.index("time_seconds")
        left_vol_idx     = header.index("left_volume")
        right_vol_idx    = header.index("right_volume")

        for row in reader:
            db_spl = float(row[1].strip("[]"))
            if db_spl > DB_SPL_THRESHOLD:
                timestamp  = row[0]
                values     = [float(x) for x in row[2:-5]]
                full_array = [0.0] * 120

                for angle, value in zip(csv_angles, values):
                    idx = (angle + 180) // 3
                    if 0 <= idx < 120:
                        full_array[idx] = value
                        all_values.append(value)

                rows_data.append((timestamp, full_array))
                extra_cols.append([
                    float(row[angle_idx]),
                    row[sample_idx],
                    row[time_idx],
                    row[left_vol_idx],
                    row[right_vol_idx],
                    db_spl
                ])

    if not rows_data:
        print(f"  [SKIP] No rows above dB threshold in {filename}")
        return

    # ── NORMALIZE ──────────────────────────────────────────────────────────────
    global_min = min(all_values)
    global_max = max(all_values)
    print(f"  DOA value range: [{global_min:.4f}, {global_max:.4f}]")

    normalized_rows = []
    for timestamp, arr in rows_data:
        norm_arr = []
        for v in arr:
            if v == 0.0:
                norm_arr.append(0.0)
            else:
                norm = (v - global_min) / (global_max - global_min)
                norm_arr.append(norm * NORMALIZING_FACTOR)
        normalized_rows.append([timestamp] + norm_arr)

    # ── ASSERT Vicon angles are finite ────────────────────────────────────────
    vicon_angles = np.array([e[0] for e in extra_cols])
    bad = np.where(~np.isfinite(vicon_angles))[0]
    if len(bad) > 0:
        raise AssertionError(
            f"Vicon angle has {len(bad)} non-finite value(s) at indices {bad.tolist()}"
        )
    print(f"  Vicon angle range: [{vicon_angles.min():.2f}, {vicon_angles.max():.2f}]")

    # ── RING ATTRACTOR ─────────────────────────────────────────────────────────
    n_timesteps = len(normalized_rows)
    bump_angles = np.zeros((n_timesteps, N_BUMP_RUNS))

    for bumpCount, row in enumerate(normalized_rows):
        print(f"  Timestep {bumpCount + 1}/{n_timesteps} — {row[0]}")
        for countIn in range(N_BUMP_RUNS):
            ring         = RA()
            ring.h_ext   = row[1:]

            for _ in range(ring.updates_per_step):
                i       = np.random.randint(0, ring.Ns)
                delta_H = utils.compute_delta_H(ring, i)
                if delta_H < 0:
                    ring.spins[i] = 1 - ring.spins[i]
                else:
                    p = np.exp(-ring.beta * delta_H)
                    if np.random.rand() < p:
                        ring.spins[i] = 1 - ring.spins[i]

            active_indices = np.where(ring.spins == 1)[0]
            if len(active_indices) > 0:
                phi = np.angle(np.sum(np.exp(1j * ring.thetas[ring.spins == 1])))
            else:
                phi = 0.0

            bump_angles[bumpCount, countIn] = math.degrees(phi)

    # ── SAVE CSV ───────────────────────────────────────────────────────────────
    bump_cols   = [f"bump{i + 1}" for i in range(N_BUMP_RUNS)]
    extra_names = ["angle", "sample_index", "time_seconds", "left_volume", "right_volume", "dB_SPL"]
    out_header  = ["timestamp"] + bump_cols + extra_names

    with open(output_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        for i, row in enumerate(normalized_rows):
            timestamp  = row[0]
            bumps      = bump_angles[i].tolist()
            extras     = extra_cols[i]
            writer.writerow([timestamp] + bumps + extras)

    print(f"  Saved CSV  → {output_csv_path}")

    # ── HEATMAP DATA ───────────────────────────────────────────────────────────
    bin_edges = np.arange(-180, 181, 10)
    n_bins    = len(bin_edges) - 1
    heatmap   = np.zeros((n_timesteps, n_bins))
    for t in range(n_timesteps):
        counts, _ = np.histogram(bump_angles[t], bins=bin_edges)
        heatmap[t] = counts

    # ── PLOT ───────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7))

    gs = GridSpec(
        nrows=2, ncols=2,
        figure=fig,
        width_ratios=[20, 1],
        height_ratios=[10, 1],
        hspace=0.05,
        wspace=0.02
    )
    gs_cb       = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:, 1], wspace=0.6)
    ax_cb_heat  = fig.add_subplot(gs_cb[0, 0])
    ax_cb_vicon = fig.add_subplot(gs_cb[0, 1])
    ax_heat     = fig.add_subplot(gs[0, 0])
    ax_vicon    = fig.add_subplot(gs[1, 0])

    # Top: bump angle heatmap
    im_heat = ax_heat.imshow(
        heatmap.T,
        aspect="auto",
        origin="lower",
        extent=[0, n_timesteps, -180, 180]
    )
    fig.colorbar(im_heat, cax=ax_cb_heat, label="Count")
    ax_heat.set_ylabel("Angle (degrees)")
    ax_heat.set_title(f"Binned Bump Angles — {filename}")
    ax_heat.set_xticklabels([])

    # Bottom: Vicon angle strip (dynamic scale)
    vicon_min   = float(vicon_angles.min())
    vicon_max   = float(vicon_angles.max())
    vicon_strip = vicon_angles.reshape(1, -1)

    im_vicon = ax_vicon.imshow(
        vicon_strip,
        aspect="auto",
        origin="lower",
        extent=[0, n_timesteps, 0, 1],
        cmap="plasma",
        vmin=vicon_min,
        vmax=vicon_max
    )
    fig.colorbar(im_vicon, cax=ax_cb_vicon,
                 label=f"Vicon angle (°) [{vicon_min:.1f}, {vicon_max:.1f}]")
    ax_vicon.set_xlabel("Timestamp index")
    ax_vicon.set_yticks([])

    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {output_plot_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run Ring Attractor on merged CSVs and save bump angle results."
    )
    parser.add_argument(
        "--exp", type=str, default=None,
        help="Experiment group to process (e.g. expA, expBA). "
             "If omitted, all files are processed."
    )
    args = parser.parse_args()

    # Determine output subfolder and file glob
    if args.exp:
        pattern    = os.path.join(PROCESS_DATA_DIR, f"{args.exp}_*.csv")
        output_dir = os.path.join(BUMP_RESULTS_DIR, args.exp)
    else:
        pattern    = os.path.join(PROCESS_DATA_DIR, "*.csv")
        output_dir = BUMP_RESULTS_DIR

    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching: {pattern}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(files)} file(s) to process.")
    print(f"Output directory: {output_dir}")

    for input_path in files:
        stem             = os.path.splitext(os.path.basename(input_path))[0]
        output_csv_path  = os.path.join(output_dir, f"{stem}_bumps.csv")
        output_plot_path = os.path.join(output_dir, f"{stem}_plot.png")
        try:
            process_file(input_path, output_csv_path, output_plot_path)
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(input_path)}: {e}")

    print(f"\nAll done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()