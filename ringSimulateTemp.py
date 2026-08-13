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
DB_SPL_THRESHOLD   = 93 # removing the threshold filter 
N_BUMP_RUNS        = 50
PROCESS_DATA_DIR   = "./ztLabCollection/ztProcessData"
BUMP_RESULTS_DIR   = "./ztBumpResults"


# ── Ring Attractor ─────────────────────────────────────────────────────────────
class RA():
    def __init__(self):
        self.Ns              = 120
        self.v               = 0.3
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


# ── Helper Functions for Group Analysis ────────────────────────────────────────
def find_contiguous_groups(spins):
    """
    Find all contiguous blocks of 1s in the spins array.
    
    Args:
        spins: 1D numpy array of 0s and 1s
    
    Returns:
        List of tuples: [(start_idx, end_idx, length), ...]
        Sorted by length in descending order.
        Returns empty list if no 1s found.
    """
    groups = []
    in_group = False
    start_idx = 0
    
    for i in range(len(spins)):
        if spins[i] == 1:
            if not in_group:
                start_idx = i
                in_group = True
        else:  # spins[i] == 0
            if in_group:
                length = i - start_idx
                groups.append((start_idx, i - 1, length))
                in_group = False
    
    # Handle case where array ends with 1s
    if in_group:
        length = len(spins) - start_idx
        groups.append((start_idx, len(spins) - 1, length))
    
    # Sort by length descending
    groups.sort(key=lambda x: x[2], reverse=True)
    return groups


def circular_mean_angle(thetas_subset):
    """
    Compute circular mean of angles using atan2.
    
    Args:
        thetas_subset: 1D array of angles in radians
    
    Returns:
        Circular mean in degrees (range -180 to 180)
    """
    if len(thetas_subset) == 0:
        return 0.0
    
    sin_sum = np.sum(np.sin(thetas_subset))
    cos_sum = np.sum(np.cos(thetas_subset))
    circular_mean_rad = np.arctan2(sin_sum, cos_sum)
    circular_mean_deg = math.degrees(circular_mean_rad)
    return circular_mean_deg


def extract_group_stats(spins, thetas, ring_obj):
    """
    Extract largest and second-largest contiguous 1s groups.
    
    Args:
        spins: 1D numpy array of spins (0s and 1s)
        thetas: 1D numpy array of angles (in radians)
        ring_obj: The RA ring object (for reference, though thetas are passed)
    
    Returns:
        Tuple: (group1_length, group1_angle, group2_length, group2_angle)
        If fewer than 2 groups, missing ones default to (0, 0.0)
    """
    groups = find_contiguous_groups(spins)
    
    # Initialize defaults
    group1_length, group1_angle = 0, 0.0
    group2_length, group2_angle = 0, 0.0
    
    if len(groups) >= 1:
        start1, end1, len1 = groups[0]
        group1_length = len1
        thetas_group1 = thetas[start1:end1 + 1]
        group1_angle = circular_mean_angle(thetas_group1)
    
    if len(groups) >= 2:
        start2, end2, len2 = groups[1]
        group2_length = len2
        thetas_group2 = thetas[start2:end2 + 1]
        group2_angle = circular_mean_angle(thetas_group2)
    
    return group1_length, group1_angle, group2_length, group2_angle


def angle_at_robot(robot_x, robot_y,
                   speakerL_x, speakerL_y,
                   speakerR_x, speakerR_y):
    """
    Interior angle at robot between
    robot→speakerL and robot→speakerR.

    Returns angle in degrees [0, 180].
    """

    v1 = np.array([
        speakerL_x - robot_x,
        speakerL_y - robot_y
    ], dtype=float)

    v2 = np.array([
        speakerR_x - robot_x,
        speakerR_y - robot_y
    ], dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return np.nan

    cos_theta = np.dot(v1, v2) / (n1 * n2)

    # numerical safety
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.degrees(np.arccos(cos_theta))

# ── Per-file pipeline ──────────────────────────────────────────────────────────
def process_file(input_path, output_csv_path, output_plot_path, include_groups=False):
    filename = os.path.basename(input_path)
    print(f"\n{'=' * 60}")
    print(f"Processing: {filename}")
    print(f"{'=' * 60}")

    all_values = []
    rows_data  = []

    # ── FIRST PASS: filter + build ring arrays ─────────────────────────────────
    with open(input_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

        expected_header = (
            ["timestamp", "dB_SPL"] +
            [f"{a:.1f}" for a in np.arange(-90, 91, 3)]
        )

        if header != expected_header:
            raise ValueError(
                "Invalid CSV header.\n"
                "Expected:\n"
                + ",".join(expected_header)
            )

        csv_angles = np.arange(-90, 91, 3)

        for row in reader:
            db_spl = float(str(row[1]).strip("[]"))

            if db_spl <= DB_SPL_THRESHOLD:
                continue

            timestamp = row[0]

            # DOA values only
            values = [float(str(x).strip("[]")) for x in row[2:]]

            full_array = [0.0] * 120

            for angle, value in zip(csv_angles, values):
                idx = int((angle + 180) // 3)
                full_array[idx] = value
                all_values.append(value)

            rows_data.append((timestamp, full_array))
        
    if not rows_data:
        print(f"  [SKIP] No rows above dB threshold in {filename}")
        return

    # ── NORMALIZE ──────────────────────────────────────────────────────────────
    global_min = min(all_values)
    global_max = max(all_values)
    print(f"  DOA value range: [{global_min:.4f}, {global_max:.4f}]")

    normalized_rows = []
    for timestamp, arr in rows_data:
        
        # print("length of hext is :",len(arr))
        norm_arr = []
        for v in arr:
            if v == 0.0:
                norm_arr.append(0.0)
            else:
                denom = global_max - global_min
                if denom == 0:
                    norm = 0.0
                else:
                    norm = (v - global_min) / denom
                norm_arr.append(norm * NORMALIZING_FACTOR)
        normalized_rows.append([timestamp] + norm_arr)

    # ── RING ATTRACTOR ─────────────────────────────────────────────────────────
    n_timesteps = len(normalized_rows)
    bump_angles = np.zeros((n_timesteps, N_BUMP_RUNS))
    
    if include_groups:
        # New arrays for group statistics 
        group1_lengths = np.zeros((n_timesteps, N_BUMP_RUNS))
        group1_angles  = np.zeros((n_timesteps, N_BUMP_RUNS))
        group2_lengths = np.zeros((n_timesteps, N_BUMP_RUNS))
        group2_angles  = np.zeros((n_timesteps, N_BUMP_RUNS))

    for bumpCount, row in enumerate(normalized_rows):
        print(f"  Timestep {bumpCount + 1}/{n_timesteps} — {row[0]}")
        for countIn in range(N_BUMP_RUNS):
            ring         = RA()
            ring.h_ext   = row[1:]
            # print("length of hext is :",len(row[1:]))

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
            
            if include_groups:
                # Extract group statistics from final spins state
                g1_len, g1_ang, g2_len, g2_ang = extract_group_stats(
                    ring.spins, ring.thetas, ring
                )
                group1_lengths[bumpCount, countIn] = g1_len
                group1_angles[bumpCount, countIn]  = g1_ang
                group2_lengths[bumpCount, countIn] = g2_len
                group2_angles[bumpCount, countIn]  = g2_ang

    # ── SAVE CSV ───────────────────────────────────────────────────────────────
    bump_cols       = [f"bump{i + 1}" for i in range(N_BUMP_RUNS)]
    group1_len_cols = [f"group1_len_{i + 1}" for i in range(N_BUMP_RUNS)]
    group1_ang_cols = [f"group1_angle_{i + 1}" for i in range(N_BUMP_RUNS)]
    group2_len_cols = [f"group2_len_{i + 1}" for i in range(N_BUMP_RUNS)]
    group2_ang_cols = [f"group2_angle_{i + 1}" for i in range(N_BUMP_RUNS)]
    
    out_header = (["timestamp"] + bump_cols)
    
    if include_groups:
        out_header += (
            group1_len_cols +
            group1_ang_cols +
            group2_len_cols +
            group2_ang_cols
        )
    

    with open(output_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        for i, row in enumerate(normalized_rows):
            
            timestamp  = row[0]
            bumps      = bump_angles[i].tolist()
            row_data = [timestamp] + bumps

            if include_groups:
                g1_lens    = group1_lengths[i].tolist()
                g1_angs    = group1_angles[i].tolist()
                g2_lens    = group2_lengths[i].tolist()
                g2_angs    = group2_angles[i].tolist()
                row_data += (
                    g1_lens +
                    g1_angs +
                    g2_lens +
                    g2_angs
                )


            writer.writerow(row_data)

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

    gs = GridSpec(1,2,figure=fig,width_ratios=[20,1],wspace=0.02)
    ax_heat = fig.add_subplot(gs[0,0])
    ax_cb_heat = fig.add_subplot(gs[0,1])

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

    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {output_plot_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run Ring Attractor on a single CSV."
    )

    parser.add_argument(
        "input_csv",
        help="Input CSV file"
    )

    parser.add_argument(
        "--output-csv",
        default="bumps.csv",
        help="Output CSV filename"
    )

    parser.add_argument(
        "--output-plot",
        default="plot.png",
        help="Output plot filename"
    )

    parser.add_argument(
        "--include-groups",
        action="store_true",
        help="Include group statistics in output CSV."
    )

    args = parser.parse_args()

    process_file(
        args.input_csv,
        args.output_csv,
        args.output_plot,
        include_groups=args.include_groups,
    )

    print("Done.")


if __name__ == "__main__":
    main()