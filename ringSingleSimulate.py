import numpy as np
from random import seed as py_seed
import csv
import math
import utils
import os
import sys
import argparse
import glob

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 100
np.random.seed(SEED)
py_seed(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
NORMALIZING_FACTOR = 0.1546
DB_SPL_THRESHOLD   = 50
PROCESS_DATA_DIR   = "./ztLabCollection/ztProcessData"
BUMP_RESULTS_DIR   = "./ztBumpResultsSingle"


# ── Ring Attractor ─────────────────────────────────────────────────────────────
class RA():
    def __init__(self):
        self.Ns               = 120
        self.v                = 0.5
        self.h_0              = 0.051
        self.h_b              = 0.0122
        self.v_0              = 60
        self.h_ext            = None
        self.beta             = 400
        self.sigma_ang        = 2 * np.pi / self.Ns
        self.thetas           = np.linspace(-np.pi, np.pi, self.Ns, endpoint=False)
        self.spins            = np.random.choice([1, 0], size=self.Ns)
        self.pos              = np.zeros(2)
        self.allocentric      = False
        self.heading          = 0
        self.updates_per_step = int(round(self.Ns * 4))


# ── Geometry helper ────────────────────────────────────────────────────────────
def angle_at_robot(robot_x, robot_y,
                   speakerL_x, speakerL_y,
                   speakerR_x, speakerR_y):
    """
    Interior angle at the robot between robot→speakerL and robot→speakerR.
    Returns angle in degrees [0, 180].
    """
    v1 = np.array([speakerL_x - robot_x, speakerL_y - robot_y], dtype=float)
    v2 = np.array([speakerR_x - robot_x, speakerR_y - robot_y], dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return np.nan

    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# ── Per-file pipeline ──────────────────────────────────────────────────────────
def process_file(input_path, output_csv_path):
    filename = os.path.basename(input_path)
    print(f"\n{'=' * 60}")
    print(f"Processing: {filename}")
    print(f"{'=' * 60}")

    all_values = []
    rows_data  = []
    extra_cols = []  # [angle, speakerL_x, speakerL_y, speakerR_x, speakerR_y,
                     #  robot_x, robot_y, sample_index, time_seconds,
                     #  left_volume, right_volume, db_spl]

    # ── FIRST PASS: filter + build ring arrays ─────────────────────────────────
    with open(input_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Find where DOA columns end (numeric angles from -90 to 90)
        robot_x_idx = header.index("robot_x")
        doa_headers = header[2:robot_x_idx]
        csv_angles  = [int(float(a)) for a in doa_headers]

        speakerL_x_idx = header.index("speakerL_x")
        speakerL_y_idx = header.index("speakerL_y")
        speakerR_x_idx = header.index("speakerR_x")
        speakerR_y_idx = header.index("speakerR_y")
        robot_y_idx    = header.index("robot_y")
        sample_idx     = header.index("sample_index")
        time_idx       = header.index("time_seconds")
        left_vol_idx   = header.index("left_volume")
        right_vol_idx  = header.index("right_volume")

        for row in reader:
            db_spl = float(
                str(row[1]).replace("[", "").replace("]", "").strip()
            )
            if db_spl > DB_SPL_THRESHOLD:
                timestamp  = row[0]
                values     = [float(x) for x in row[2:robot_x_idx]]
                full_array = [0.0] * 120

                for angle, value in zip(csv_angles, values):
                    idx = (angle + 180) // 3
                    if 0 <= idx < 120:
                        full_array[idx] = value
                        all_values.append(value)

                rows_data.append((timestamp, full_array))

                robot_x    = float(row[robot_x_idx])
                robot_y    = float(row[robot_y_idx])
                speakerL_x = float(row[speakerL_x_idx])
                speakerL_y = float(row[speakerL_y_idx])
                speakerR_x = float(row[speakerR_x_idx])
                speakerR_y = float(row[speakerR_y_idx])

                angle = angle_at_robot(
                    robot_x, robot_y,
                    speakerL_x, speakerL_y,
                    speakerR_x, speakerR_y
                )

                extra_cols.append([
                    angle,
                    speakerL_x,
                    speakerL_y,
                    speakerR_x,
                    speakerR_y,
                    robot_x,
                    robot_y,
                    row[sample_idx],
                    row[time_idx],
                    row[left_vol_idx],
                    row[right_vol_idx],
                    db_spl,
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
                denom = global_max - global_min
                norm  = 0.0 if denom == 0 else (v - global_min) / denom
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
    spin_cols   = [f"spin_{i}" for i in range(120)]

    out_header = (
        ["timestamp", "bump_angle"]
        + spin_cols
        + [
            "angle",
            "speakerL_x", "speakerL_y",
            "speakerR_x", "speakerR_y",
            "robot_x", "robot_y",
            "sample_index", "time_seconds",
            "left_volume", "right_volume",
            "dB_SPL",
        ]
    )

    with open(output_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        
        ring = RA()

        for bumpCount, row in enumerate(normalized_rows):
            print(f"  Timestep {bumpCount + 1}/{n_timesteps} — {row[0]}")

            ring.h_ext = row[1:]
            print("length of hext is :",len(row[1:]))

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

            bump_angle = math.degrees(phi)

            writer.writerow(
                [row[0], bump_angle]
                + ring.spins.tolist()
                + extra_cols[bumpCount]
            )

    print(f"  Saved CSV  → {output_csv_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run Ring Attractor on merged CSVs and save bump angle + spin results."
    )
    parser.add_argument(
        "--exp", type=str, default=None,
        help="Experiment group to process (e.g. expA, expBA). "
             "If omitted, all files in data-dir are processed."
    )
    parser.add_argument(
        "--data-dir", type=str, default=PROCESS_DATA_DIR,
        help=f"Directory containing input CSVs (default: {PROCESS_DATA_DIR})."
    )
    args = parser.parse_args()

    if args.exp:
        pattern    = os.path.join(args.data_dir, f"{args.exp}*.csv")
        output_dir = os.path.join(BUMP_RESULTS_DIR, args.exp)
    else:
        pattern    = os.path.join(args.data_dir, "*.csv")
        output_dir = BUMP_RESULTS_DIR

    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching: {pattern}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(files)} file(s) to process.")
    print(f"Output directory: {output_dir}")

    for input_path in files:
        stem            = os.path.splitext(os.path.basename(input_path))[0]
        output_csv_path = os.path.join(output_dir, f"{stem}_bumps.csv")
        try:
            process_file(input_path, output_csv_path)
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(input_path)}: {e}")

    print(f"\nAll done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()