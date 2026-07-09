import numpy as np
from random import seed as py_seed
import csv
import math
import utils
import ast
import matplotlib
matplotlib.use("Agg")

import time

start_time = time.time()
combo = 0

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 100
np.random.seed(SEED)
py_seed(SEED)

N_BUMP_RUNS = 50
INPUT_CSV = "sample.csv"
OUTPUT_CSV = "parameter_study_results.csv"

# ---------------------------------------------------------------------
# Parameter sweep (coarse study)
# ---------------------------------------------------------------------
v_values = np.arange(0.3, 0.8, 0.1)
h_b_values = np.arange(0.005, 0.02, 0.001)
nf_values = np.arange(0.05, 0.30, 0.01)

# ---------------------------------------------------------------------
# Ring Attractor
# ---------------------------------------------------------------------
class RA():
    def __init__(self):
        self.Ns = 120
        self.v = 0.5
        self.h_0 = 0.051
        self.h_b = 0.0122
        self.v_0 = 60
        self.h_ext = None
        self.beta = 400
        self.sigma_ang = 2 * np.pi / self.Ns
        self.thetas = np.linspace(-np.pi, np.pi,
                                  self.Ns,
                                  endpoint=False)
        self.spins = np.random.choice([1, 0], size=self.Ns)
        self.pos = np.zeros(2)
        self.allocentric = False
        self.heading = 0
        self.updates_per_step = int(round(self.Ns * 4))


# ---------------------------------------------------------------------
# Circular bump finding
# ---------------------------------------------------------------------
def find_circular_groups(spins):
    """
    Returns:
        [
            (start_idx, end_idx, length),
            ...
        ]

    Ring is circular, so first and last groups are merged.
    """
    n = len(spins)

    if np.all(spins == 0):
        return []

    if np.all(spins == 1):
        return [(0, n - 1, n)]

    groups = []

    in_group = False
    start = None

    for i in range(n):
        if spins[i] == 1 and not in_group:
            start = i
            in_group = True

        elif spins[i] == 0 and in_group:
            groups.append((start, i - 1, i - start))
            in_group = False

    if in_group:
        groups.append((start, n - 1, n - start))

    # merge wraparound groups
    if len(groups) >= 2:
        first_starts_at_zero = groups[0][0] == 0
        last_ends_at_end = groups[-1][1] == n - 1

        if first_starts_at_zero and last_ends_at_end:
            first = groups[0]
            last = groups[-1]

            merged_length = first[2] + last[2]

            merged = (
                last[0],
                first[1],
                merged_length
            )

            groups = [merged] + groups[1:-1]

    return groups


# ---------------------------------------------------------------------
# Circular mean angle
# ---------------------------------------------------------------------
def circular_mean_angle(theta_subset):
    if len(theta_subset) == 0:
        return 0.0

    s = np.sum(np.sin(theta_subset))
    c = np.sum(np.cos(theta_subset))

    return math.degrees(np.arctan2(s, c))


# ---------------------------------------------------------------------
# Extract lengths and angles of all bumps
# ---------------------------------------------------------------------
def extract_bump_statistics(spins, thetas):

    groups = find_circular_groups(spins)

    bump_lengths = []
    bump_angles = []

    n = len(spins)

    for start, end, length in groups:

        bump_lengths.append(int(length))

        # wraparound bump
        if start > end:
            idx = list(range(start, n))
            idx.extend(range(0, end + 1))
            theta_subset = thetas[idx]
        else:
            theta_subset = thetas[start:end + 1]

        bump_angles.append(
            circular_mean_angle(theta_subset)
        )

    return len(groups), bump_lengths, bump_angles


# ---------------------------------------------------------------------
# Read exactly one row from sample.csv
# ---------------------------------------------------------------------
def build_full_array():

    with open(INPUT_CSV, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)

    doa_headers = header[2:]
    csv_angles = [int(float(a)) for a in doa_headers]
    values = [float(x) for x in row[2:]]

    full_array = np.zeros(120)

    for angle, value in zip(csv_angles, values):
        idx = (angle + 180) // 3

        if 0 <= idx < 120:
            full_array[idx] = value

    return full_array
# ---------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------
def normalize_array(arr, nf):

    arr = np.array(arr)

    nonzero = arr[arr != 0]

    if len(nonzero) == 0:
        return arr.copy()

    gmin = nonzero.min()
    gmax = nonzero.max()

    denom = gmax - gmin

    out = np.zeros_like(arr)

    if denom == 0:
        return out

    mask = arr != 0

    out[mask] = (
        (arr[mask] - gmin)
        / denom
        * nf
    )

    return out


# ---------------------------------------------------------------------
# Main parameter study
# ---------------------------------------------------------------------
def main():

    base_array = build_full_array()

    total_combinations = (
        len(v_values)
        * len(h_b_values)
        * len(nf_values)
    )

    print(f"Parameter combinations : {total_combinations}")
    print(
        f"Total simulations      : "
        f"{total_combinations * N_BUMP_RUNS}"
    )

    with open(OUTPUT_CSV,
              "w",
              newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "v",
            "h_b",
            "nf",
            "run",
            "n_bumps",
            "bump_lengths",
            "bump_angles",
            "phi"
        ])

        combo = 0

        for nf in nf_values:

            h_ext = normalize_array(
                base_array,
                nf
            )

            for v in v_values:
                for h_b in h_b_values:

                    combo += 1

                    elapsed = time.time() - start_time
                    avg = elapsed / combo
                    remaining = (total_combinations - combo) * avg

                    print(
                        f"[{combo}/{total_combinations}] "
                        f"v={v:.2f}, "
                        f"h_b={h_b:.4f}, "
                        f"nf={nf:.2f}, "
                        f"ETA={remaining/60:.1f} min"
                    )

                    for run in range(N_BUMP_RUNS):

                        ring = RA()

                        ring.v = float(v)
                        ring.h_b = float(h_b)
                        ring.h_ext = h_ext

                        for _ in range(
                                ring.updates_per_step):

                            i = np.random.randint(
                                0,
                                ring.Ns
                            )

                            delta_H = (
                                utils.compute_delta_H(
                                    ring,
                                    i
                                )
                            )

                            if delta_H < 0:
                                ring.spins[i] = (
                                    1 - ring.spins[i]
                                )
                            else:
                                p = np.exp(
                                    -ring.beta * delta_H
                                )

                                if np.random.rand() < p:
                                    ring.spins[i] = (
                                        1 - ring.spins[i]
                                    )

                        n_bumps, lengths, angles = (
                            extract_bump_statistics(
                                ring.spins,
                                ring.thetas
                            )
                        )

                        active = np.where(
                            ring.spins == 1
                        )[0]

                        if len(active) > 0:
                            phi = math.degrees(
                                np.angle(
                                    np.sum(
                                        np.exp(
                                            1j
                                            * ring.thetas[
                                                active
                                            ]
                                        )
                                    )
                                )
                            )
                        else:
                            phi = 0.0

                        writer.writerow([
                            v,
                            h_b,
                            nf,
                            run + 1,
                            n_bumps,
                            str(lengths),
                            str(angles),
                            phi
                        ])

                        f.flush()

    print("Finished.")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()