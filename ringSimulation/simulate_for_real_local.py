import numpy as np
from random import seed as py_seed
import csv
import math
import utils
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

seed = 100
np.random.seed(seed)
py_seed(seed)

class RA():
    def __init__(self):
        self.Ns = 120
        self.v = 0.5
        self.h_0 = 0.051
        self.h_b = 0.0122
        self.v_0 = 60
        self.h_ext = None
        self.beta = 4000
        self.sigma_ang = 5*np.pi / self.Ns
        self.thetas = np.linspace(-np.pi, np.pi, self.Ns, endpoint=False)
        self.spins = np.random.choice([1,0], size=self.Ns)
        self.pos = np.zeros(2)
        self.allocentric = False
        self.heading = 0
        self.updates_per_step = int(round(self.Ns * 4))

import csv

normalizing_factor = 0.1546

input_file = "../dataFromReal/doa_results/expA/roomA_test_20260330_114124.csv"
rows_data = []

# ---------- FIRST PASS: read + collect values ----------
with open(input_file, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)

    csv_angles = [int(float(a)) for a in header[2:]]

    for row in reader:
        # print("row[1]: ", row[1])

        # Remove brackets and convert to float
        db_spl = float(row[1].strip("[]"))

        if db_spl > 90:
            timestamp = row[0]
            values = [float(x) for x in row[2:]]

            full_array = [0.0] * 120

            for angle, value in zip(csv_angles, values):
                idx = (angle + 180) // 3
                if 0 <= idx < 120:
                    full_array[idx] = value

            rows_data.append((timestamp, full_array))

# ---------- NORMALIZE PER TIMESTAMP ----------
normalized_rows = []

for timestamp, arr in rows_data:
    # Get min/max for THIS array only (excluding zero-padded values)
    non_zero_values = [v for v in arr if v != 0.0]
    
    if len(non_zero_values) > 0:
        local_min = min(non_zero_values)
        local_max = max(non_zero_values)
    else:
        local_min = 0.0
        local_max = 1.0  # avoid division by zero
    
    norm_arr = []
    for v in arr:
        if v == 0.0:
            norm_arr.append(0.0)  # keep padded zeros as zero
        else:
            norm = (v - local_min) / (local_max - local_min)
            norm *= normalizing_factor
            norm_arr.append(norm)

    normalized_rows.append([timestamp] + norm_arr)

print("Done! Data normalized per timestamp to 0–0.1546")

print("length of normalized_rows: ",len(normalized_rows))

bump_angles = np.zeros(len(normalized_rows))
collection_hxt = np.zeros((len(normalized_rows), 120))
spins_history = np.zeros((len(normalized_rows), 120), dtype=int)

for bumpCount, row in enumerate(normalized_rows):
    ring = RA()
    timestamp = row[0]
    h_ext_values = row[1:]

    ring.h_ext = h_ext_values

    for _ in range (ring.updates_per_step):
        i = np.random.randint(0,ring.Ns)
        delta_H = utils.compute_delta_H(ring,i)

        if delta_H < 0:
            # Energy difference negative — always accept
            ring.spins[i] = 1 - ring.spins[i]
        else:
            # Energy difference positive — accept with probability exp(-beta * delta_H)
            p = np.exp(-ring.beta * delta_H)
            if np.random.rand() < p:
                ring.spins[i] = 1 - ring.spins[i]
                
    active_indices = np.where(ring.spins == 1)[0]
    n_active = len(active_indices)
    if n_active > 0:
        phi = np.angle(np.sum(np.exp(1j * ring.thetas[ring.spins == 1])))
    else:
        phi = 0.0  
    bump_angles[bumpCount] = math.degrees(phi)
    collection_hxt[bumpCount] = ring.h_ext
    spins_history[bumpCount] = copy.deepcopy(ring.spins)
    if bump_angles[bumpCount] < -60:
        print("timestamp: ", timestamp, "bump angle: ", bump_angles[bumpCount])

print("length of bump_angles: ",len(bump_angles))

howLong = len(bump_angles)

# Create figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

def update(forCount):
    ax_line, ax_raster = axes
    ax_line.clear()
    ax_raster.clear()

    y = collection_hxt[forCount]
    x = list(range(-180, 180, 3))

    # LEFT
    ax_line.plot(x, y)
    ax_line.axvline(x=bump_angles[forCount], color='red', linestyle='--', linewidth=2)
    ax_line.set_xlabel("Angle (degrees)")
    ax_line.set_ylabel("Normalized Value")
    ax_line.set_title("H ext at time step {}".format(forCount))

    # RIGHT
    im = ax_raster.imshow(
        spins_history[:howLong],
        aspect='auto',
        origin='lower',
        cmap='gray',
        interpolation='nearest',
        extent=[-180, 180, 0, howLong]
    )

    time_axis = np.arange(howLong)
    ax_raster.plot(bump_angles[:howLong], time_axis, color='red', linewidth=2)

    ax_raster.axhline(y=forCount, color='yellow', linestyle=':', linewidth=2)

    ax_raster.set_title(
        f"timestep={forCount}, bump angle={bump_angles[forCount]:.1f}°"
    )
    ax_raster.set_ylabel("time step")
    ax_raster.set_xlabel("angle")

    return []

ani = FuncAnimation(
    fig,
    update,
    frames=howLong,
    interval=1000,
    repeat=True
)

plt.show()

# - non gausian, there is no nice bump angle
# - 