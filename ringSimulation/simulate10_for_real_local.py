import numpy as np
from random import seed as py_seed
import csv
import math
import utils
import copy
import matplotlib.pyplot as plt

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

bump_angles = np.zeros((len(normalized_rows), 20))

for bumpCount, row in enumerate(normalized_rows):
    print("Processing timestamp: ", row[0])
    for countIn in range(20): 
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
        bump_angles[bumpCount, countIn] = math.degrees(phi)
        if bump_angles[bumpCount, countIn] < -60:
            print("timestamp: ", timestamp, "bump angle: ", bump_angles[bumpCount, countIn])

bin_edges = np.arange(-180, 181, 10)   # edges
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
n_bins = len(bin_centers)

heatmap = np.zeros((len(bump_angles), n_bins))

for t in range(len(bump_angles)):
    values = bump_angles[t]
    
    counts, _ = np.histogram(values, bins=bin_edges)
    heatmap[t] = counts

plt.figure(figsize=(12, 6))

plt.imshow(
    heatmap,
    aspect='auto',
    origin='lower',
    extent=[-180, 180, 0, len(bump_angles)]
)

plt.colorbar(label="Count")
plt.xlabel("Angle (degrees)")
plt.ylabel("Timestamp index")
plt.title("Binned Bump Angles Over Time")

plt.show()