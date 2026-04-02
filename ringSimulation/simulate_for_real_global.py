import numpy as np
from random import seed as py_seed
import csv
import math
import utils
import copy
import matplotlib.pyplot as plt

seed = 100
np.random.seed(seed)
py_seed(seed)

target_pos1 = np.array([50.0, 50.0])  
target_pos2 = np.array([50.0, -50.0])  

class RA():
    def __init__(self):
        self.Ns = 120
        self.v = 0.5
        self.h_0 = 0.051
        self.h_b = 0.0122
        self.v_0 = 60
        self.h_ext = None
        self.beta = 4000
        self.sigma_ang = 2*np.pi / self.Ns
        self.thetas = np.linspace(-np.pi, np.pi, self.Ns, endpoint=False)
        self.spins = np.random.choice([1,0], size=self.Ns)
        self.pos = np.zeros(2)
        self.allocentric = False
        self.heading = 0
        self.updates_per_step = int(round(self.Ns * 4))

import csv

normalizing_factor = 0.1546

input_file = "../dataFromReal/doa_results/expA/roomA_test_20260330_114124.csv"
output_file = "expanded_output.csv"
all_values = []
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
                    all_values.append(value)  # ONLY real values

            rows_data.append((timestamp, full_array))

# ---------- GLOBAL MIN/MAX ----------
global_min = min(all_values)
global_max = max(all_values)

print("Min:", global_min)
print("Max:", global_max)

# ---------- SECOND PASS: normalize ----------
normalized_rows = []

for timestamp, arr in rows_data:
    norm_arr = []

    for v in arr:
        if v == 0.0:
            norm_arr.append(0.0)  # keep padded zeros as zero
        else:
            norm = (v - global_min) / (global_max - global_min)
            norm *= normalizing_factor
            norm_arr.append(norm)

    normalized_rows.append([timestamp] + norm_arr)

# ---------- WRITE OUTPUT ----------
angles_full = list(range(-180, 180, 3))
header_out = ["timestamp"] + [str(a) for a in angles_full]

with open(output_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_out)
    writer.writerows(normalized_rows)

print("Done! Data normalized to 0–0.1546")

print("length of normalized_rows: ",len(normalized_rows))

# # Pick a random row
# rwo_to_plot = normalized_rows[168]

# timestamp = rwo_to_plot[0]
# y = rwo_to_plot[1:]

# # X axis (angles)
# x = list(range(-180, 180, 3))

# # Labels
# xlabel = "Angle (degrees)"
# ylabel = "Normalized Value"
# title = f"Radiation Pattern @ {timestamp}"

# # Plot
# plt.figure(figsize=(5, 4))
# plt.plot(x, y)
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.title(title)
# plt.show()


bump_angles = np.zeros(len(normalized_rows))

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
    if bump_angles[bumpCount] < -60:
        print("timestamp: ", timestamp, "bump angle: ", bump_angles[bumpCount])

print("length of bump_angles: ",len(bump_angles))

#Labels
xlabel = "time step"
ylabel = "bump angle (degrees)"
title = "Bump Angle Over Time"

# Plot
plt.figure(figsize=(5, 4))
plt.plot(bump_angles)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.show()