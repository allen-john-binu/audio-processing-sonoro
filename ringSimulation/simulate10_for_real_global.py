import numpy as np
from random import seed as py_seed
import csv
import math
import utils
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
        self.sigma_ang = 2*np.pi / self.Ns
        self.thetas = np.linspace(-np.pi, np.pi, self.Ns, endpoint=False)
        self.spins = np.random.choice([1,0], size=self.Ns)
        self.pos = np.zeros(2)
        self.allocentric = False
        self.heading = 0
        self.updates_per_step = int(round(self.Ns * 4))

normalizing_factor = 0.1546

input_file = "../processData/expA_exA1.csv"
output_file = "expanded_output.csv"
all_values = []
rows_data  = []

# ── FIRST PASS: read + collect values ─────────────────────────────────────────
with open(input_file, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)

    csv_angles = [int(float(a)) for a in header[2:-5]]

    for row in reader:
        db_spl = float(row[1].strip("[]"))

        if db_spl > 90:
            timestamp  = row[0]
            values     = [float(x) for x in row[2:]]
            full_array = [0.0] * 120

            for angle, value in zip(csv_angles, values):
                idx = (angle + 180) // 3
                if 0 <= idx < 120:
                    full_array[idx] = value
                    all_values.append(value)

            rows_data.append((timestamp, full_array))

# ── GLOBAL MIN/MAX ─────────────────────────────────────────────────────────────
global_min = min(all_values)
global_max = max(all_values)
print("Min:", global_min)
print("Max:", global_max)

# ── SECOND PASS: normalize ────────────────────────────────────────────────────
normalized_rows = []

for timestamp, arr in rows_data:
    norm_arr = []
    for v in arr:
        if v == 0.0:
            norm_arr.append(0.0)
        else:
            norm = (v - global_min) / (global_max - global_min)
            norm *= normalizing_factor
            norm_arr.append(norm)
    normalized_rows.append([timestamp] + norm_arr)

# ── WRITE OUTPUT ───────────────────────────────────────────────────────────────
angles_full = list(range(-180, 180, 3))
header_out  = ["timestamp"] + [str(a) for a in angles_full]

with open(output_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_out)
    writer.writerows(normalized_rows)

print("Done! Data normalized to 0–0.1546")
print("length of normalized_rows:", len(normalized_rows))

# ── RING ATTRACTOR ─────────────────────────────────────────────────────────────
bump_angles  = np.zeros((len(normalized_rows), 20))
vicon_angles = []   # collect Vicon angle per filtered timestep

# Re-read the CSV to extract Vicon angle for filtered rows
with open(input_file, newline='') as f:
    reader    = csv.reader(f)
    header    = next(reader)
    # Vicon angle is the last column before the 5 Real columns
    # Based on merged CSV: timestamp, dB_SPL, [61 DOA cols], angle, sample_index, time_seconds, left_volume, right_volume
    angle_col_idx = header.index("angle")

    for row in reader:
        db_spl = float(row[1].strip("[]"))
        if db_spl > 90:
            vicon_angles.append(float(row[angle_col_idx]))

# ── ASSERT Vicon angles are within 0–180 ──────────────────────────────────────
vicon_angles = np.array(vicon_angles)
out_of_range = np.where((vicon_angles < 0) | (vicon_angles > 180))[0]
if len(out_of_range) > 0:
    raise AssertionError(
        f"Vicon angle out of [0, 180] range at {len(out_of_range)} timestep(s): "
        f"indices {out_of_range.tolist()}, values {vicon_angles[out_of_range].tolist()}"
    )
print("Vicon angle assertion passed — all values within [0, 180]")

for bumpCount, row in enumerate(normalized_rows):
    print("Processing timestamp:", row[0])
    for countIn in range(20):
        ring      = RA()
        timestamp = row[0]
        h_ext_values = row[1:]

        ring.h_ext = h_ext_values

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
        n_active       = len(active_indices)
        if n_active > 0:
            phi = np.angle(np.sum(np.exp(1j * ring.thetas[ring.spins == 1])))
        else:
            phi = 0.0

        bump_angles[bumpCount, countIn] = math.degrees(phi)
        if bump_angles[bumpCount, countIn] < -60:
            print("timestamp:", timestamp, "bump angle:", bump_angles[bumpCount, countIn])

print("length of bump_angles:", len(bump_angles))

# ── HEATMAP ────────────────────────────────────────────────────────────────────
bin_edges   = np.arange(-180, 181, 10)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
n_bins      = len(bin_centers)
n_timesteps = len(bump_angles)

heatmap = np.zeros((n_timesteps, n_bins))
for t in range(n_timesteps):
    counts, _ = np.histogram(bump_angles[t], bins=bin_edges)
    heatmap[t] = counts

# ── PLOT: heatmap (top) + Vicon angle strip (bottom) + shared colorbars ──────
fig = plt.figure(figsize=(14, 7))

# GridSpec: 2 rows for plots, 2 cols — plots in col 0, colorbars in col 1
# col 1 is split into 2 sub-rows for the two colorbars side by side (stacked vertically)
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

gs = GridSpec(
    nrows=2, ncols=2,
    figure=fig,
    width_ratios=[20, 1],       # plots much wider than colorbar column
    height_ratios=[10, 1],
    hspace=0.05,
    wspace=0.02
)

# — Colorbar column: split into 2 side-by-side colorbars —
gs_cb = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:, 1], wspace=0.6)
ax_cb_heat  = fig.add_subplot(gs_cb[0, 0])   # left colorbar  → Count
ax_cb_vicon = fig.add_subplot(gs_cb[0, 1])   # right colorbar → Vicon angle

ax_heat  = fig.add_subplot(gs[0, 0])
ax_vicon = fig.add_subplot(gs[1, 0])

# — Top: bump angle heatmap (transposed: X = timestamp, Y = angle) —
im_heat = ax_heat.imshow(
    heatmap.T,
    aspect="auto",
    origin="lower",
    extent=[0, n_timesteps, -180, 180]
)
fig.colorbar(im_heat, cax=ax_cb_heat, label="Count")
ax_heat.set_ylabel("Angle (degrees)")
ax_heat.set_title("Binned Bump Angles Over Time")
ax_heat.set_xticklabels([])   # hide X labels — shared with strip below

# — Bottom: Vicon angle strip (1 row, colour-coded 0–180) —
vicon_strip = vicon_angles.reshape(1, -1)   # shape (1, n_timesteps)

im_vicon = ax_vicon.imshow(
    vicon_strip,
    aspect="auto",
    origin="lower",
    extent=[0, n_timesteps, 0, 1],
    cmap="plasma",
    vmin=0,
    vmax=180
)
fig.colorbar(im_vicon, cax=ax_cb_vicon, label="Vicon angle (°)")
ax_vicon.set_xlabel("Timestamp index")
ax_vicon.set_yticks([])       # no Y axis on strip

plt.show()