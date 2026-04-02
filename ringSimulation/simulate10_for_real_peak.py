import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.signal as signal
from random import seed as py_seed
import math
import utils

#negative angle robot is on the left, positive angle robot is on the right. 

# -----------------------------
# 1. Load CSV
# -----------------------------
file_path = "../dataFromReal/doa_results/expA/roomA_test_20260330_114124.csv"
df = pd.read_csv(file_path)

angle_cols = df.columns[2:]
angles = np.array([float(a) for a in angle_cols])

spl_array = df["dB_SPL"].astype(str).str.strip("[]").astype(float).values
intensity_matrix = df.iloc[:, 2:].values.astype(float)

# -----------------------------
# 2. Angle spacing fix
# -----------------------------
angle_step = angles[1] - angles[0]
y_min = angles[0] - angle_step / 2
y_max = angles[-1] + angle_step / 2
# -----------------------------

threshold_init = 72

# -----------------------------
# 4. Filtering function (REAL filtering)
# -----------------------------
def filter_data(threshold):
    mask = spl_array >= threshold

    # Filter rows completely
    filtered_intensity = intensity_matrix[mask]
    filtered_spl = spl_array[mask]

    peak_angles_1 = []
    peak_angles_2 = []
    intensity_diffs = []

    for row in filtered_intensity:
        peaks, _ = signal.find_peaks(row)

        if len(peaks) >= 2:
            # Take top 2 peaks by intensity
            top2 = peaks[np.argsort(row[peaks])[-2:]]
            p1, p2 = top2

            a1, a2 = angles[p1], angles[p2]
            i1, i2 = row[p1], row[p2]

            peak_angles_1.append(a1)
            peak_angles_2.append(a2)

            intensity_diffs.append(abs(i1 - i2))

        elif len(peaks) == 1:
            # Only one peak
            p1 = peaks[0]

            a1 = angles[p1]
            i1 = row[p1]

            peak_angles_1.append(a1)
            peak_angles_2.append(np.nan)

            intensity_diffs.append(np.nan)

        else:
            # No peaks
            peak_angles_1.append(np.nan)
            peak_angles_2.append(np.nan)
            intensity_diffs.append(np.nan)

    new_time = np.arange(len(filtered_intensity))

    return (
        filtered_intensity,
        np.array(peak_angles_1),
        np.array(peak_angles_2),
        np.array(intensity_diffs),
        new_time,
        filtered_spl
    )

# Initial data
filtered_intensity, peak1, peak2, intensity_diffs, time_steps, filtered_spl = filter_data(threshold_init)

print("Initial filtering done. Number of valid rows: ", len(peak1), len(peak2))

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

bump_angles = np.zeros((len(peak1), 20))

for (bumpCount, (angle1, angle2)) in enumerate(zip(peak1, peak2)):
    print("Processing time step: ", bumpCount, "peak angles: ", angle1, angle2)
    for countIn in range(20): 
        ring = RA()

        if not np.isnan(angle1) and not np.isnan(angle2):
            utils.compute_h_ext_multiple(ring,[math.radians(angle1), math.radians(angle2)])
        elif not np.isnan(angle1):
            utils.compute_h_ext_multiple(ring,[math.radians(angle1)])
        elif not np.isnan(angle2):
            utils.compute_h_ext_multiple(ring,[math.radians(angle2)])

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
plt.ylabel("target angle (degrees)")
plt.title("Binned Bump Angles Over target angles")

plt.show()

# for angle1, angle2 in zip(peak1, peak2):
#     print("Peak 1 angle: ", angle1)
#     print("Peak 2 angle: ", angle2)