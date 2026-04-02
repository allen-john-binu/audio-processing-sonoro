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

input_angles = [a for a in range(1,91)]

bump_angles = np.zeros((len(input_angles), 20))

for bumpCount, row in enumerate(input_angles):
    print("Processing timestamp: ", row)
    for countIn in range(20): 
        ring = RA()

        utils.compute_h_ext_multiple(ring,[math.radians(row), -math.radians(row)])

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
        print("row angle: ", row, "bump angle: ", bump_angles[bumpCount, countIn])

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