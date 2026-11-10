import numpy as np
import copy

def compute_h_ext(ring, theta_diff):
    difference_target = (ring.thetas - theta_diff + np.pi) % (2 * np.pi) - np.pi
    ring.h_ext = (ring.h_0 / np.sqrt(2 * np.pi * (ring.sigma_ang**2))) * np.exp(- (difference_target ** 2) / (2 * (ring.sigma_ang**2) ) )

def compute_h_ext_multiple(ring, theta_diffs):
    ring.h_ext = np.zeros_like(ring.thetas)
    for theta_diff in theta_diffs:
        difference_target = (ring.thetas - theta_diff + np.pi) % (2 * np.pi) - np.pi
        contribution = (ring.h_0 / np.sqrt(2 * np.pi * (ring.sigma_ang**2))) * \
                       np.exp(-(difference_target ** 2) / (2 * (ring.sigma_ang**2)))
        ring.h_ext += contribution

def compute_J(ring,i):
    kernel = None
    alpha_i = copy.deepcopy(ring.thetas[i])
    angle_diffs = np.abs((copy.deepcopy(ring.thetas) - alpha_i + np.pi) % (2 * np.pi) - np.pi)
    kernel = np.cos(np.pi * ((angle_diffs / np.pi) ** ring.v))
    return kernel

def compute_delta_H(ring, i):
    H = np.zeros(2)
    j_curr = compute_J(ring,i)
    j_curr[i] = 0 

    interaction_sum = np.dot(j_curr, ring.spins)
   
    spin_flip = np.array([ring.spins[i], 1-ring.spins[i]])

    interaction_energy = (interaction_sum * spin_flip) / (ring.Ns - 1)

    if ring.h_ext is None:
        input_energy = 0
    else:
        input_energy = ring.h_ext[i] * spin_flip

    leak_energy = ring.h_b * spin_flip
    H = -1 * (interaction_energy + input_energy - leak_energy)
    return (H[1] - H[0])

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()