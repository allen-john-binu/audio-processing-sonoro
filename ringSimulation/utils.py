import numpy as np
import copy
import matplotlib.pyplot as plt

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

def plot_summary_no_target(ring,pos_alloc,pos_ego,spins_history,bump_angles,L):

    all_pos = np.vstack([pos_ego, pos_alloc])

    xmin, ymin = all_pos.min(axis=0)
    xmax, ymax = all_pos.max(axis=0)

    # optional padding
    pad = 0.05 * max(xmax - xmin, ymax - ymin)
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    t = np.arange(pos_alloc.shape[0])  
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # network activity as a function of time
    ax_raster = axes[0] 
    im = ax_raster.imshow(spins_history.T, aspect='auto', origin='lower',
                        cmap='gray', interpolation='nearest', extent=[0, L, -180, 180])
    ax_raster.set_title(f" beta={ring.beta:.2f}  — network spins over time")
    ax_raster.set_ylabel("unit index")
    ax_raster.set_xlabel("time step")
    time_axis = np.arange(L) 
    # ax_raster.plot(time_axis, bump_angles, color='red', linewidth=2, label='bump angle')

    # Add legend (colorbar)
    cbar = plt.colorbar(im, ax=ax_raster, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0 (black)', '1 (white)'])
    cbar.set_label("Spin value")

    # egocentric trajectory
    # ax_ego = axes[1] 
    # pos = pos_ego
    # ax_ego.scatter(pos[:, 0], pos[:, 1], linewidth=1)
    # ax_ego.scatter(pos[0, 0], pos[0, 1], color='green', s=20, label='start')
    # ax_ego.scatter(pos[-1, 0], pos[-1, 1], color='red', s=20, label='end')
    # ax_ego.set_title(f" Egocentric (beta={ring.beta:.2f})")
    # ax_ego.set_aspect('equal')
    # ax_ego.set_xlabel("x")
    # ax_ego.set_ylabel("y")
    # ax_ego.legend()

    ax_ego = axes[1]
    ax_ego.set_xlim(xmin, xmax)
    ax_ego.set_ylim(ymin, ymax)
    ax_ego.set_aspect('equal', adjustable='box')
    pos = pos_ego
    t = np.arange(pos.shape[0])

    sc_ego = ax_ego.scatter(
        pos[:, 0],
        pos[:, 1],
        c=t,
        cmap='viridis',
        s=10
    )

    # ax_ego.scatter(pos[0, 0], pos[0, 1], color='green', s=30, label='start')
    # ax_ego.scatter(pos[-1, 0], pos[-1, 1], color='red', s=30, label='end')

    ax_ego.set_title(f"Egocentric (beta={ring.beta:.2f})")
    ax_ego.set_aspect('equal')
    ax_ego.set_xlabel("x")
    ax_ego.set_ylabel("y")
    ax_ego.legend()

    cbar_ego = plt.colorbar(sc_ego, ax=ax_ego)
    cbar_ego.set_label("Time step")


    # allocentric trajectory
    # ax_alloc = axes[2] 
    # pos = pos_alloc
    # ax_alloc.scatter(pos[:, 0], pos[:, 1], linewidth=1)
    # ax_alloc.scatter(pos[0, 0], pos[0, 1], color='green', s=20, label='start')
    # ax_alloc.scatter(pos[-1, 0], pos[-1, 1], color='red', s=20, label='end')
    # ax_alloc.set_title(f" Allocentric (beta={ring.beta:.2f})")
    # ax_alloc.set_aspect('equal')
    # ax_alloc.set_xlabel("x")
    # ax_alloc.set_ylabel("y")
    # ax_alloc.legend()

    ax_alloc = axes[2]
    ax_alloc.set_xlim(xmin, xmax)
    ax_alloc.set_ylim(ymin, ymax)
    ax_alloc.set_aspect('equal', adjustable='box')
    pos = pos_alloc
    t = np.arange(pos.shape[0])

    sc_alloc = ax_alloc.scatter(
        pos[:, 0],
        pos[:, 1],
        c=t,
        cmap='viridis',
        s=10
    )

    # ax_alloc.scatter(pos[0, 0], pos[0, 1], color='green', s=30, label='start')
    # ax_alloc.scatter(pos[-1, 0], pos[-1, 1], color='red', s=30, label='end')

    ax_alloc.set_title(f"Allocentric (beta={ring.beta:.2f})")
    ax_alloc.set_aspect('equal')
    ax_alloc.set_xlabel("x")
    ax_alloc.set_ylabel("y")
    ax_alloc.legend()

    cbar_allo = plt.colorbar(sc_alloc, ax=ax_alloc)
    cbar_allo.set_label("Time step")


    plt.tight_layout()
    plt.show()

def plot_summary_with_target(ring,pos,spins_history,bump_angles,target_pos,L,end_needed=True):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # network activity as a function of time
    ax_raster = axes[0] 
    im = ax_raster.imshow(spins_history.T, aspect='auto', origin='lower',
                        cmap='gray', interpolation='nearest', extent=[0, L, -180, 180])
    ax_raster.set_title(f" beta={ring.beta:.2f}  — network spins over time")
    ax_raster.set_ylabel("unit index")
    ax_raster.set_xlabel("time step")
    time_axis = np.arange(L) 
    ax_raster.plot(time_axis, bump_angles[0:L], color='red', linewidth=2, label='bump angle')

    # Add legend (colorbar)
    cbar = plt.colorbar(im, ax=ax_raster, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0 (black)', '1 (white)'])
    cbar.set_label("Spin value")

    #  trajectory
    ax_alloc = axes[1] 
    ax_alloc.plot(pos[:L, 0], pos[:L, 1], linewidth=1)
    ax_alloc.scatter(pos[0, 0], pos[0, 1], color='green', s=20, label='start')
    if end_needed:
        ax_alloc.scatter(pos[-1, 0], pos[-1, 1], color='red', s=20, label='end')
    ax_alloc.scatter(*target_pos, color='black', marker='x', s=40, label='target')
    type_str = "Allocentric" if ring.allocentric else "Egocentric"
    ax_alloc.set_title(f" {type_str}  (beta={ring.beta:.2f})")
    ax_alloc.set_aspect('equal')
    ax_alloc.set_xlabel("x")
    ax_alloc.set_ylabel("y")

    plt.tight_layout()
    plt.show()

def plot_summary_with_multiple_targets(ring, pos, spins_history, bump_angles, target_positions, L):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # -------- Network activity over time --------
    ax_raster = axes[0]
    im = ax_raster.imshow(spins_history[:L].T, aspect='auto', origin='lower',
                          cmap='gray', interpolation='nearest',
                          extent=[0, L, -180, 180])

    ax_raster.set_title(f"beta={ring.beta:.2f} — network spins over time")
    ax_raster.set_ylabel("unit index")
    ax_raster.set_xlabel("time step")

    time_axis = np.arange(L)
    ax_raster.plot(time_axis, bump_angles[:L], color='red', linewidth=2, label='bump angle')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_raster, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0 (black)', '1 (white)'])
    cbar.set_label("Spin value")

    # -------- Trajectory --------
    ax_alloc = axes[1]
    ax_alloc.plot(pos[:L, 0], pos[:L, 1], linewidth=1)
    ax_alloc.scatter(pos[0, 0], pos[0, 1], color='green', s=20, label='start')

    # Plot multiple targets
    for i, target_pos in enumerate(target_positions):
        if i == 0:
            ax_alloc.scatter(*target_pos, color='black', marker='x', s=40, label='target')
        else:
            ax_alloc.scatter(*target_pos, color='black', marker='x', s=40)
    
    type_str = "Allocentric" if ring.allocentric else "Egocentric"

    ax_alloc.set_title(f"{type_str} (beta={ring.beta:.2f})")
    ax_alloc.set_aspect('equal')
    ax_alloc.set_xlabel("x")
    ax_alloc.set_ylabel("y")
    ax_alloc.legend()

    plt.tight_layout()
    plt.show()


def plot_any_line(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()