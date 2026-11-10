import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# Ring Attractor (RA)
# ============================================================

class RA:
    def __init__(self):
        self.Ns = 120
        self.v = 0.5
        self.h_0 = 0.051
        self.h_b = 0.0122
        self.v_0 = 60
        self.h_ext = None
        self.beta = 400
        self.sigma_ang = 2 * np.pi / self.Ns

        # 120 equally spaced neurons from -pi to pi
        self.thetas = np.linspace(
            -np.pi,
            np.pi,
            self.Ns,
            endpoint=False
        )

        # Random initial spin state
        # 1 = active
        # 0 = inactive
        self.spins = np.random.choice(
            [1, 0],
            size=self.Ns
        )

        self.pos = np.zeros(2)
        self.allocentric = False
        self.heading = 0

        # 120 * 4 = 480 attempted updates
        self.updates_per_step = int(round(self.Ns * 4))


# ============================================================
# Kernel calculation
# ============================================================

def compute_J(ring, i):

    alpha_i = ring.thetas[i]

    angle_diffs = np.abs(
        (ring.thetas - alpha_i + np.pi) % (2 * np.pi) - np.pi
    )

    kernel = np.cos(
        np.pi * (
            (angle_diffs / np.pi) ** ring.v
        )
    )

    return kernel


# ============================================================
# Compute delta H
# ============================================================

def compute_delta_H(ring, i):

    H = np.zeros(2)

    j_curr = compute_J(ring, i)

    # No self-interaction
    j_curr[i] = 0

    interaction_sum = np.dot(
        j_curr,
        ring.spins
    )

    spin_flip = np.array([
        ring.spins[i],
        1 - ring.spins[i]
    ])

    interaction_energy = (
        interaction_sum * spin_flip
    ) / (ring.Ns - 1)

    if ring.h_ext is None:
        input_energy = 0
    else:
        input_energy = (
            ring.h_ext[i] * spin_flip
        )

    leak_energy = (
        ring.h_b * spin_flip
    )

    H = -1 * (
        interaction_energy
        + input_energy
        - leak_energy
    )

    return H[1] - H[0]

def compute_h_ext(ring, theta_diff):
    difference_target = (ring.thetas - theta_diff + np.pi) % (2 * np.pi) - np.pi
    ring.h_ext = (ring.h_0 / np.sqrt(2 * np.pi * (ring.sigma_ang**2))) * np.exp(- (difference_target ** 2) / (2 * (ring.sigma_ang**2) ) )
    return ring

# ============================================================
# Run one dynamic
# ============================================================

def run_dynamic(ring):

    # Exactly ring.updates_per_step attempted updates
    # In the current RA class, this is 480.
    ring = compute_h_ext(ring,0)
    for _ in range(ring.updates_per_step):

        i = np.random.randint(
            0,
            ring.Ns
        )

        delta_H = compute_delta_H(
            ring,
            i
        )

        if delta_H < 0:

            ring.spins[i] = 1 - ring.spins[i]

        else:

            p = np.exp(
                -ring.beta * delta_H
            )

            if np.random.rand() < p:

                ring.spins[i] = 1 - ring.spins[i]


# ============================================================
# Plot final ring state
# ============================================================

def plot_ring_state(
    ring,
    v,
    output_filename
):

    # --------------------------------------------------------
    # Convert theta positions to Cartesian coordinates
    # --------------------------------------------------------

    x = np.cos(ring.thetas)
    y = np.sin(ring.thetas)

    # --------------------------------------------------------
    # Colormap
    #
    # spin = 0 -> black
    # spin = 1 -> yellow
    # --------------------------------------------------------

    cmap = ListedColormap([
        "black",
        "yellow"
    ])

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    # --------------------------------------------------------
    # Plot the final spin state
    # --------------------------------------------------------

    ax.scatter(
        x,
        y,
        c=ring.spins,
        cmap=cmap,
        vmin=0,
        vmax=1,
        s=55,
        edgecolors="none",
        zorder=2
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_aspect("equal")

    # No axes
    ax.axis("off")

    # Keep some space around the ring
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)

    # --------------------------------------------------------
    # Save as PDF
    # --------------------------------------------------------

    plt.savefig(
        output_filename,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Values of v
    v_values = [
        0.3,
        0.4,
        0.5,
        0.6
    ]

    # --------------------------------------------------------
    # Run an independent simulation for each v
    # --------------------------------------------------------

    for v in v_values:

        # New ring = new random initial spin state
        ring = RA()

        # Set exponent for this simulation
        ring.v = v

        # Run exactly one dynamic
        # = 480 attempted spin updates
        run_dynamic(ring)

        # Output filename
        filename = f"ring_state_v_{v:.1f}.pdf"

        # Plot final state
        plot_ring_state(
            ring=ring,
            v=v,
            output_filename=filename
        )

        print(
            f"Created: {filename}"
        )