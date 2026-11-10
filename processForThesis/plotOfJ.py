import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


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

        self.spins = np.random.choice(
            [1, 0],
            size=self.Ns
        )

        self.pos = np.zeros(2)
        self.allocentric = False
        self.heading = 0
        self.updates_per_step = int(round(self.Ns * 4))


# ============================================================
# Kernel calculation
#
# This is the same calculation as in your code.
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
# Plot one ring
# ============================================================

def plot_ring(ring, i, nu, output_filename):
    """
    Plot the 120-neuron ring for a given value of nu.

    Parameters
    ----------
    ring : RA
        Ring attractor object.

    i : int
        Reference neuron index.
        Here we use i = 60 exactly.

    nu : float
        Exponent in the kernel.

    output_filename : str
        Name of the output PDF file.
    """

    # Set the exponent
    ring.v = nu

    # Compute kernel values relative to neuron i
    J = compute_J(ring, i)

    # --------------------------------------------------------
    # Convert theta positions to Cartesian coordinates
    # --------------------------------------------------------

    x = np.cos(ring.thetas)
    y = np.sin(ring.thetas)

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    # --------------------------------------------------------
    # Blue-white-red colormap
    #
    # Positive J -> blue
    # Zero      -> white
    # Negative J -> red
    # --------------------------------------------------------

    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red",
        [
            "darkred",
            "red",
            "white",
            "cornflowerblue",
            "blue",
            "darkblue"
        ]
    )

    norm = Normalize(
        vmin=-1,
        vmax=1
    )

    # --------------------------------------------------------
    # Plot all 120 neurons
    # --------------------------------------------------------

    ax.scatter(
        x,
        y,
        c=J,
        cmap=cmap,
        norm=norm,
        s=55,                  # neuron size
        edgecolors="none",
        zorder=2
    )

    # --------------------------------------------------------
    # Highlight reference neuron i = 60
    # --------------------------------------------------------

    ax.scatter(
        x[i],
        y[i],
        s=75,
        c="black",
        edgecolors="black",
        linewidths=0.5,
        zorder=3
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

    # Create the ring
    ring = RA()

    # Reference neuron
    # IMPORTANT:
    # This is Python index 60, i.e. the 61st neuron
    # when counted from 1.
    i = 60

    # Values of nu requested
    nu_values = [
        0.3,
        0.4,
        0.5,
        0.6
    ]

    # Generate one PDF for each value of nu
    for nu in nu_values:

        filename = f"ring_nu_{nu:.1f}.pdf"

        plot_ring(
            ring=ring,
            i=i,
            nu=nu,
            output_filename=filename
        )

        print(
            f"Created: {filename}"
        )