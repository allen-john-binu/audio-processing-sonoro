import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


# ============================================================
# Ring Attractor (RA)
# ============================================================

class RA:
    def __init__(self):
        self.Ns = 120

        # Parameters from your original code
        self.h_0 = 0.051
        self.sigma_ang = 2 * np.pi / self.Ns

        # 120 neuron angular positions
        self.thetas = np.linspace(
            -np.pi,
            np.pi,
            self.Ns,
            endpoint=False
        )

        # This will contain h_ext for all 120 neurons
        self.h_ext = None


# ============================================================
# External input calculation
#
# This is your code preserved exactly in terms of the
# mathematical operation and circular angular difference.
# ============================================================

def compute_h_ext(ring, theta_diff):

    difference_target = (
        (ring.thetas - theta_diff + np.pi)
        % (2 * np.pi)
        - np.pi
    )

    ring.h_ext = (
        ring.h_0
        / np.sqrt(2 * np.pi * (ring.sigma_ang ** 2))
    ) * np.exp(
        -(difference_target ** 2)
        / (2 * (ring.sigma_ang ** 2))
    )

    return ring.h_ext


# ============================================================
# Plot the ring
# ============================================================

def plot_h_ext_ring(
    ring,
    theta_diff,
    output_filename
):

    # --------------------------------------------------------
    # Calculate h_ext for all 120 neurons
    # --------------------------------------------------------

    h_ext = compute_h_ext(
        ring,
        theta_diff
    )

    # --------------------------------------------------------
    # Convert angular positions to Cartesian coordinates
    #
    # theta = 0°     -> right
    # theta = 90°    -> upper
    # theta = 45°    -> upper-right
    # --------------------------------------------------------

    x = np.cos(ring.thetas)
    y = np.sin(ring.thetas)

    # --------------------------------------------------------
    # Green colormap
    #
    # Low h_ext  -> light green
    # High h_ext -> dark green
    # --------------------------------------------------------

    green_cmap = LinearSegmentedColormap.from_list(
        "custom_green",
        [
            "#E8F5E9",  # very light green
            "#C8E6C9",
            "#81C784",
            "#4CAF50",
            "#2E7D32",
            "#006400"   # dark green
        ]
    )

    # Normalize according to the actual h_ext range
    norm = Normalize(
        vmin=np.min(h_ext),
        vmax=np.max(h_ext)
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    # --------------------------------------------------------
    # Plot 120 neurons
    # --------------------------------------------------------

    ax.scatter(
        x,
        y,
        c=h_ext,
        cmap=green_cmap,
        norm=norm,
        s=55,
        edgecolors="none"
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_aspect("equal")

    # No axes, labels, ticks, etc.
    ax.axis("off")

    # Give a little space around the ring
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)

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

    # Target angle = 45 degrees
    theta_diff = np.deg2rad(45)

    # Output filename
    output_filename = "ring_h_ext_theta_45.pdf"

    # Generate the ring
    plot_h_ext_ring(
        ring=ring,
        theta_diff=theta_diff,
        output_filename=output_filename
    )

    print(f"Created: {output_filename}")

    # --------------------------------------------------------
    # Optional numerical check
    # --------------------------------------------------------
    #
    # The neuron at 45 degrees is Python index 75 because
    # the neurons are separated by 3 degrees.
    #
    # This section is only for checking the calculation and
    # does NOT add anything to the PDF.
    # --------------------------------------------------------

    target_index = np.argmin(
        np.abs(
            (
                ring.thetas
                - theta_diff
                + np.pi
            ) % (2 * np.pi) - np.pi
        )
    )

    print(f"Target neuron index: {target_index}")
    print(f"Target neuron angle: "
          f"{np.rad2deg(ring.thetas[target_index]):.2f} degrees")
    print(f"Maximum h_ext: {np.max(ring.h_ext):.6f}")
    print(f"Minimum h_ext: {np.min(ring.h_ext):.6e}")