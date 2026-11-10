
import numpy as np
from random import seed as py_seed
import matplotlib.pyplot as plt
import utils

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

seed = 100
np.random.seed(seed)
py_seed(seed)

target_pos1 = np.array([150.0, 30.0])
target_pos2 = np.array([150.0, -30.0])

L = 1000
N_RUNS = 20


class RA:
    def __init__(self):
        self.Ns = 120
        self.v = 0.7
        self.h_0 = 0.051
        self.h_b = 0.0122
        self.v_0 = 60
        self.h_ext = None
        self.beta = 4000
        self.sigma_ang = 5 * np.pi / self.Ns
        self.thetas = np.linspace(
            -np.pi, np.pi, self.Ns, endpoint=False
        )
        self.spins = np.random.choice([1, 0], size=self.Ns)
        self.pos = np.zeros(2)
        self.allocentric = False
        self.heading = 0
        self.updates_per_step = int(round(self.Ns * 4))


# --------------------------------------------------
# 2. Utility functions
# --------------------------------------------------

def euclidean_distance(p1, p2):
    return np.linalg.norm(p2 - p1)


def target_separation_angle(position):
    """
    Angle between directions from position to both targets.
    Returns an angle in degrees, between 0 and 180.
    """
    vec1 = target_pos1 - position
    vec2 = target_pos2 - position

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return np.nan

    # atan2 is numerically robust and handles the full angle range.
    cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    dot = np.dot(vec1, vec2)

    return np.degrees(np.arctan2(abs(cross), dot))


def calculate_separation_angles(trajectory):
    """Calculate target separation angle at every recorded position."""
    return np.array([
        target_separation_angle(p) for p in trajectory
    ])



def find_turning_point(trajectory):
    """
    Find the position with the largest absolute heading change
    between consecutive valid movement vectors.

    Returns:
        index: trajectory index of the turning point
        position: coordinates of the turning point
        turn_angle: heading change in degrees
    """
    n = len(trajectory)

    if n < 3:
        return None, None, np.nan

    displacement = np.diff(trajectory, axis=0)
    displacement_norm = np.linalg.norm(displacement, axis=1)

    turn_angles = np.full(n - 2, np.nan)

    for i in range(1, n - 1):
        v1 = displacement[i - 1]
        v2 = displacement[i]

        if (
            displacement_norm[i - 1] <= 1e-12
            or displacement_norm[i] <= 1e-12
        ):
            continue

        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = np.dot(v1, v2)

        turn_angles[i - 1] = np.degrees(
            np.arctan2(abs(cross), dot)
        )

    if np.all(np.isnan(turn_angles)):
        return None, None, np.nan

    # Index within turn_angles, not the trajectory.
    angle_idx = int(np.nanargmax(turn_angles))

    # Corresponding position index in the trajectory.
    position_idx = angle_idx + 1

    return (
        position_idx,
        trajectory[position_idx].copy(),
        turn_angles[angle_idx]
    )


# --------------------------------------------------
# 3. One simulation run
# --------------------------------------------------

def run_simulation(L=1000):
    ring = RA()

    pos = np.zeros((L, 2))
    heading = 0.0

    previous_distance1 = 1_000_000
    previous_distance2 = 1_000_000

    target_reached_L = -1

    for t in range(L):

        dist_to_target1 = euclidean_distance(pos[t], target_pos1)
        dist_to_target2 = euclidean_distance(pos[t], target_pos2)

        # Preserve the original stopping condition.
        if (dist_to_target1 < 10.0) or (dist_to_target2 < 10.0):
            target_reached_L = t
            break

        previous_distance1 = dist_to_target1
        previous_distance2 = dist_to_target2

        theta_target1 = np.arctan2(
            target_pos1[1] - pos[t, 1],
            target_pos1[0] - pos[t, 0]
        )

        theta_target2 = np.arctan2(
            target_pos2[1] - pos[t, 1],
            target_pos2[0] - pos[t, 0]
        )

        if ring.allocentric:
            theta_diff1 = theta_target1
            theta_diff2 = theta_target2
        else:
            theta_diff1 = (theta_target1 - heading) % (2 * np.pi)
            theta_diff2 = (theta_target2 - heading) % (2 * np.pi)

        utils.compute_h_ext_multiple(
            ring, [theta_diff1, theta_diff2]
        )

        # Original Metropolis spin updates.
        for _ in range(ring.updates_per_step):
            i = np.random.randint(0, ring.Ns)
            delta_H = utils.compute_delta_H(ring, i)

            if delta_H < 0:
                ring.spins[i] = 1 - ring.spins[i]
            else:
                p = np.exp(-ring.beta * delta_H)
                if np.random.rand() < p:
                    ring.spins[i] = 1 - ring.spins[i]

        active_indices = np.where(ring.spins == 1)[0]

        if len(active_indices) > 0:
            phi = np.angle(
                np.sum(np.exp(1j * ring.thetas[active_indices]))
            )
        else:
            phi = 0.0

        # Update position, preserving the original motion rules.
        if ring.allocentric:
            if t < L - 1:
                pos[t + 1, 0] = (
                    pos[t, 0] + ring.v_0 * np.cos(phi) / ring.Ns
                )
                pos[t + 1, 1] = (
                    pos[t, 1] + ring.v_0 * np.sin(phi) / ring.Ns
                )
        else:
            heading = (phi + heading) % (2 * np.pi)

            if t < L - 1:
                pos[t + 1, 0] = (
                    pos[t, 0]
                    + ring.v_0 * np.cos(heading) / ring.Ns
                )
                pos[t + 1, 1] = (
                    pos[t, 1]
                    + ring.v_0 * np.sin(heading) / ring.Ns
                )

    # The final valid position is pos[t] when stopping early.
    if target_reached_L == -1:
        trajectory = pos.copy()
    else:
        trajectory = pos[:target_reached_L + 1].copy()

    return trajectory, target_reached_L


# --------------------------------------------------
# 4. Run 20 independent simulations
# --------------------------------------------------

trajectories = []
run_results = []

for run in range(N_RUNS):
    trajectory, stop_step = run_simulation(L)

    angles = calculate_separation_angles(trajectory)

    turn_idx, turn_pos, turn_angle = find_turning_point(
        trajectory
    )

    if turn_idx is not None:
        separation_at_turn = angles[turn_idx]
    else:
        separation_at_turn = np.nan

    trajectories.append(trajectory)

    run_results.append({
        "run": run + 1,
        "trajectory": trajectory,
        "angles": angles,
        "stop_step": stop_step,
        "turn_idx": turn_idx,
        "turn_pos": turn_pos,
        "turn_angle": turn_angle,
        "separation_at_turn": separation_at_turn,
    })

    if turn_idx is not None:
        print(
            f"Run {run + 1:02d} | "
            f"Turning step: {turn_idx} | "
            f"Position: ({turn_pos[0]:.3f}, {turn_pos[1]:.3f}) | "
            f"Turn angle: {turn_angle:.2f} deg | "
            f"Target separation: {separation_at_turn:.2f} deg"
        )
    else:
        print(
            f"Run {run + 1:02d} | "
            "No valid turning point detected."
        )


# --------------------------------------------------
# 5. Plot the 20 overlaid trajectories
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 8))

for result in run_results:
    tr = result["trajectory"]

    ax.plot(
        tr[:, 0], tr[:, 1],
        linewidth=1.2,
        alpha=0.7,
        label=f"Run {result['run']}"
    )

    ax.scatter(tr[0, 0], tr[0, 1], s=15, marker="o")

    if result["turn_pos"] is not None:
        ax.scatter(
            result["turn_pos"][0],
            result["turn_pos"][1],
            s=65,
            marker="X",
            edgecolors="black",
            linewidths=0.5
        )

ax.scatter(
    *target_pos1,
    marker="*",
    s=220,
    edgecolors="black",
    label="Target 1"
)

ax.scatter(
    *target_pos2,
    marker="*",
    s=220,
    edgecolors="black",
    label="Target 2"
)

ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title("20 Trajectories with Turning Points")
ax.grid(True, alpha=0.3)
ax.set_aspect("equal", adjustable="datalim")
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    fontsize=8
)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 6. Plot target separation angle versus time
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

for result in run_results:
    angles = result["angles"]
    time_steps = np.arange(len(angles))

    ax.plot(
        time_steps,
        angles,
        linewidth=1.4,
        alpha=0.8,
        label=f"Run {result['run']}"
    )

    # Mark the separation angle at the detected turning point.
    idx = result["turn_idx"]

    if idx is not None and np.isfinite(angles[idx]):
        ax.scatter(
            idx,
            angles[idx],
            marker="X",
            s=65,
            edgecolors="black",
            linewidths=0.5,
            zorder=5
        )

ax.set_xlabel("Time step")
ax.set_ylabel("Angle between target directions (degrees)")
ax.set_title("Target Separation Angle vs Time — 20 Runs")
ax.set_ylim(0, 180)
ax.grid(True, alpha=0.3)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    fontsize=8
)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. Summary table
# --------------------------------------------------

print("\nSUMMARY OF TURNING POINTS")
print("-" * 100)
print(
    f"{'Run':>4} {'Step':>8} {'X':>12} {'Y':>12} "
    f"{'Turn angle (deg)':>19} {'Target separation (deg)':>26}"
)

for result in run_results:
    idx = result["turn_idx"]
    p = result["turn_pos"]

    if idx is None:
        print(f"{result['run']:>4} {'N/A':>8}")
        continue

    print(
        f"{result['run']:>4} "
        f"{idx:>8} "
        f"{p[0]:>12.4f} "
        f"{p[1]:>12.4f} "
        f"{result['turn_angle']:>19.4f} "
        f"{result['separation_at_turn']:>26.4f}"
    )
