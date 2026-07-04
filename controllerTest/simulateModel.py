import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER PARAMETERS
# ============================================================

# Estimated distance traveled by:
# motor speed = 80
# duration = 0.5 s
FORWARD_DISTANCE = 0.03          # meters (3 cm)

# Wheelbase: distance between left and right wheels (meters)
WHEELBASE = 0.095                # meters — update to your measured value

# Thymio2 motor speed at FORWARD_DISTANCE / 0.5s
# i.e. motor value that corresponds to FORWARD_DISTANCE in 0.5s
BASE_MOTOR_SPEED = 80            # Thymio2 units (maps to FORWARD_DISTANCE / 0.5s)

# Thymio2 motor range
MOTOR_MIN = -500
MOTOR_MAX = 500

# Turn speed used by Type A controller
TURN_SPEED_DEG_PER_S = 90.0     # deg/s — update to your measured value

# Bezier smoothness
ALPHA = 1.0

# Number of command steps (higher = smoother but more commands)
N_STEPS = 50

# Angles to simulate
TEST_ANGLES = [-90, -60, -30, 30, 60, 90]


# ============================================================
# TIMING
# ============================================================

def turn_time(theta_deg):
    """Time Type A spends rotating in place (seconds)."""
    return abs(theta_deg) / TURN_SPEED_DEG_PER_S


def total_time(theta_deg):
    """
    Shared time budget for both controllers:
        Type A: turn_time + 0.5s forward
        Type B: same total, spent entirely on the smooth curve
    """
    return turn_time(theta_deg) + 0.5


# ============================================================
# SPEED SCALING
# ============================================================

# Linear speed in m/s derived from calibration:
#   FORWARD_DISTANCE meters in 0.5 seconds
LINEAR_SPEED_MS = FORWARD_DISTANCE / 0.5   # m/s

# Thymio2 motor units per m/s
#   BASE_MOTOR_SPEED units → LINEAR_SPEED_MS m/s
MOTOR_PER_MS = BASE_MOTOR_SPEED / LINEAR_SPEED_MS


def clamp_motor(value):
    return np.clip(value, MOTOR_MIN, MOTOR_MAX)


def speed_to_motor(v_ms):
    """Convert m/s to Thymio2 motor units."""
    return clamp_motor(v_ms * MOTOR_PER_MS)


# ============================================================
# TYPE A (CURRENT CONTROLLER)
# ============================================================

def simulate_type_a(theta_deg):
    """
    Current behavior:
        rotate(theta)    — takes turn_time(theta) seconds
        forward(0.5s)    — at BASE_MOTOR_SPEED

    Returns:
        commands: list of (t_start, t_end, left, right)
        endpoint: (x, y)
    """
    theta = np.deg2rad(theta_deg)
    t_turn = turn_time(theta_deg)

    # During turn: one wheel forward, one backward (rotate in place)
    # Sign convention: positive theta = counter-clockwise = left wheel back
    turn_motor = BASE_MOTOR_SPEED
    if theta_deg >= 0:
        left_turn, right_turn = -turn_motor, turn_motor
    else:
        left_turn, right_turn = turn_motor, -turn_motor

    commands = []

    if t_turn > 0:
        commands.append((0.0, t_turn, left_turn, right_turn))

    commands.append((t_turn, t_turn + 0.5, BASE_MOTOR_SPEED, BASE_MOTOR_SPEED))

    x = FORWARD_DISTANCE * np.cos(theta)
    y = FORWARD_DISTANCE * np.sin(theta)

    return commands, (x, y)


# ============================================================
# CUBIC BEZIER
# ============================================================

def bezier_curve(P0, P1, P2, P3, n=300):
    t = np.linspace(0, 1, n)
    x = (
        (1 - t)**3 * P0[0]
        + 3 * (1 - t)**2 * t * P1[0]
        + 3 * (1 - t) * t**2 * P2[0]
        + t**3 * P3[0]
    )
    y = (
        (1 - t)**3 * P0[1]
        + 3 * (1 - t)**2 * t * P1[1]
        + 3 * (1 - t) * t**2 * P2[1]
        + t**3 * P3[1]
    )
    return x, y


def bezier_derivatives(P0, P1, P2, P3, n=300):
    """
    Analytical first and second derivatives of the cubic Bezier.
    Used to compute curvature without finite differences.
    """
    t = np.linspace(0, 1, n)

    # First derivative
    dx = (
        3 * (1 - t)**2 * (P1[0] - P0[0])
        + 6 * (1 - t) * t * (P2[0] - P1[0])
        + 3 * t**2 * (P3[0] - P2[0])
    )
    dy = (
        3 * (1 - t)**2 * (P1[1] - P0[1])
        + 6 * (1 - t) * t * (P2[1] - P1[1])
        + 3 * t**2 * (P3[1] - P2[1])
    )

    # Second derivative
    ddx = (
        6 * (1 - t) * (P2[0] - 2 * P1[0] + P0[0])
        + 6 * t * (P3[0] - 2 * P2[0] + P1[0])
    )
    ddy = (
        6 * (1 - t) * (P2[1] - 2 * P1[1] + P0[1])
        + 6 * t * (P3[1] - 2 * P2[1] + P1[1])
    )

    return dx, dy, ddx, ddy


def bezier_curvature(dx, dy, ddx, ddy):
    """
    κ = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
    Positive κ = left turn, negative κ = right turn.
    """
    num = dx * ddy - dy * ddx
    den = (dx**2 + dy**2) ** 1.5
    # Avoid division by zero
    kappa = np.where(den > 1e-12, num / den, 0.0)
    return kappa


# ============================================================
# TYPE B (BIOLOGICAL TRAJECTORY)
# ============================================================

def simulate_type_b(theta_deg, alpha=ALPHA, n_steps=N_STEPS):
    """
    Smooth Bezier trajectory converted to Thymio2 motor commands.

    Steps:
        1. Build Bezier control points (same as before)
        2. Compute curvature analytically at N_STEPS points
        3. Derive left/right wheel speeds from differential drive kinematics
        4. Scale to Thymio2 motor units
        5. Distribute evenly over total_time(theta_deg)

    Returns:
        commands:  list of (t_start, t_end, left_motor, right_motor)
        xs, ys:    Bezier path arrays (for plotting)
    """
    theta = np.deg2rad(theta_deg)
    d = FORWARD_DISTANCE
    T = total_time(theta_deg)

    # --- Control points ---
    P0 = np.array([0.0, 0.0])
    P1 = np.array([alpha * d, 0.0])
    P3 = np.array([d * np.cos(theta), d * np.sin(theta)])
    P2 = np.array([
        P3[0] - alpha * d * np.cos(theta),
        P3[1] - alpha * d * np.sin(theta)
    ])

    # --- Dense curve for plotting ---
    xs, ys = bezier_curve(P0, P1, P2, P3, n=300)

    # --- Sparse samples for motor commands ---
    t_param = np.linspace(0, 1, n_steps)

    # Positions at each step
    bx, by = bezier_curve(P0, P1, P2, P3, n=n_steps)

    # Derivatives at each step
    dx, dy, ddx, ddy = bezier_derivatives(P0, P1, P2, P3, n=n_steps)

    # Curvature at each step
    kappa = bezier_curvature(dx, dy, ddx, ddy)

    # --- Differential drive kinematics ---
    L = WHEELBASE
    v = LINEAR_SPEED_MS          # constant linear speed along the path

    v_l = v * (1 - kappa * L / 2)
    v_r = v * (1 + kappa * L / 2)

    # Scale to Thymio2 motor units
    left_motors  = speed_to_motor(v_l)
    right_motors = speed_to_motor(v_r)

    # --- Time stamps ---
    dt = T / n_steps
    commands = []

    for i in range(n_steps):
        t_start = i * dt
        t_end   = (i + 1) * dt
        commands.append((t_start, t_end, left_motors[i], right_motors[i]))

    return commands, xs, ys, P0, P3


# ============================================================
# PLOT: TRAJECTORY COMPARISON
# ============================================================

def plot_trajectories(alpha=ALPHA):

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, angle in zip(axes, TEST_ANGLES):

        # Type A
        cmds_a, goal = simulate_type_a(angle)
        ax.scatter(*goal, s=120, marker="x", color="red",
                   zorder=5, label="Type A goal")

        # Draw Type A path (rotate in place then straight line)
        theta = np.deg2rad(angle)
        ax.annotate(
            "", xy=goal,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red",
                            lw=1.5, linestyle="dashed")
        )

        # Type B
        cmds_b, xs, ys, start, end = simulate_type_b(angle, alpha=alpha)
        ax.plot(xs, ys, linewidth=2.5, color="royalblue", label="Type B path")
        ax.scatter(*start, s=80, color="green", zorder=5, label="Start")
        ax.scatter(*end,   s=80, color="royalblue", zorder=5,
                   marker="o", label="Type B end")

        # Direction arrows along curve
        for i in [50, 150, 250]:
            if i + 1 < len(xs):
                dx = xs[i + 1] - xs[i]
                dy = ys[i + 1] - ys[i]
                ax.arrow(xs[i], ys[i], dx, dy,
                         head_width=0.001, length_includes_head=True,
                         color="royalblue")

        ax.set_title(f"{angle}°  |  T = {total_time(angle):.2f}s")
        ax.grid(True)
        ax.axis("equal")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(
        f"Type A vs Type B Trajectories  (alpha={alpha})",
        fontsize=15
    )
    plt.tight_layout()
    plt.show()


# ============================================================
# PLOT: MOTOR COMMANDS OVER TIME
# ============================================================

def plot_motor_commands(alpha=ALPHA):

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, angle in zip(axes, TEST_ANGLES):

        T = total_time(angle)

        # --- Type A commands ---
        cmds_a, _ = simulate_type_a(angle)
        for (ts, te, l, r) in cmds_a:
            ax.hlines(l, ts, te, colors="red",    linewidth=2.5,
                      linestyles="--", label="A left"  if ts == 0 else "")
            ax.hlines(r, ts, te, colors="salmon",  linewidth=2.5,
                      linestyles="--", label="A right" if ts == 0 else "")

        # --- Type B commands ---
        cmds_b, xs, ys, _, _ = simulate_type_b(angle, alpha=alpha)
        t_mids  = [(ts + te) / 2 for ts, te, l, r in cmds_b]
        lefts   = [l for ts, te, l, r in cmds_b]
        rights  = [r for ts, te, l, r in cmds_b]

        ax.plot(t_mids, lefts,  color="royalblue",  linewidth=2,
                label="B left")
        ax.plot(t_mids, rights, color="dodgerblue", linewidth=2,
                linestyle=":", label="B right")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline( BASE_MOTOR_SPEED, color="gray", linewidth=0.8,
                    linestyle=":", alpha=0.6)
        ax.axhline(-BASE_MOTOR_SPEED, color="gray", linewidth=0.8,
                    linestyle=":", alpha=0.6)

        ax.set_xlim(0, T)
        ax.set_ylim(MOTOR_MIN - 20, MOTOR_MAX + 20)
        ax.set_title(f"{angle}°  |  T = {T:.2f}s")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Motor speed")
        ax.grid(True, alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate legend
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc="upper right")

    fig.suptitle(
        f"Motor Commands Over Time  (alpha={alpha})",
        fontsize=15
    )
    plt.tight_layout()
    plt.show()


# ============================================================
# PRINT COMMAND TABLE (for debugging / upload to robot)
# ============================================================

def print_commands(theta_deg, alpha=ALPHA):

    print(f"\n{'='*60}")
    print(f"Angle: {theta_deg}°   Total time: {total_time(theta_deg):.3f}s")
    print(f"{'='*60}")

    print("\n--- Type A ---")
    cmds_a, goal = simulate_type_a(theta_deg)
    print(f"{'t_start':>8}  {'t_end':>6}  {'left':>6}  {'right':>6}")
    for ts, te, l, r in cmds_a:
        print(f"{ts:8.3f}  {te:6.3f}  {l:6.0f}  {r:6.0f}")

    print("\n--- Type B ---")
    cmds_b, *_ = simulate_type_b(theta_deg, alpha=alpha)
    print(f"{'t_start':>8}  {'t_end':>6}  {'left':>6}  {'right':>6}")
    for ts, te, l, r in cmds_b:
        print(f"{ts:8.3f}  {te:6.3f}  {l:6.1f}  {r:6.1f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Trajectory comparison
    plot_trajectories(alpha=ALPHA)

    # Motor speed profiles over time
    plot_motor_commands(alpha=ALPHA)

    # Print command table for one angle
    print_commands(theta_deg=45, alpha=ALPHA)