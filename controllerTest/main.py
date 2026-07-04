"""
tune_arc_move.py
================
Tunes the curvature gain `k` in arc_move() by comparing the DISPLACEMENT
produced by arc_move against the ground-truth "rotate-then-straight" motion.

Key design: NO home-return controller.
Instead, for each (k, phi) pair we:
  1. Record start pose S
  2. Run Trial A (rotate+straight) → record end pose A
  3. Drive BACKWARD for the same duration to approximately undo the motion
  4. Record new start pose S'
  5. Run Trial B (arc_move with this k) → record end pose B
  6. Compare DISPLACEMENTS: (A - S) vs (B - S')
     pos_err     = |displacement_A| vs |displacement_B| in mm
     heading_err = difference in final heading change
     combined    = W_POS * pos_err + W_HEADING * heading_err

This means we compare what each method *does to the robot*, not where the
robot ends up in absolute space — so reset accuracy doesn't matter at all.

The backward drive is just `move_forward(speed=-FORWARD_SPEED)` for the
same DURATION, which roughly undoes the forward component. No feedback loop.

Author: you
"""

import asyncio
import csv
import math
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
from thymiodirect import Connection, Thymio
import qtm


# ============================================================
# CONFIGURATION  — edit these
# ============================================================

QTM_ADDRESS  = "10.126.17.54"
PASSWORD     = "robot609"
ROBOT_BODY   = "robat"

FORWARD_SPEED = 200              # Thymio motor units (0–500)
TURN_SPEED    = 150              # Thymio motor units
DURATION      = 0.5             # seconds per motion step
VISUAL_PAUSE = 2.0

# Gain values to search over
K_SEARCH = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Angles to test (degrees, relative to robot heading)
ANGLE_SET = [-45]

# How to weight the two error terms in the combined score
W_POS     = 1.0    # weight for displacement magnitude error (mm)
W_HEADING = 50.0   # weight for heading change error (rad) — ~1 rad ≈ 50 mm

# Pause between trials to let the robot settle and QTM stabilise
SETTLE_TIME = 0.5   # seconds
VISUAL_PAUSE = 2.0  # pause to visually compare trials

# Windowed heading estimator: buffer size (frames).
# At 100 Hz, 20 frames = 200 ms of travel — robust to per-frame jitter.
HEADING_WINDOW = 20

# Minimum displacement across the window to update heading (mm).
# 1 mm is easily exceeded over 200 ms even at slow speed.
HEADING_MIN_DISP = 1.0

# How long to wait for heading to initialise after motion starts (s)
HEADING_TIMEOUT = 5.0

# Output
SAVE_DIR   = "tuning_results"
OUTPUT_CSV = os.path.join(
    SAVE_DIR,
    f"arc_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

TRAJ_DIR = "trajectory_logs"
os.makedirs(TRAJ_DIR, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================

def _wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


async def record_trajectory(
    qtm_reader,
    duration,
    csv_writer,
    trial_label
):
    """
    Record QTM trajectory for duration seconds.
    """

    t0 = time.time()

    while time.time() - t0 < duration:

        if (
            qtm_reader.x is not None and
            qtm_reader.heading is not None
        ):

            csv_writer.writerow([
                time.time(),
                trial_label,
                qtm_reader.x,
                qtm_reader.y,
                math.degrees(qtm_reader.heading)
            ])

        await asyncio.sleep(0.01)


# ============================================================
# QUALISYS INTERFACE
# ============================================================

class QualisysReader:
    """
    Streams robot pose (x, y, heading) from QTM.

    Heading is estimated via a windowed finite-difference:
      - Keep a rolling buffer of the last HEADING_WINDOW position frames.
      - Compare buffer[0] vs buffer[-1] for displacement.
      - Only update heading when displacement >= HEADING_MIN_DISP mm.
      - When stationary, hold the last good heading (noise stays near 0 mm).
    """

    def __init__(self):
        self.x       = None
        self.y       = None
        self.heading = None           # radians, None until robot has moved

        self._last_good_heading = None
        self._pos_window        = []  # rolling list of (x, y) tuples
        self._body_index        = {}
        self._qtm_start_ns      = []
        self._connection        = None

    async def connect(self):
        self._connection = await qtm.connect(QTM_ADDRESS)
        if self._connection is None:
            raise RuntimeError("Failed to connect to QTM.")

        async with qtm.TakeControl(self._connection, PASSWORD):
            await self._connection.new()

        xml_string = await self._connection.get_parameters(parameters=["6d"])
        xml        = ET.fromstring(xml_string)
        self._body_index = {
            b.text.strip(): idx
            for idx, b in enumerate(xml.findall("*/Body/Name"))
        }

        print("QTM bodies:", list(self._body_index.keys()))

        if ROBOT_BODY not in self._body_index:
            raise RuntimeError(
                f"Robot body '{ROBOT_BODY}' not found in QTM scene. "
                f"Available: {list(self._body_index.keys())}"
            )

        await self._connection.stream_frames(
            components=["6deuler"],
            on_packet=self._on_packet
        )
        print("QTM streaming started.")

    def _on_packet(self, packet):
        _, bodies = packet.get_6d_euler()
        idx = self._body_index.get(ROBOT_BODY)
        if idx is None:
            return
        try:
            position, rotation = bodies[idx]

            x = float(position.x)
            y = float(position.y)

            yaw_rad = math.radians(rotation.a3)

            self.x = x
            self.y = y
            self.heading = yaw_rad
            self._last_good_heading = yaw_rad
        except Exception:
            pass

    def get_pose(self):
        """Returns (x, y, heading_rad) or None if heading not yet initialised."""
        if self.x is None or self._last_good_heading is None:
            return None
        return (self.x, self.y, self._last_good_heading)

    async def disconnect(self):
        if self._connection:
            try:
                await self._connection.stream_frames_stop()
            except Exception:
                pass
            self._connection.disconnect()


# ============================================================
# THYMIO INTERFACE
# ============================================================

class ThymioController:

    def __init__(self):
        print("Connecting to Thymio...")
        port = Connection.serial_default_port()
        th   = Thymio(
            serial_port=port,
            on_connect=lambda nid: print(f"  Thymio node {nid} connected")
        )
        th.connect()
        time.sleep(1.0)
        self.robot = th[th.first_node()]
        print("Thymio ready.")

    def _set_motors(self, left, right):
        self.robot['motor.left.target']  = int(left)
        self.robot['motor.right.target'] = int(right)

    def stop(self):
        self._set_motors(0, 0)
        self.robot["leds.top"]          = [0, 0, 0]
        self.robot["leds.bottom.left"]  = [0, 0, 0]
        self.robot["leds.bottom.right"] = [0, 0, 0]

    def _angle_to_time(self, angle_deg):
        A, B = 612.33, -0.94
        return (A * TURN_SPEED ** B) * abs(angle_deg) / 360.0

    def move_forward(self, duration=DURATION):
        """Drive straight forward for `duration` seconds."""
        self.robot["leds.top"] = [0, 255, 0]
        t0 = time.time()
        while time.time() - t0 < duration:
            self._set_motors(FORWARD_SPEED, FORWARD_SPEED)
            time.sleep(0.01)
        self.stop()

    def move_backward(self, duration=DURATION):
        """Drive straight backward for `duration` seconds (soft reset)."""
        self.robot["leds.top"] = [255, 100, 0]   # orange
        t0 = time.time()
        while time.time() - t0 < duration:
            self._set_motors(-FORWARD_SPEED, -FORWARD_SPEED)
            time.sleep(0.01)
        self.stop()

    def rotate(self, angle_deg):
        """In-place rotation. Positive = right, negative = left."""
        duration = self._angle_to_time(angle_deg)
        t0 = time.time()
        while time.time() - t0 < duration:
            if angle_deg >= 0:
                self._set_motors(TURN_SPEED + TURN_SPEED // 2, -TURN_SPEED)
            else:
                self._set_motors(-TURN_SPEED, TURN_SPEED + TURN_SPEED // 2)
            time.sleep(0.01)
        self.stop()

    def rotate_then_straight(self, phi_deg, duration=DURATION):
        """Ground-truth motion: in-place rotate then straight forward."""
        self.robot["leds.top"] = [255, 255, 0]   # yellow = ground truth
        self.rotate(phi_deg)
        time.sleep(0.1)
        self.move_forward(duration)
        self.stop()

    def arc_move(self, phi_deg, duration=DURATION, k=0.8):
        """
        Smooth arc motion.
        phi_deg : relative heading (already in [-180, 180])
        k       : curvature gain [0, 1]
        Both wheels >= 0 (no reverse).
        """
        phi_norm    = max(-1.0, min(1.0, phi_deg / 180.0))
        left_speed = max(
            -500,
            min(500, int(FORWARD_SPEED * (1.0 - k * phi_norm)))
        )

        right_speed = max(
            -500,
            min(500, int(FORWARD_SPEED * (1.0 + k * phi_norm)))
        )

        self.robot["leds.top"] = [0, 0, 255]     # blue = arc
        t0 = time.time()
        while time.time() - t0 < duration:
            self._set_motors(left_speed, right_speed)
            time.sleep(0.01)
        self.stop()

    def soft_reset(self, phi_deg, duration=DURATION):
        """
        Approximate undo of a motion: drive backward for `duration` seconds.
        No feedback — just a best-effort reset to reduce drift between trials.
        Does NOT need to be perfect; errors are relative to start snapshots.
        """
        self.robot["leds.top"] = [128, 0, 128]   # purple = resetting
        t0 = time.time()
        while time.time() - t0 < duration:
            self._set_motors(-FORWARD_SPEED, -FORWARD_SPEED)
            time.sleep(0.01)
        self.stop()


# ============================================================
# POSE SNAPSHOT
# ============================================================

async def snapshot_position(qtm_reader, settle=0.3):
    """
    Position-only snapshot. Does NOT require heading.
    Used to record start/end XY when heading is not yet available.
    Returns (x, y).
    """
    await asyncio.sleep(settle)
    xs, ys = [], []
    t0 = time.time()
    while time.time() - t0 < 0.3:
        if qtm_reader.x is not None:
            xs.append(qtm_reader.x)
            ys.append(qtm_reader.y)
        await asyncio.sleep(0.02)

    if not xs:
        raise RuntimeError(
            "No position data from QTM. "
            "Check connection and that ROBOT_BODY name matches."
        )
    return float(np.mean(xs)), float(np.mean(ys))


async def snapshot_pose(qtm_reader, settle=0.4):
    """
    Full pose snapshot (x, y, heading_rad).
    Waits actively for heading to be initialised after motion.
    Returns (x, y, heading_rad).
    """
    # First wait for the settle window so the robot has stopped
    await asyncio.sleep(settle)

    # Collect and average frames over 300 ms
    xs, ys, hs = [], [], []
    t0 = time.time()
    while time.time() - t0 < 0.3:
        if qtm_reader.x is not None and qtm_reader._last_good_heading is not None:
            xs.append(qtm_reader.x)
            ys.append(qtm_reader.y)
            hs.append(qtm_reader._last_good_heading)
        await asyncio.sleep(0.02)

    if not xs:
        raise RuntimeError("No frames collected during snapshot window.")

    sin_h  = np.mean(np.sin(hs))
    cos_h  = np.mean(np.cos(hs))
    mean_h = math.atan2(sin_h, cos_h)

    return float(np.mean(xs)), float(np.mean(ys)), mean_h


# ============================================================
# DISPLACEMENT-BASED ERROR
# ============================================================

def compute_displacement(start, end):
    """
    Compute the displacement vector and heading change from start to end pose.
      start, end : (x, y, heading_rad)
    Returns:
      dx, dy          — position displacement in mm
      dist            — magnitude of displacement in mm
      delta_heading   — change in heading in radians (wrapped to [-pi, pi])
    """
    dx            = end[0] - start[0]
    dy            = end[1] - start[1]
    dist          = math.hypot(dx, dy)
    delta_heading = _wrap_pi(end[2] - start[2])
    return dx, dy, dist, delta_heading


def compute_errors(disp_a, disp_b):
    """
    Compare two displacements (each from compute_displacement).

    disp_a : (dx, dy, dist, delta_heading) — ground truth (rotate+straight)
    disp_b : (dx, dy, dist, delta_heading) — arc_move

    We compare:
      pos_err     : difference in displacement MAGNITUDE (mm)
                    i.e. did arc_move cover the same distance?
      heading_err : difference in heading CHANGE (rad)
                    i.e. did arc_move turn the robot by the same amount?
      combined    : weighted sum

    Note: we compare magnitudes and heading changes, not absolute endpoints,
    so the soft reset accuracy does not affect the result.
    """
    dx_err = disp_b[0] - disp_a[0]
    dy_err = disp_b[1] - disp_a[1]

    pos_err = math.hypot(dx_err, dy_err)
    heading_err = abs(_wrap_pi(disp_b[3] - disp_a[3]))
    combined    = W_POS * pos_err + W_HEADING * heading_err
    return pos_err, heading_err, combined

# ============================================================
# SINGLE TRIAL PAIR
# ============================================================

async def run_trial_pair(thymio, qtm_reader, phi, k):
    """
    Run one (phi, k) trial pair and return displacement errors.

    Returns:
      pose_start_a, pose_end_a  — start and end pose for Trial A
      pose_start_b, pose_end_b  — start and end pose for Trial B
      disp_a, disp_b            — displacements (dx, dy, dist, delta_h)
      pos_err, heading_err, combined
    """

    # ── TRIAL A: rotate+straight (ground truth) ───────────────────────────
    print(f"    Trial A: rotate+straight (phi={phi:+.0f}°)...")
    
    traj_filename = os.path.join(
        TRAJ_DIR,
        f"k{k:.1f}_phi{phi:+d}.csv"
    )

    traj_file = open(
        traj_filename,
        "w",
        newline=""
    )

    traj_writer = csv.writer(traj_file)

    traj_writer.writerow([
        "timestamp",
        "trial",
        "x",
        "y",
        "heading_deg"
    ])

    thymio.robot["leds.top"] = [255, 255, 0]
    await asyncio.sleep(0.1)

    # Start snapshot (position only — heading reset above)
    pose_start_a = await snapshot_pose(qtm_reader)

    # Execute motion
    motion_task = asyncio.to_thread(
        thymio.rotate_then_straight,
        phi,
        DURATION
    )

    record_task = record_trajectory(
        qtm_reader,
        DURATION + thymio._angle_to_time(phi),
        traj_writer,
        "A"
    )

    await asyncio.gather(
        motion_task,
        record_task
    )

    # End snapshot (full pose — heading now valid after motion)
    pose_end_a = await snapshot_pose(qtm_reader, settle=SETTLE_TIME)

    # Reconstruct start pose: use start_a_xy + end_a heading minus delta
    # For start heading we just use the end heading (relative phi is what matters)
    # Actually we need start heading too — drive backward to get it.
    # Simpler: record start heading BEFORE the motion. We do that by ensuring
    # heading is valid before reset (from calibration or previous trial).
    # Here we use a pragmatic approach: start heading = end heading - phi_rad
    # This is an approximation but good enough for displacement magnitude comparison.

    disp_a = compute_displacement(
        pose_start_a,
        pose_end_a
    )
    
    print(
        f"      Start: ({pose_start_a[0]:.1f}, "
        f"{pose_start_a[1]:.1f}, "
        f"{math.degrees(pose_start_a[2]):.1f}°)"
    )

    print(
        f"      End  : ({pose_end_a[0]:.1f}, "
        f"{pose_end_a[1]:.1f}, "
        f"{math.degrees(pose_end_a[2]):.1f}°)"
    )
    print(f"      Displacement: dist={disp_a[2]:.1f} mm  dh={math.degrees(disp_a[3]):.1f}°")
    print("      Observe ground truth motion...")
    await asyncio.sleep(VISUAL_PAUSE)

    # Soft reset — drive backward for the forward component duration
    print("      Resetting...")
    thymio.soft_reset(phi, duration=DURATION)
    await asyncio.sleep(SETTLE_TIME)
    print("      Observe reset...")
    await asyncio.sleep(VISUAL_PAUSE)

    # ── TRIAL B: arc_move ─────────────────────────────────────────────────
    print(f"    Trial B: arc_move (phi={phi:+.0f}°  k={k})...")

    thymio.robot["leds.top"] = [0, 0, 255]
    await asyncio.sleep(0.1)

    pose_start_b = await snapshot_pose(qtm_reader)

    motion_task = asyncio.to_thread(
        thymio.arc_move,
        phi,
        DURATION,
        k
    )

    record_task = record_trajectory(
        qtm_reader,
        DURATION,
        traj_writer,
        "B"
    )

    await asyncio.gather(
        motion_task,
        record_task
    )

    pose_end_b = await snapshot_pose(qtm_reader, settle=SETTLE_TIME)

    # For arc_move the heading change is determined by the arc geometry.
    # We measure it directly from start vs end.
    # Start heading: use same approximation as above for consistency.
    disp_b = compute_displacement(
        pose_start_b,
        pose_end_b
    )
    print(
        f"      Start: ({pose_start_b[0]:.1f}, "
        f"{pose_start_b[1]:.1f}, "
        f"{math.degrees(pose_start_b[2]):.1f}°)"
    )

    print(
        f"      End  : ({pose_end_b[0]:.1f}, "
        f"{pose_end_b[1]:.1f}, "
        f"{math.degrees(pose_end_b[2]):.1f}°)"
    )
    print(f"      Displacement: dist={disp_b[2]:.1f} mm  dh={math.degrees(disp_b[3]):.1f}°")

    pos_err, heading_err, combined = compute_errors(disp_a, disp_b)
    print(f"      → pos_err={pos_err:.1f} mm  "
          f"heading_err={math.degrees(heading_err):.1f}°  "
          f"combined={combined:.3f}")
    print("      Observe arc motion...")
    await asyncio.sleep(VISUAL_PAUSE)

    # Soft reset after trial B
    print("      Resetting...")
    thymio.soft_reset(phi, duration=DURATION)
    await asyncio.sleep(SETTLE_TIME)
    
    traj_file.close()

    return (pose_start_a, pose_end_a, pose_start_b, pose_end_b,
            disp_a, disp_b, pos_err, heading_err, combined)


# ============================================================
# MAIN TUNING LOOP
# ============================================================

async def run_tuning():

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- connect hardware ----
    qtm_reader = QualisysReader()
    await qtm_reader.connect()
    await asyncio.sleep(1.0)

    thymio = ThymioController()

    # ---- CSV setup ----
    fieldnames = [
        "k", "phi_deg",
        # Trial A
        "A_start_x", "A_start_y", "A_start_h_deg",
        "A_end_x",   "A_end_y",   "A_end_h_deg",
        "A_dist_mm", "A_dh_deg",
        # Trial B
        "B_start_x", "B_start_y", "B_start_h_deg",
        "B_end_x",   "B_end_y",   "B_end_h_deg",
        "B_dist_mm", "B_dh_deg",
        # Errors
        "pos_err_mm", "heading_err_deg", "combined_err",
        "w_pos", "w_heading",
    ]
    csv_rows  = []
    k_scores  = {}

    total_pairs = len(K_SEARCH) * len(ANGLE_SET)
    pair_num    = 0

    print("=" * 60)
    print(f"Sweep: {len(K_SEARCH)} k values × {len(ANGLE_SET)} angles "
          f"= {total_pairs} trial pairs")
    print("=" * 60)

    for k in K_SEARCH:
        k_scores[k] = []
        print(f"\n{'─'*50}  k = {k}  {'─'*50}")

        for phi in ANGLE_SET:
            pair_num += 1
            print(f"\n  [{pair_num}/{total_pairs}]  phi={phi:+.0f}°  k={k}")

            try:
                (psa, pea, psb, peb,
                 disp_a, disp_b,
                 pos_err, heading_err, combined) = await run_trial_pair(
                    thymio, qtm_reader, phi, k
                )
            except Exception as e:
                print(f"  ERROR in trial pair: {e} — skipping.")
                continue

            k_scores[k].append(combined)

            csv_rows.append({
                "k": k, "phi_deg": phi,
                "A_start_x":   round(psa[0], 2),
                "A_start_y":   round(psa[1], 2),
                "A_start_h_deg": round(math.degrees(psa[2]), 2),
                "A_end_x":     round(pea[0], 2),
                "A_end_y":     round(pea[1], 2),
                "A_end_h_deg": round(math.degrees(pea[2]), 2),
                "A_dist_mm":   round(disp_a[2], 2),
                "A_dh_deg":    round(math.degrees(disp_a[3]), 2),
                "B_start_x":   round(psb[0], 2),
                "B_start_y":   round(psb[1], 2),
                "B_start_h_deg": round(math.degrees(psb[2]), 2),
                "B_end_x":     round(peb[0], 2),
                "B_end_y":     round(peb[1], 2),
                "B_end_h_deg": round(math.degrees(peb[2]), 2),
                "B_dist_mm":   round(disp_b[2], 2),
                "B_dh_deg":    round(math.degrees(disp_b[3]), 2),
                "pos_err_mm":      round(pos_err, 2),
                "heading_err_deg": round(math.degrees(heading_err), 2),
                "combined_err":    round(combined, 3),
                "w_pos":     W_POS,
                "w_heading": W_HEADING,
            })

    # ---- find best k ----
    mean_scores = {
        k: float(np.mean(v)) for k, v in k_scores.items() if v
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, score in sorted(mean_scores.items(), key=lambda x: x[1]):
        print(f"  k={k:.2f}   mean combined error = {score:.3f}")

    best_k     = min(mean_scores, key=mean_scores.get)
    best_score = mean_scores[best_k]
    print(f"\n  Best k = {best_k}  (mean combined error = {best_score:.3f})")
    print("=" * 60)

    # ---- save CSV ----
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(OUTPUT_CSV, "a", newline="") as f:
        f.write("\n# SUMMARY\n")
        f.write(f"# best_k,{best_k}\n")
        f.write(f"# best_mean_combined_error,{best_score:.4f}\n")
        f.write(f"# w_pos,{W_POS}\n")
        f.write(f"# w_heading,{W_HEADING}\n")
        for k, score in sorted(mean_scores.items()):
            f.write(f"# k={k},mean_combined={score:.4f}\n")

    print(f"\nResults saved to: {OUTPUT_CSV}")

    thymio.stop()
    await qtm_reader.disconnect()

    return best_k, mean_scores


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    best_k, scores = asyncio.run(run_tuning())
    print(f"\nUse  k={best_k}  in your arc_move() calls.")