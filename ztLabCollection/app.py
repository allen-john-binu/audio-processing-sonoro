from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
from scipy import signal
import os

app = Flask(__name__, static_folder="static")

# ── Config ─────────────────────────────────────────────────────────────────────
REAL_BASE   = "dataFromReal"
QUALIS_BASE = "dataFromQualis"

QUALIS_COLS = ["timestamp", "robot_x", "robot_y", "speakerL_x", "speakerL_y", "speakerR_x", "speakerR_y"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_and_merge(name):
    """Load Real + Qualis CSVs and merge on timestamp."""
    real_path   = os.path.join(REAL_BASE,   name)
    qualis_path = os.path.join(QUALIS_BASE, name)

    real   = pd.read_csv(real_path)
    qualis = pd.read_csv(qualis_path)

    real["timestamp"]   = pd.to_datetime(real["timestamp"])
    qualis["timestamp"] = pd.to_datetime(qualis["timestamp"])

    real_sorted   = real.sort_values("timestamp").reset_index(drop=True)
    qualis_sorted = qualis.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        real_sorted,
        qualis_sorted[QUALIS_COLS].rename(columns={"timestamp": "timestamp_qualis"}),
        left_on="timestamp",
        right_on="timestamp_qualis",
        direction="nearest"
    ).drop(columns=["timestamp_qualis"])

    return merged


def compute_peaks(intensity_matrix, angles):
    """Compute top-2 peak angles and intensity difference per row."""
    peak_angles_1  = []
    peak_angles_2  = []
    intensity_diffs = []

    for row in intensity_matrix:
        peaks, _ = signal.find_peaks(row)

        if len(peaks) >= 2:
            top2 = peaks[np.argsort(row[peaks])[-2:]]
            p1, p2 = top2
            peak_angles_1.append(float(angles[p1]))
            peak_angles_2.append(float(angles[p2]))
            intensity_diffs.append(float(abs(row[p1] - row[p2])))
        elif len(peaks) == 1:
            peak_angles_1.append(float(angles[peaks[0]]))
            peak_angles_2.append(None)
            intensity_diffs.append(None)
        else:
            peak_angles_1.append(None)
            peak_angles_2.append(None)
            intensity_diffs.append(None)

    return peak_angles_1, peak_angles_2, intensity_diffs


def build_payload(merged):
    """Build the full JSON payload from a merged dataframe."""
    # Angle columns are all columns after timestamp and dB_SPL
    angle_cols = [c for c in merged.columns if c not in
                  ["timestamp", "dB_SPL", "robot_x", "robot_y",
                   "speakerL_x", "speakerL_y", "speakerR_x", "speakerR_y"]]

    angles = np.array([float(a) for a in angle_cols])

    # Parse dB_SPL — stored as "[69.001]" strings
    spl_array = (
        merged["dB_SPL"]
        .astype(str)
        .str.strip("[]")
        .astype(float)
        .values
    )

    intensity_matrix = merged[angle_cols].values.astype(float)

    # Elapsed seconds from first timestamp
    t0      = merged["timestamp"].iloc[0]
    elapsed = (merged["timestamp"] - t0).dt.total_seconds().values

    # Trajectory columns
    traj = {
        "robot_x":    merged["robot_x"].tolist(),
        "robot_y":    merged["robot_y"].tolist(),
        "speakerL_x": merged["speakerL_x"].tolist(),
        "speakerL_y": merged["speakerL_y"].tolist(),
        "speakerR_x": merged["speakerR_x"].tolist(),
        "speakerR_y": merged["speakerR_y"].tolist(),
    }

    peak1, peak2, diffs = compute_peaks(intensity_matrix, angles)

    # Scale to raw min/max for display
    zmin = float(intensity_matrix.min())
    zmax = float(intensity_matrix.max())

    traj_angle_L, traj_angle_R = compute_trajectory_angles(traj)

    return {
        "angles":           angles.tolist(),
        "elapsed":          elapsed.tolist(),
        "spl":              spl_array.tolist(),
        "spl_min":          float(spl_array.min()),
        "spl_max":          float(spl_array.max()),
        "intensity_matrix": intensity_matrix.tolist(),
        "peak1":            peak1,
        "peak2":            peak2,
        "diffs":            diffs,
        "trajectory":       traj,
        "traj_angle_L":     traj_angle_L,
        "traj_angle_R":     traj_angle_R,
        "zmin":             zmin,
        "zmax":             zmax,
    }


def compute_trajectory_angles(traj):
    """
    For each timestep compute the angle (degrees) at the robot position between:
      - (robot -> speakerL) and (robot -> robot_next)
      - (robot -> speakerR) and (robot -> robot_next)
    Last point is None — no robot_next exists.
    """
    rx = np.array(traj["robot_x"],    dtype=float)
    ry = np.array(traj["robot_y"],    dtype=float)
    lx = np.array(traj["speakerL_x"], dtype=float)
    ly = np.array(traj["speakerL_y"], dtype=float)
    sx = np.array(traj["speakerR_x"], dtype=float)
    sy = np.array(traj["speakerR_y"], dtype=float)

    n = len(rx)
    angle_L = [None] * n
    angle_R = [None] * n

    for i in range(n - 1):
        dx, dy = rx[i+1] - rx[i], ry[i+1] - ry[i]
        move_norm = np.hypot(dx, dy)
        if move_norm < 1e-12:
            continue
        move = np.array([dx / move_norm, dy / move_norm])

        vl = np.array([lx[i] - rx[i], ly[i] - ry[i]])
        nl = np.hypot(*vl)
        if nl > 1e-12:
            angle_L[i] = float(np.degrees(np.arccos(np.clip(np.dot(move, vl / nl), -1.0, 1.0))))

        vr = np.array([sx[i] - rx[i], sy[i] - ry[i]])
        nr = np.hypot(*vr)
        if nr > 1e-12:
            angle_R[i] = float(np.degrees(np.arccos(np.clip(np.dot(move, vr / nr), -1.0, 1.0))))

    return angle_L, angle_R


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/runs")
def get_runs():
    if not os.path.isdir(REAL_BASE):
        return jsonify({"runs": [], "error": f"Folder not found: {REAL_BASE}"})
    runs = sorted(f for f in os.listdir(REAL_BASE) if f.endswith(".csv"))
    return jsonify({"runs": runs})


@app.route("/api/load/<name>")
def load_run(name):
    try:
        merged  = load_and_merge(name)
        payload = build_payload(merged)
        return jsonify({"ok": True, "data": payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/filter", methods=["POST"])
def filter_run():
    """Re-filter an already-loaded dataset by SPL threshold."""
    body      = request.json
    threshold = float(body["threshold"])
    spl       = np.array(body["spl"])
    elapsed   = np.array(body["elapsed"])
    intensity = np.array(body["intensity_matrix"])
    angles    = np.array(body["angles"])
    traj      = body["trajectory"]

    mask = spl >= threshold

    f_elapsed   = elapsed[mask].tolist()
    f_intensity = intensity[mask]   # still raw values from client
    f_spl       = spl[mask].tolist()

    # Filter trajectory by same mask
    f_traj = {k: [v for v, m in zip(traj[k], mask) if m] for k in traj}

    peak1, peak2, diffs = compute_peaks(f_intensity, angles)

    f_traj_angle_L, f_traj_angle_R = compute_trajectory_angles(f_traj)

    return jsonify({
        "ok":               True,
        "elapsed":          f_elapsed,
        "spl":              f_spl,
        "intensity_matrix": f_intensity.tolist(),
        "peak1":            peak1,
        "peak2":            peak2,
        "diffs":            diffs,
        "trajectory":       f_traj,
        "traj_angle_L":     f_traj_angle_L,
        "traj_angle_R":     f_traj_angle_R,
        # zmin/zmax locked to full-run range — colorscale stays stable while sliding
        "zmin":             body["zmin"],
        "zmax":             body["zmax"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)