#!/usr/bin/env python3
"""
bump_viewer/app.py
──────────────────
Interactive bump-angle + bump-statistics visualization server.
Two stacked Plotly charts, one dB_SPL slider controls both.

Usage:
    python3 app.py --data-dir ./oldResult
    python3 app.py --data-dir /path/to/data --port 5000
"""

import argparse
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
DATA_DIR: pathlib.Path = None  # set at startup


# ─────────────────────────────────────────────────────────────────────────────
# Directory scanning
# ─────────────────────────────────────────────────────────────────────────────

def scan_experiments() -> dict:
    result = {}
    if DATA_DIR is None or not DATA_DIR.is_dir():
        return result
    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        csvs = sorted(p.name for p in folder.glob("*_bumps.csv"))
        if csvs:
            result[folder.name] = csvs
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Shared data helpers
# ─────────────────────────────────────────────────────────────────────────────

def bump_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if c.startswith("bump") and c[4:].isdigit()]
    cols.sort(key=lambda c: int(c[4:]))
    return cols


def build_heatmap(df: pd.DataFrame, b_cols: list) -> np.ndarray:
    """Returns ndarray shape (n_timesteps, n_angle_bins=36)."""
    bin_edges = np.arange(-180, 181, 10)
    n_bins = len(bin_edges) - 1
    heatmap = np.zeros((len(df), n_bins))
    for i, (_, row) in enumerate(df[b_cols].iterrows()):
        counts, _ = np.histogram(row.dropna().values, bins=bin_edges)
        heatmap[i] = counts
    return heatmap


def compute_stats(df: pd.DataFrame, b_cols: list) -> dict:
    """
    Per-row statistics over bump columns:
      mean_all   — mean of all bump values
      pos_sd     — std of positive bump values
      neg_sd     — std of negative bump values
      count_metric — (#positive - #negative)
    """
    n = len(df)
    mean_all     = np.full(n, np.nan)
    pos_sd       = np.full(n, np.nan)
    neg_sd       = np.full(n, np.nan)
    count_metric = np.full(n, np.nan)
    bump_vals    = df[b_cols].values

    for i in range(n):
        row = bump_vals[i]
        row = row[~np.isnan(row)]
        if len(row) == 0:
            continue
        mean_all[i] = np.mean(row)
        pos = row[row > 0]
        neg = row[row < 0]
        if len(pos) > 0:
            pos_sd[i] = np.std(pos, ddof=0)
        if len(neg) > 0:
            neg_sd[i] = np.std(neg, ddof=0)
        count_metric[i] = len(pos) - len(neg)

    return dict(
        mean_all=mean_all,
        pos_sd=pos_sd,
        neg_sd=neg_sd,
        count_metric=count_metric,
    )


def _nan_to_none(arr: np.ndarray) -> list:
    """Convert numpy array to list, replacing NaN with None for JSON."""
    return [None if np.isnan(v) else float(v) for v in arr]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/experiments")
def api_experiments():
    try:
        return jsonify({"ok": True, "experiments": scan_experiments()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/load")
def api_load():
    """Return db range for slider setup."""
    experiment = request.args.get("experiment", "")
    run        = request.args.get("run", "")
    if not experiment or not run:
        return jsonify({"ok": False, "error": "experiment and run required"})

    csv_path = DATA_DIR / experiment / run
    try:
        df = pd.read_csv(csv_path)
        required = {"dB_SPL", "angle", "left_volume", "right_volume"}
        missing  = required - set(df.columns)
        if missing:
            return jsonify({"ok": False, "error": f"Missing columns: {missing}"})
        b_cols = bump_columns(df)
        if not b_cols:
            return jsonify({"ok": False, "error": "No bump columns found in CSV"})
        return jsonify({
            "ok": True,
            "db_min":  float(df["dB_SPL"].min()),
            "db_max":  float(df["dB_SPL"].max()),
            "n_total": len(df),
            "n_bumps": len(b_cols),
        })
    except FileNotFoundError:
        return jsonify({"ok": False, "error": f"File not found: {csv_path}"})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()})


@app.route("/api/plot")
def api_plot():
    """Bump-angle view: volume + heatmap + angle panels."""
    experiment = request.args.get("experiment", "")
    run        = request.args.get("run", "")
    threshold  = request.args.get("threshold", type=float, default=None)
    if not experiment or not run:
        return jsonify({"ok": False, "error": "experiment and run required"})

    csv_path = DATA_DIR / experiment / run
    try:
        df = pd.read_csv(csv_path)
        required = {"dB_SPL", "angle", "left_volume", "right_volume"}
        missing  = required - set(df.columns)
        if missing:
            return jsonify({"ok": False, "error": f"Missing columns: {missing}"})

        b_cols = bump_columns(df)
        if not b_cols:
            return jsonify({"ok": False, "error": "No bump columns found"})

        db_min = float(df["dB_SPL"].min())
        db_max = float(df["dB_SPL"].max())
        if threshold is None:
            threshold = db_min

        df_f = df[df["dB_SPL"] >= threshold].reset_index(drop=True)
        n    = len(df_f)

        if n == 0:
            return jsonify({"ok": True, "empty": True, "threshold": threshold})

        xs = list(range(n))

        # ── Volume traces ──────────────────────────────────────────────────
        vol_traces = [
            {
                "x": xs, "y": (df_f["left_volume"] + 0.45).tolist(),
                "type": "scatter", "mode": "lines",
                "name": "left_volume (+0.45)",
                "line": {"color": "#e53935", "width": 1.2},
            },
            {
                "x": xs, "y": df_f["right_volume"].tolist(),
                "type": "scatter", "mode": "lines",
                "name": "right_volume",
                "line": {"color": "#43a047", "width": 1.2},
            },
        ]

        # ── Heatmap trace ──────────────────────────────────────────────────
        heatmap      = build_heatmap(df_f, b_cols)
        heatmap_full = build_heatmap(df,   b_cols)
        z_max        = float(heatmap_full.max()) if heatmap_full.max() > 0 else 1.0
        bin_centers  = list(range(-175, 180, 10))

        heat_traces = [{
            "type": "heatmap",
            "x": xs,
            "y": bin_centers,
            "z": heatmap.T.tolist(),
            "colorscale": "Viridis",
            "zmin": 0,
            "zmax": z_max,
            "showscale": True,
            "colorbar": {"title": "Count", "thickness": 12, "len": 0.6,
                         "y": 0.72, "yanchor": "bottom"},
        }]

        # ── Angle trace ────────────────────────────────────────────────────
        angles = df_f["angle"].tolist()
        angle_traces = [{
            "x": xs, "y": angles,
            "type": "scatter", "mode": "lines",
            "name": "angle",
            "line": {"color": "#2196f3", "width": 1.2},
            "showlegend": False,
        }]

        # First angle > 90° marker
        over90_idx = next((i for i, a in enumerate(angles) if a > 90), None)
        shapes      = []
        annotations = []
        if over90_idx is not None:
            shapes.append({
                "type": "line",
                "xref": "x3", "yref": "paper",
                "x0": over90_idx, "x1": over90_idx,
                "y0": 0.515, "y1": 0.575,
                "line": {"color": "red", "dash": "dot", "width": 1.5},
            })
            annotations.append({
                "xref": "x3", "yref": "paper",
                "x": over90_idx, "y": 0.578,
                "text": f"first >90° (i={over90_idx})",
                "showarrow": False,
                "font": {"size": 9, "color": "red"},
                "xanchor": "left",
            })

        return jsonify({
            "ok": True, "empty": False,
            "n": n,
            "db_min": db_min, "db_max": db_max,
            "threshold": threshold,
            "vol_traces":   vol_traces,
            "heat_traces":  heat_traces,
            "angle_traces": angle_traces,
            "shapes":       shapes,
            "annotations":  annotations,
            "z_max":        z_max,
        })

    except FileNotFoundError:
        return jsonify({"ok": False, "error": f"File not found: {csv_path}"})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()})


@app.route("/api/plot_stats")
def api_plot_stats():
    """Stats view: mean + std-dev + count-metric panels."""
    experiment = request.args.get("experiment", "")
    run        = request.args.get("run", "")
    threshold  = request.args.get("threshold", type=float, default=None)
    if not experiment or not run:
        return jsonify({"ok": False, "error": "experiment and run required"})

    csv_path = DATA_DIR / experiment / run
    try:
        df = pd.read_csv(csv_path)
        if "dB_SPL" not in df.columns:
            return jsonify({"ok": False, "error": "Missing column: dB_SPL"})

        b_cols = bump_columns(df)
        if not b_cols:
            return jsonify({"ok": False, "error": "No bump columns found"})

        db_min = float(df["dB_SPL"].min())
        if threshold is None:
            threshold = db_min

        df_f = df[df["dB_SPL"] >= threshold].reset_index(drop=True)
        n    = len(df_f)

        if n == 0:
            return jsonify({"ok": True, "empty": True, "threshold": threshold})

        xs    = list(range(n))
        stats = compute_stats(df_f, b_cols)

        # ── Mean trace ─────────────────────────────────────────────────────
        mean_traces = [{
            "x": xs, "y": _nan_to_none(stats["mean_all"]),
            "type": "scatter", "mode": "lines",
            "name": "mean",
            "line": {"color": "#c9d1e0", "width": 1.4},
        }]

        # ── Std-dev traces ─────────────────────────────────────────────────
        sd_traces = [
            {
                "x": xs, "y": _nan_to_none(stats["pos_sd"]),
                "type": "scatter", "mode": "lines",
                "name": "positive SD",
                "line": {"color": "#f59e0b", "width": 1.0},
            },
            {
                "x": xs, "y": _nan_to_none(stats["neg_sd"]),
                "type": "scatter", "mode": "lines",
                "name": "negative SD",
                "line": {"color": "#60a5fa", "width": 1.0},
            },
        ]

        # ── Count metric trace + zero-line ─────────────────────────────────
        count_traces = [
            {
                "x": xs, "y": _nan_to_none(stats["count_metric"]),
                "type": "scatter", "mode": "lines",
                "name": "pos − neg count",
                "line": {"color": "#a78bfa", "width": 1.4},
            },
            {
                # zero reference line rendered as a scatter trace
                "x": [0, n - 1], "y": [0, 0],
                "type": "scatter", "mode": "lines",
                "name": "zero",
                "line": {"color": "#4ade80", "width": 1.0, "dash": "solid"},
                "showlegend": False,
            },
        ]

        return jsonify({
            "ok": True, "empty": False,
            "n": n,
            "threshold": threshold,
            "mean_traces":  mean_traces,
            "sd_traces":    sd_traces,
            "count_traces": count_traces,
        })

    except FileNotFoundError:
        return jsonify({"ok": False, "error": f"File not found: {csv_path}"})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()})


# ─────────────────────────────────────────────────────────────────────────────
# HTML Template
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bump Angle Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #0c0e14;
    --surface:  #13161f;
    --surface2: #1a1e2b;
    --border:   #252a3a;
    --accent:   #4a90d9;
    --red:      #e53935;
    --green:    #43a047;
    --text:     #d8dde8;
    --dim:      #626880;
    --mono:     'JetBrains Mono', monospace;
    --display:  'Syne', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ───────────────────────────────────────── */
  header {
    padding: 16px 28px 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 16px;
    background: var(--surface);
  }
  header h1 {
    font-family: var(--display);
    font-size: 19px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
  }
  header .sub {
    font-size: 10px;
    color: var(--dim);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* ── Controls ─────────────────────────────────────── */
  .controls {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 12px 28px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .ctrl-group { display: flex; flex-direction: column; gap: 4px; }
  .ctrl-label {
    font-size: 10px;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  select {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    padding: 6px 28px 6px 10px;
    border-radius: 4px;
    min-width: 160px;
    cursor: pointer;
    outline: none;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23626880'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
    transition: border-color 0.15s;
  }
  select:hover, select:focus { border-color: var(--accent); }

  /* ── Slider ───────────────────────────────────────── */
  .slider-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
    min-width: 260px;
  }
  .slider-row { display: flex; align-items: center; gap: 12px; }
  input[type=range] {
    flex: 1;
    accent-color: var(--accent);
    height: 4px;
    cursor: pointer;
  }
  .slider-val {
    font-size: 12px;
    color: var(--accent);
    min-width: 84px;
    text-align: right;
    font-weight: 600;
  }
  .slider-range {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--dim);
  }

  /* ── Status ───────────────────────────────────────── */
  .status {
    margin-left: auto;
    font-size: 11px;
    padding: 5px 10px;
    border-radius: 4px;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--dim);
    white-space: nowrap;
  }
  .status.ok   { color: #4ade80; border-color: #4ade80; }
  .status.err  { color: var(--red); border-color: var(--red); }
  .status.warn { color: #f9a825; border-color: #f9a825; }
  .status.busy { color: var(--accent); border-color: var(--accent); }

  /* ── Main layout ──────────────────────────────────── */
  main {
    flex: 1;
    padding: 16px 28px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  /* ── Error box ────────────────────────────────────── */
  #error-box {
    display: none;
    background: #1a0d0d;
    border: 1px solid #7b1a1a;
    border-radius: 6px;
    padding: 14px 18px;
    color: #ff8a80;
    font-size: 12px;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 180px;
    overflow-y: auto;
  }
  #error-box.visible { display: block; }
  #error-box .err-title {
    font-family: var(--display);
    font-weight: 700;
    font-size: 13px;
    color: #ff5252;
    margin-bottom: 6px;
  }

  /* ── Plot containers ──────────────────────────────── */
  .plot-section {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .plot-label {
    font-size: 10px;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding-left: 2px;
  }
  .plot-box {
    width: 100%;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    overflow: hidden;
    position: relative;
  }
  #plot-bump  { height: calc(52vh - 80px); min-height: 320px; }
  #plot-stats { height: calc(44vh - 60px); min-height: 270px; }

  /* ── Spinner (shared, shows over whichever is loading) ── */
  .spinner-overlay {
    display: none;
    position: absolute;
    inset: 0;
    align-items: center;
    justify-content: center;
    background: rgba(12,14,20,0.5);
    z-index: 10;
    border-radius: 6px;
  }
  .spinner-overlay.visible { display: flex; }
  .spin-ring {
    width: 26px; height: 26px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Hint ─────────────────────────────────────────── */
  .hint-msg {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--dim);
    font-size: 13px;
  }
</style>
</head>
<body>

<header>
  <h1>Bump Angle Viewer</h1>
  <span class="sub">bump plots + statistics · single dB_SPL threshold</span>
</header>

<div class="controls">
  <div class="ctrl-group">
    <span class="ctrl-label">Experiment</span>
    <select id="sel-exp" onchange="onExpChange()">
      <option value="">— loading —</option>
    </select>
  </div>

  <div class="ctrl-group">
    <span class="ctrl-label">Run</span>
    <select id="sel-run" onchange="onRunChange()" disabled>
      <option value="">— select experiment —</option>
    </select>
  </div>

  <div class="slider-wrap" id="slider-wrap" style="opacity:0.35;pointer-events:none">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <span class="ctrl-label">dB_SPL threshold</span>
      <span class="slider-val" id="slider-val">—</span>
    </div>
    <div class="slider-row">
      <input type="range" id="db-slider" min="0" max="1" step="0.01" value="0"
             oninput="onSliderInput()" onchange="onSliderCommit()">
    </div>
    <div class="slider-range">
      <span id="range-min">—</span>
      <span id="range-max">—</span>
    </div>
  </div>

  <div class="status" id="status">ready</div>
</div>

<main>
  <div id="error-box">
    <div class="err-title">⚠ Error</div>
    <div id="error-text"></div>
  </div>

  <!-- Bump plot (top) -->
  <div class="plot-section">
    <span class="plot-label">Bump view — volume · heatmap · angle</span>
    <div class="plot-box" id="bump-wrap">
      <div class="spinner-overlay" id="spin-bump"><div class="spin-ring"></div></div>
      <div id="plot-bump"><div class="hint-msg">← Select an experiment and run to begin</div></div>
    </div>
  </div>

  <!-- Stats plot (bottom) -->
  <div class="plot-section">
    <span class="plot-label">Statistics view — mean · std dev · count metric</span>
    <div class="plot-box" id="stats-wrap">
      <div class="spinner-overlay" id="spin-stats"><div class="spin-ring"></div></div>
      <div id="plot-stats"><div class="hint-msg">↑ Stats will appear here</div></div>
    </div>
  </div>
</main>

<script>
// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
let experiments     = {};
let dbMin = 0, dbMax = 1;
let bumpInitialised  = false;
let statsInitialised = false;
let sliderDebounce   = null;
let currentExp = "", currentRun = "";
let zMaxFixed  = 1;   // heatmap colorscale locked on first load

// ─────────────────────────────────────────────────────────────────────────────
// Startup
// ─────────────────────────────────────────────────────────────────────────────
async function init() {
  setStatus("scanning…", "busy");
  try {
    const res  = await fetch("/api/experiments");
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);

    experiments = data.experiments;
    const sel   = document.getElementById("sel-exp");
    sel.innerHTML = '<option value="">— choose —</option>';
    for (const exp of Object.keys(experiments).sort()) {
      const opt = document.createElement("option");
      opt.value = exp;
      opt.textContent = exp;
      sel.appendChild(opt);
    }
    setStatus("ready", "");
  } catch (e) {
    showError("Failed to scan experiments:\n" + e.message);
    setStatus("error", "err");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dropdown handlers
// ─────────────────────────────────────────────────────────────────────────────
function onExpChange() {
  const exp    = document.getElementById("sel-exp").value;
  const runSel = document.getElementById("sel-run");
  runSel.innerHTML = '<option value="">— choose run —</option>';

  if (!exp) { runSel.disabled = true; disableSlider(); return; }

  for (const r of (experiments[exp] || [])) {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = r.replace("_bumps.csv", "");
    runSel.appendChild(opt);
  }
  runSel.disabled = false;
  disableSlider();
  hideError();
}

async function onRunChange() {
  const exp = document.getElementById("sel-exp").value;
  const run = document.getElementById("sel-run").value;
  if (!exp || !run) return;

  currentExp = exp;
  currentRun = run;

  setStatus("loading…", "busy");
  showSpinner("bump",  true);
  showSpinner("stats", true);
  hideError();
  disableSlider();

  try {
    const res  = await fetch(`/api/load?experiment=${enc(exp)}&run=${enc(run)}`);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);

    dbMin   = data.db_min;
    dbMax   = data.db_max;
    zMaxFixed = null;   // will lock on first plot response

    setupSlider(dbMin, dbMax);
    enableSlider();

    await fetchBoth(dbMin, true);
  } catch (e) {
    showError(e.message);
    setStatus("error", "err");
    showSpinner("bump",  false);
    showSpinner("stats", false);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Slider
// ─────────────────────────────────────────────────────────────────────────────
function setupSlider(min, max) {
  const s = document.getElementById("db-slider");
  s.min   = min;
  s.max   = max;
  s.step  = (max !== min) ? (max - min) / 200 : 0.1;
  s.value = min;
  document.getElementById("slider-val").textContent  = min.toFixed(2) + " dB";
  document.getElementById("range-min").textContent   = min.toFixed(1);
  document.getElementById("range-max").textContent   = max.toFixed(1);
}

function enableSlider()  {
  const w = document.getElementById("slider-wrap");
  w.style.opacity = "1"; w.style.pointerEvents = "auto";
}
function disableSlider() {
  const w = document.getElementById("slider-wrap");
  w.style.opacity = "0.35"; w.style.pointerEvents = "none";
}

function onSliderInput() {
  const val = parseFloat(document.getElementById("db-slider").value);
  document.getElementById("slider-val").textContent = val.toFixed(2) + " dB";
  clearTimeout(sliderDebounce);
  sliderDebounce = setTimeout(() => fetchBoth(val, false), 150);
}
function onSliderCommit() {
  clearTimeout(sliderDebounce);
  fetchBoth(parseFloat(document.getElementById("db-slider").value), false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fetch both plots in parallel
// ─────────────────────────────────────────────────────────────────────────────
async function fetchBoth(threshold, firstLoad) {
  showSpinner("bump",  true);
  showSpinner("stats", true);
  setStatus("plotting…", "busy");

  const [bumpResult, statsResult] = await Promise.allSettled([
    fetchBump(threshold, firstLoad),
    fetchStats(threshold),
  ]);

  showSpinner("bump",  false);
  showSpinner("stats", false);

  const bumpOk  = bumpResult.status  === "fulfilled" && bumpResult.value  === true;
  const statsOk = statsResult.status === "fulfilled" && statsResult.value === true;

  const val = parseFloat(document.getElementById("db-slider").value);
  if (bumpOk && statsOk) {
    setStatus(`n rows · threshold = ${val.toFixed(2)} dB`, "ok");
  } else if (bumpOk || statsOk) {
    setStatus("partial error — see above", "warn");
  } else {
    setStatus("error", "err");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fetch + render: bump plot
// ─────────────────────────────────────────────────────────────────────────────
async function fetchBump(threshold, firstLoad) {
  try {
    const url  = `/api/plot?experiment=${enc(currentExp)}&run=${enc(currentRun)}&threshold=${threshold}`;
    const res  = await fetch(url);
    const data = await res.json();

    if (!data.ok) { showError("Bump plot: " + data.error); return false; }

    if (data.empty) { renderBumpEmpty(); return true; }

    if (firstLoad || zMaxFixed === null) zMaxFixed = data.z_max;
    data.heat_traces[0].zmax = zMaxFixed;

    renderBump(data, threshold);
    return true;
  } catch (e) {
    showError("Bump fetch error: " + e.message);
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fetch + render: stats plot
// ─────────────────────────────────────────────────────────────────────────────
async function fetchStats(threshold) {
  try {
    const url  = `/api/plot_stats?experiment=${enc(currentExp)}&run=${enc(currentRun)}&threshold=${threshold}`;
    const res  = await fetch(url);
    const data = await res.json();

    if (!data.ok) { showError("Stats plot: " + data.error); return false; }

    if (data.empty) { renderStatsEmpty(); return true; }

    renderStats(data, threshold);
    return true;
  } catch (e) {
    showError("Stats fetch error: " + e.message);
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Plotly: bump chart (3 sub-panels)
// ─────────────────────────────────────────────────────────────────────────────
const BASE_LAYOUT = {
  paper_bgcolor: "#13161f",
  plot_bgcolor:  "#0c0e14",
  font: { family: "JetBrains Mono, monospace", size: 11, color: "#d8dde8" },
  margin: { l: 62, r: 24, t: 36, b: 42 },
};

const CFG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d"],
  displaylogo: false,
  toImageButtonOptions: { scale: 2 },
};

function renderBump(data, threshold) {
  removeHint("plot-bump");

  const vol   = data.vol_traces.map(t   => ({ ...t, xaxis:"x",  yaxis:"y"  }));
  const heat  = data.heat_traces.map(t  => ({ ...t, xaxis:"x2", yaxis:"y2" }));
  const angle = data.angle_traces.map(t => ({ ...t, xaxis:"x3", yaxis:"y3" }));

  const layout = {
    ...BASE_LAYOUT,
    title: {
      text: `${currentRun.replace("_bumps.csv","")}  ·  threshold ≥ ${threshold.toFixed(2)} dB_SPL  ·  n = ${data.n}`,
      font: { family:"Syne, sans-serif", size:13, color:"#ffffff" },
      x: 0.04,
    },
    showlegend: true,
    legend: { x:1.01, y:0.98, font:{size:10}, bgcolor:"rgba(0,0,0,0)" },

    xaxis:  { domain:[0,1], anchor:"y",  showticklabels:false, gridcolor:"#1e2333", zeroline:false },
    yaxis:  { domain:[0.72,1.0], anchor:"x",  title:{text:"Volume",font:{size:10}}, range:[0.5,1.5], gridcolor:"#1e2333", zeroline:false },

    xaxis2: { domain:[0,1], anchor:"y2", showticklabels:false, gridcolor:"#1e2333", zeroline:false },
    yaxis2: { domain:[0.27,0.68], anchor:"x2", title:{text:"Angle bin (°)",font:{size:10}}, gridcolor:"#1e2333", zeroline:false },

    xaxis3: { domain:[0,1], anchor:"y3", title:{text:"Timestamp index",font:{size:10}}, gridcolor:"#1e2333", zeroline:false },
    yaxis3: { domain:[0.0,0.24],  anchor:"x3", title:{text:"Angle (°)",font:{size:10}}, gridcolor:"#1e2333", zeroline:false },

    shapes:      data.shapes,
    annotations: data.annotations,
  };

  if (!bumpInitialised) {
    Plotly.newPlot("plot-bump", [...vol,...heat,...angle], layout, CFG);
    bumpInitialised = true;
  } else {
    Plotly.react("plot-bump", [...vol,...heat,...angle], layout);
  }
}

function renderBumpEmpty() {
  removeHint("plot-bump");
  const layout = { ...BASE_LAYOUT,
    annotations:[{ text:"No data matches threshold", xref:"paper", yref:"paper",
      x:0.5, y:0.5, showarrow:false, font:{size:15,color:"#626880"} }]
  };
  if (!bumpInitialised) { Plotly.newPlot("plot-bump",[],layout,CFG); bumpInitialised=true; }
  else                  { Plotly.react("plot-bump",[],layout); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Plotly: stats chart (3 sub-panels)
// ─────────────────────────────────────────────────────────────────────────────
function renderStats(data, threshold) {
  removeHint("plot-stats");

  const mean  = data.mean_traces.map(t  => ({ ...t, xaxis:"x",  yaxis:"y"  }));
  const sd    = data.sd_traces.map(t    => ({ ...t, xaxis:"x2", yaxis:"y2" }));
  const count = data.count_traces.map(t => ({ ...t, xaxis:"x3", yaxis:"y3" }));

  const layout = {
    ...BASE_LAYOUT,
    title: {
      text: `Statistics  ·  threshold ≥ ${threshold.toFixed(2)} dB_SPL  ·  n = ${data.n}`,
      font: { family:"Syne, sans-serif", size:13, color:"#ffffff" },
      x: 0.04,
    },
    showlegend: true,
    legend: { x:1.01, y:0.98, font:{size:10}, bgcolor:"rgba(0,0,0,0)" },

    xaxis:  { domain:[0,1], anchor:"y",  showticklabels:false, gridcolor:"#1e2333", zeroline:false },
    yaxis:  { domain:[0.70,1.0],  anchor:"x",  title:{text:"Mean (°)",font:{size:10}}, gridcolor:"#1e2333", zeroline:false },

    xaxis2: { domain:[0,1], anchor:"y2", showticklabels:false, gridcolor:"#1e2333", zeroline:false },
    yaxis2: { domain:[0.36,0.66], anchor:"x2", title:{text:"Std Dev (°)",font:{size:10}}, gridcolor:"#1e2333", zeroline:false },

    xaxis3: { domain:[0,1], anchor:"y3", title:{text:"Timestamp index",font:{size:10}}, gridcolor:"#1e2333", zeroline:false },
    yaxis3: { domain:[0.0,0.32],  anchor:"x3", title:{text:"Count (pos−neg)",font:{size:10}}, gridcolor:"#1e2333", zeroline:false },
  };

  if (!statsInitialised) {
    Plotly.newPlot("plot-stats", [...mean,...sd,...count], layout, CFG);
    statsInitialised = true;
  } else {
    Plotly.react("plot-stats", [...mean,...sd,...count], layout);
  }
}

function renderStatsEmpty() {
  removeHint("plot-stats");
  const layout = { ...BASE_LAYOUT,
    annotations:[{ text:"No data matches threshold", xref:"paper", yref:"paper",
      x:0.5, y:0.5, showarrow:false, font:{size:15,color:"#626880"} }]
  };
  if (!statsInitialised) { Plotly.newPlot("plot-stats",[],layout,CFG); statsInitialised=true; }
  else                   { Plotly.react("plot-stats",[],layout); }
}

// ─────────────────────────────────────────────────────────────────────────────
// UI helpers
// ─────────────────────────────────────────────────────────────────────────────
function enc(s)  { return encodeURIComponent(s); }

function removeHint(id) {
  const h = document.querySelector(`#${id} .hint-msg`);
  if (h) h.remove();
}

function setStatus(msg, cls) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = "status " + cls;
}

function showError(msg) {
  const box = document.getElementById("error-box");
  document.getElementById("error-text").textContent = msg;
  box.classList.add("visible");
}
function hideError() {
  document.getElementById("error-box").classList.remove("visible");
}

function showSpinner(which, on) {
  document.getElementById(`spin-${which}`).classList.toggle("visible", on);
}

// ─────────────────────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(description="Bump Angle interactive viewer")
    parser.add_argument("--data-dir", required=True,
                        help="Root folder containing experiment sub-folders")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    DATA_DIR = pathlib.Path(args.data_dir).expanduser().resolve()
    if not DATA_DIR.is_dir():
        print(f"ERROR: not a directory: {DATA_DIR}")
        sys.exit(1)

    print(f"  Data directory : {DATA_DIR}")
    print(f"  Serving at     : http://{args.host}:{args.port}")
    print()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()