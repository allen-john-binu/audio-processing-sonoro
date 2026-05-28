#!/usr/bin/env python3
"""
bump_viewer/app.py
──────────────────
Interactive bump-angle visualization server.

Usage:
    python3 app.py --data-dir ./oldResult
    python3 app.py --data-dir /path/to/data --port 5000
"""

import argparse
import json
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

def scan_experiments() -> dict[str, list[str]]:
    """
    Scan DATA_DIR for experiment folders containing *_bumps.csv files.

    Returns
    -------
    dict mapping experiment name -> sorted list of run CSV filenames
    """
    result: dict[str, list[str]] = {}

    if DATA_DIR is None or not DATA_DIR.is_dir():
        return result

    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue

        csvs = sorted(
            p.name for p in folder.glob("*_bumps.csv")
        )

        if csvs:
            result[folder.name] = csvs

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def bump_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("bump") and c[4:].isdigit()]
    cols.sort(key=lambda c: int(c[4:]))
    return cols


def build_heatmap(df: pd.DataFrame, b_cols: list[str]) -> np.ndarray:
    """Returns ndarray shape (n_timesteps, n_angle_bins)."""
    bin_edges = np.arange(-180, 181, 10)
    n_bins = len(bin_edges) - 1
    n_timesteps = len(df)
    heatmap = np.zeros((n_timesteps, n_bins))

    for i, (_, row) in enumerate(df[b_cols].iterrows()):
        counts, _ = np.histogram(row.dropna().values, bins=bin_edges)
        heatmap[i] = counts

    return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/experiments")
def api_experiments():
    try:
        data = scan_experiments()
        return jsonify({"ok": True, "experiments": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/load")
def api_load():
    """
    Load a CSV and return:
      - db_min, db_max
      - x-axis length (full unfiltered)
    """
    experiment = request.args.get("experiment", "")
    run = request.args.get("run", "")

    if not experiment or not run:
        return jsonify({"ok": False, "error": "experiment and run required"})

    csv_path = DATA_DIR / experiment / run

    try:
        df = pd.read_csv(csv_path)

        required = {"dB_SPL", "angle", "left_volume", "right_volume"}
        missing = required - set(df.columns)
        if missing:
            return jsonify({"ok": False, "error": f"Missing columns: {missing}"})

        b_cols = bump_columns(df)
        if not b_cols:
            return jsonify({"ok": False, "error": "No bump columns found in CSV"})

        return jsonify({
            "ok": True,
            "db_min": float(df["dB_SPL"].min()),
            "db_max": float(df["dB_SPL"].max()),
            "n_total": len(df),
            "n_bumps": len(b_cols),
        })

    except FileNotFoundError:
        return jsonify({"ok": False, "error": f"File not found: {csv_path}"})
    except Exception as e:
        return jsonify({"ok": False, "error": traceback.format_exc()})


@app.route("/api/plot")
def api_plot():
    """
    Filter CSV by dB_SPL threshold and return Plotly-ready JSON traces.
    """
    experiment = request.args.get("experiment", "")
    run = request.args.get("run", "")
    threshold = request.args.get("threshold", type=float, default=None)

    if not experiment or not run:
        return jsonify({"ok": False, "error": "experiment and run required"})

    csv_path = DATA_DIR / experiment / run

    try:
        df = pd.read_csv(csv_path)

        required = {"dB_SPL", "angle", "left_volume", "right_volume"}
        missing = required - set(df.columns)
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
        n = len(df_f)

        if n == 0:
            return jsonify({"ok": True, "empty": True, "threshold": threshold})

        xs = list(range(n))

        # ── Panel 1: volume ────────────────────────────────────────────────
        left_vol = (df_f["left_volume"] + 0.45).tolist()
        right_vol = df_f["right_volume"].tolist()

        vol_traces = [
            {
                "x": xs, "y": left_vol,
                "type": "scatter", "mode": "lines",
                "name": "left_volume (+0.45)",
                "line": {"color": "#e53935", "width": 1.2},
            },
            {
                "x": xs, "y": right_vol,
                "type": "scatter", "mode": "lines",
                "name": "right_volume",
                "line": {"color": "#43a047", "width": 1.2},
            },
        ]

        # ── Panel 2: heatmap ───────────────────────────────────────────────
        heatmap = build_heatmap(df_f, b_cols)  # (n_timesteps, 36)

        # z for Plotly heatmap: rows = angle bins, cols = time
        z = heatmap.T.tolist()  # shape (36, n)

        bin_centers = list(range(-175, 180, 10))  # 36 values

        # Fixed colorscale range uses full (unfiltered) data
        df_full_heat = build_heatmap(df, b_cols)
        z_max = float(df_full_heat.max()) if df_full_heat.max() > 0 else 1.0

        heat_traces = [
            {
                "type": "heatmap",
                "x": xs,
                "y": bin_centers,
                "z": z,
                "colorscale": "Viridis",
                "zmin": 0,
                "zmax": z_max,
                "showscale": True,
                "colorbar": {"title": "Count", "thickness": 14, "len": 0.9},
            }
        ]

        # ── Panel 3: angle ────────────────────────────────────────────────
        angles = df_f["angle"].tolist()

        angle_traces = [
            {
                "x": xs, "y": angles,
                "type": "scatter", "mode": "lines",
                "name": "angle",
                "line": {"color": "#2196f3", "width": 1.2},
                "showlegend": False,
            }
        ]

        # First angle > 90°
        over90_idx = next(
            (i for i, a in enumerate(angles) if a > 90), None
        )

        shapes = []
        annotations = []
        if over90_idx is not None:
            shapes.append({
                "type": "line",
                "xref": "x3", "yref": "paper",
                "x0": over90_idx, "x1": over90_idx,
                "y0": 0, "y1": 0.28,   # panel 3 occupies bottom ~28%
                "line": {"color": "red", "dash": "dot", "width": 1.5},
            })
            annotations.append({
                "xref": "x3", "yref": "paper",
                "x": over90_idx, "y": 0.29,
                "text": f"first >90° (i={over90_idx})",
                "showarrow": False,
                "font": {"size": 9, "color": "red"},
                "xanchor": "left",
            })

        return jsonify({
            "ok": True,
            "empty": False,
            "n": n,
            "db_min": db_min,
            "db_max": db_max,
            "threshold": threshold,
            "vol_traces": vol_traces,
            "heat_traces": heat_traces,
            "angle_traces": angle_traces,
            "shapes": shapes,
            "annotations": annotations,
            "z_max": z_max,
        })

    except FileNotFoundError:
        return jsonify({"ok": False, "error": f"File not found: {csv_path}"})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()})


# ─────────────────────────────────────────────────────────────────────────────
# HTML Template (single-page app)
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
    --bg:        #0c0e14;
    --surface:   #13161f;
    --surface2:  #1a1e2b;
    --border:    #252a3a;
    --accent:    #4a90d9;
    --accent2:   #e53935;
    --accent3:   #43a047;
    --text:      #d8dde8;
    --text-dim:  #626880;
    --mono:      'JetBrains Mono', monospace;
    --display:   'Syne', sans-serif;
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

  /* ── Header ── */
  header {
    padding: 18px 28px 14px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 16px;
    background: var(--surface);
  }

  header h1 {
    font-family: var(--display);
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
  }

  header .sub {
    font-size: 11px;
    color: var(--text-dim);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* ── Control bar ── */
  .controls {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 14px 28px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }

  .ctrl-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .ctrl-label {
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  select {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    padding: 6px 10px;
    border-radius: 4px;
    min-width: 160px;
    cursor: pointer;
    outline: none;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23626880'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
    padding-right: 28px;
    transition: border-color 0.15s;
  }

  select:hover, select:focus {
    border-color: var(--accent);
  }

  /* ── Slider section ── */
  .slider-wrap {
    display: flex;
    flex-direction: column;
    gap: 5px;
    flex: 1;
    min-width: 240px;
  }

  .slider-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  input[type=range] {
    flex: 1;
    accent-color: var(--accent);
    height: 4px;
    cursor: pointer;
  }

  .slider-val {
    font-size: 12px;
    color: var(--accent);
    min-width: 80px;
    text-align: right;
    font-weight: 600;
  }

  .slider-range {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-dim);
  }

  /* ── Status badge ── */
  .status {
    margin-left: auto;
    font-size: 11px;
    padding: 5px 10px;
    border-radius: 4px;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text-dim);
    white-space: nowrap;
  }

  .status.ok   { color: var(--accent3); border-color: var(--accent3); }
  .status.err  { color: var(--accent2); border-color: var(--accent2); }
  .status.warn { color: #f9a825;        border-color: #f9a825; }

  /* ── Main plot area ── */
  main {
    flex: 1;
    padding: 20px 28px;
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  #plot {
    width: 100%;
    height: calc(100vh - 180px);
    min-height: 560px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    overflow: hidden;
  }

  /* ── Error overlay ── */
  #error-box {
    display: none;
    background: #1a0d0d;
    border: 1px solid #7b1a1a;
    border-radius: 6px;
    padding: 18px 22px;
    margin-bottom: 16px;
    color: #ff8a80;
    font-size: 12px;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 200px;
    overflow-y: auto;
  }

  #error-box.visible { display: block; }

  #error-box .err-title {
    font-family: var(--display);
    font-weight: 700;
    font-size: 13px;
    color: #ff5252;
    margin-bottom: 8px;
  }

  /* ── Empty state ── */
  #empty-msg {
    display: none;
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    color: var(--text-dim);
  }

  #empty-msg.visible { display: block; }

  #plot-wrap { position: relative; flex: 1; }

  /* ── Loading spinner ── */
  #spinner {
    display: none;
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 28px; height: 28px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    z-index: 10;
  }

  #spinner.visible { display: block; }

  @keyframes spin { to { transform: translate(-50%, -50%) rotate(360deg); } }

  /* ── Hint text ── */
  #hint {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-dim);
    font-size: 13px;
    gap: 8px;
  }
</style>
</head>
<body>

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

  <div id="plot-wrap">
    <div id="spinner"></div>
    <div id="plot">
      <div id="hint">← Select an experiment and run to begin</div>
    </div>
  </div>
</main>

<script>
// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
let experiments = {};
let dbMin = 0, dbMax = 1;
let plotInitialised = false;
let sliderDebounce = null;
let currentExp = "", currentRun = "";
let zMax = 1;  // fixed colorscale across slider moves

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────
async function init() {
  setStatus("scanning…", "");
  try {
    const res = await fetch("/api/experiments");
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);

    experiments = data.experiments;
    const sel = document.getElementById("sel-exp");
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
  const exp = document.getElementById("sel-exp").value;
  const runSel = document.getElementById("sel-run");
  runSel.innerHTML = '<option value="">— choose run —</option>';

  if (!exp) {
    runSel.disabled = true;
    disableSlider();
    return;
  }

  const runs = experiments[exp] || [];
  for (const r of runs) {
    const opt = document.createElement("option");
    opt.value = r;
    // Show a friendlier label: strip _bumps.csv suffix
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

  setStatus("loading…", "");
  showSpinner(true);
  hideError();
  disableSlider();

  try {
    const res = await fetch(`/api/load?experiment=${encodeURIComponent(exp)}&run=${encodeURIComponent(run)}`);
    const data = await res.json();

    if (!data.ok) throw new Error(data.error);

    dbMin = data.db_min;
    dbMax = data.db_max;
    zMax  = null;  // will be set from first plot call

    setupSlider(dbMin, dbMax);
    enableSlider();

    // Trigger first plot at db_min
    await fetchAndPlot(dbMin, true);
  } catch (e) {
    showError(e.message);
    setStatus("error", "err");
    showSpinner(false);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Slider
// ─────────────────────────────────────────────────────────────────────────────
function setupSlider(min, max) {
  const slider = document.getElementById("db-slider");
  const steps = 200;
  slider.min = min;
  slider.max = max;
  slider.step = (max !== min) ? (max - min) / steps : 0.1;
  slider.value = min;
  document.getElementById("slider-val").textContent = min.toFixed(2) + " dB";
  document.getElementById("range-min").textContent = min.toFixed(1);
  document.getElementById("range-max").textContent = max.toFixed(1);
}

function enableSlider() {
  const w = document.getElementById("slider-wrap");
  w.style.opacity = "1";
  w.style.pointerEvents = "auto";
}

function disableSlider() {
  const w = document.getElementById("slider-wrap");
  w.style.opacity = "0.35";
  w.style.pointerEvents = "none";
}

function onSliderInput() {
  const val = parseFloat(document.getElementById("db-slider").value);
  document.getElementById("slider-val").textContent = val.toFixed(2) + " dB";
  // Debounce — only fire plot 150 ms after user stops dragging
  clearTimeout(sliderDebounce);
  sliderDebounce = setTimeout(() => fetchAndPlot(val, false), 150);
}

function onSliderCommit() {
  clearTimeout(sliderDebounce);
  const val = parseFloat(document.getElementById("db-slider").value);
  fetchAndPlot(val, false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Plot fetch + render
// ─────────────────────────────────────────────────────────────────────────────
async function fetchAndPlot(threshold, firstLoad) {
  showSpinner(true);
  setStatus("plotting…", "");

  try {
    const url = `/api/plot?experiment=${encodeURIComponent(currentExp)}&run=${encodeURIComponent(currentRun)}&threshold=${threshold}`;
    const res  = await fetch(url);
    const data = await res.json();

    showSpinner(false);

    if (!data.ok) throw new Error(data.error);

    if (data.empty) {
      setStatus("0 rows match threshold", "warn");
      renderEmpty();
      return;
    }

    // Store zMax from first load so colorscale stays fixed
    if (firstLoad) zMax = data.z_max;

    // Clamp zMax into heatmap trace
    data.heat_traces[0].zmax = zMax;

    renderPlot(data);
    setStatus(`n = ${data.n} rows  |  threshold = ${threshold.toFixed(2)} dB`, "ok");

  } catch (e) {
    showSpinner(false);
    showError(e.message);
    setStatus("error", "err");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Plotly rendering
// ─────────────────────────────────────────────────────────────────────────────
const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "#13161f",
  plot_bgcolor:  "#0c0e14",
  font: { family: "JetBrains Mono, monospace", size: 11, color: "#d8dde8" },
  margin: { l: 60, r: 20, t: 36, b: 40 },
  showlegend: true,
  legend: { x: 1.01, y: 1, font: { size: 10 }, bgcolor: "rgba(0,0,0,0)" },
  grid: { rows: 3, columns: 1, pattern: "independent", roworder: "top to bottom" },
};

function renderPlot(data) {
  const hint = document.getElementById("hint");
  if (hint) hint.remove();

  const volTraces   = data.vol_traces.map(t => ({ ...t, xaxis: "x",  yaxis: "y"  }));
  const heatTraces  = data.heat_traces.map(t => ({ ...t, xaxis: "x2", yaxis: "y2" }));
  const angleTraces = data.angle_traces.map(t => ({ ...t, xaxis: "x3", yaxis: "y3" }));

  const allTraces = [...volTraces, ...heatTraces, ...angleTraces];

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    title: {
      text: `${currentRun.replace("_bumps.csv","")}  ·  threshold ≥ ${parseFloat(document.getElementById("db-slider").value).toFixed(2)} dB_SPL`,
      font: { family: "Syne, sans-serif", size: 14, color: "#ffffff" },
      x: 0.04,
    },

    // Row sizing
    xaxis:  { domain: [0, 1], anchor: "y",  showticklabels: false, gridcolor: "#1e2333", zeroline: false },
    yaxis:  { domain: [0.72, 1.0], anchor: "x", title: { text: "Volume", font:{size:11} }, range:[0.5,1.5], gridcolor: "#1e2333", zeroline: false },

    xaxis2: { domain: [0, 1], anchor: "y2", showticklabels: false, gridcolor: "#1e2333", zeroline: false },
    yaxis2: { domain: [0.28, 0.70], anchor: "x2", title: { text: "Angle bin (°)", font:{size:11} }, gridcolor: "#1e2333", zeroline: false },

    xaxis3: { domain: [0, 1], anchor: "y3", title: { text: "Timestamp index", font:{size:11} }, gridcolor: "#1e2333", zeroline: false },
    yaxis3: { domain: [0.00, 0.26], anchor: "x3", title: { text: "Angle (°)", font:{size:11} }, gridcolor: "#1e2333", zeroline: false },

    shapes: data.shapes,
    annotations: data.annotations,
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["select2d", "lasso2d"],
    displaylogo: false,
    toImageButtonOptions: { filename: currentRun.replace(".csv",""), scale: 2 },
  };

  if (!plotInitialised) {
    Plotly.newPlot("plot", allTraces, layout, config);
    plotInitialised = true;
  } else {
    Plotly.react("plot", allTraces, layout, config);
  }
}

function renderEmpty() {
  const emptyLayout = {
    ...PLOTLY_LAYOUT_BASE,
    annotations: [{
      text: "No data matches the current threshold",
      xref: "paper", yref: "paper",
      x: 0.5, y: 0.5,
      showarrow: false,
      font: { size: 16, color: "#626880" },
    }],
  };

  if (!plotInitialised) {
    Plotly.newPlot("plot", [], emptyLayout, { responsive: true, displaylogo: false });
    plotInitialised = true;
  } else {
    Plotly.react("plot", [], emptyLayout);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// UI helpers
// ─────────────────────────────────────────────────────────────────────────────
function setStatus(msg, cls) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = "status " + cls;
}

function showError(msg) {
  const box  = document.getElementById("error-box");
  const text = document.getElementById("error-text");
  text.textContent = msg;
  box.classList.add("visible");
}

function hideError() {
  document.getElementById("error-box").classList.remove("visible");
}

function showSpinner(on) {
  document.getElementById("spinner").classList.toggle("visible", on);
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

    parser = argparse.ArgumentParser(
        description="Bump Angle interactive viewer"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root folder containing experiment sub-folders",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run on (default: 5000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    DATA_DIR = pathlib.Path(args.data_dir).expanduser().resolve()

    if not DATA_DIR.is_dir():
        print(f"ERROR: data-dir does not exist or is not a directory: {DATA_DIR}")
        sys.exit(1)

    print(f"  Data directory : {DATA_DIR}")
    print(f"  Serving at     : http://{args.host}:{args.port}")
    print()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()