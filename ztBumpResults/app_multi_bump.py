#!/usr/bin/env python3
"""
Multi-run Bump Heatmap Viewer

Usage:
    python app.py --data-dir ./oldResult
"""

import argparse
import pathlib
import traceback

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
DATA_DIR = None


def scan_experiments():
    return sorted(
        p.stem.replace("_bumps", "")
        for p in DATA_DIR.glob("*_bumps.csv")
    )


def bump_columns(df):
    cols = [c for c in df.columns if c.startswith("bump") and c[4:].isdigit()]
    cols.sort(key=lambda x: int(x[4:]))
    return cols


def build_heatmap(df, b_cols):
    bin_edges = np.arange(-180, 181, 10)
    n_bins = len(bin_edges) - 1
    heatmap = np.zeros((len(df), n_bins))

    for i, (_, row) in enumerate(df[b_cols].iterrows()):
        counts, _ = np.histogram(row.dropna().values, bins=bin_edges)
        heatmap[i] = counts

    return heatmap


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/experiments")
def experiments():
    return jsonify({"ok": True, "experiments": scan_experiments()})


@app.route("/api/load")
def load():
    exp = request.args.get("experiment", "")
    path = DATA_DIR / f"{exp}_bumps.csv"

    try:
        df = pd.read_csv(path)
        return jsonify({
            "ok": True,
            "db_min": float(df["dB_SPL"].min()),
            "db_max": float(df["dB_SPL"].max())
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/heatmap")
def heatmap():
    exp = request.args.get("experiment", "")
    threshold = request.args.get("threshold", type=float)

    try:
        path = DATA_DIR / f"{exp}_bumps.csv"
        df = pd.read_csv(path)

        b_cols = bump_columns(df)

        if threshold is None:
            threshold = float(df["dB_SPL"].min())

        df = df[df["dB_SPL"] >= threshold].reset_index(drop=True)

        hm = build_heatmap(df, b_cols)

        return jsonify({
            "ok": True,
            "experiment": exp,
            "n": len(df),
            "x": list(range(len(df))),
            "y": list(range(-175, 180, 10)),
            "z": hm.T.tolist(),
            "zmax": float(hm.max()) if hm.size else 1.0
        })

    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()})


HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Multi Run Bump Heatmaps</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>

<style>
body{
 background:#0c0e14;
 color:#ddd;
 font-family:monospace;
 margin:0;
}

.header{
 padding:16px;
 border-bottom:1px solid #333;
}

.controls{
 padding:16px;
 display:flex;
 gap:20px;
 align-items:flex-start;
 border-bottom:1px solid #333;
}

.dropdown{
 position:relative;
}

.dropdown-btn{
 background:#1b1f2a;
 border:1px solid #444;
 color:white;
 padding:8px 12px;
 cursor:pointer;
}

.dropdown-content{
 display:none;
 position:absolute;
 background:#1b1f2a;
 border:1px solid #444;
 max-height:300px;
 overflow:auto;
 min-width:250px;
 z-index:1000;
 padding:8px;
}

.dropdown.open .dropdown-content{
 display:block;
}

.plot-card{
 margin:12px;
 border:1px solid #333;
 background:#13161f;
 border-radius:6px;
}

.plot-title{
 padding:8px 12px;
 border-bottom:1px solid #333;
 font-weight:bold;
}

.plot{
 height:300px;
}
</style>
</head>
<body>

<div class="header">
<h2>Multi Run Bump Heatmap Viewer</h2>
</div>

<div class="controls">

<div class="dropdown" id="dropdown">
<button class="dropdown-btn" onclick="toggleDropdown()">Select Runs ▼</button>
<div class="dropdown-content" id="runList"></div>
</div>

<div>
<div>dB Threshold</div>
<input type="range" id="slider" min="0" max="1" step="0.1">
<div id="sliderValue"></div>
</div>

</div>

<div id="plots"></div>

<script>

let runs=[];
let selectedRuns=new Set();
let dbMin=0;
let dbMax=1;

function toggleDropdown(){
 document.getElementById("dropdown").classList.toggle("open");
}

async function init(){

 let r=await fetch('/api/experiments');
 let d=await r.json();

 runs=d.experiments;

 const runList=document.getElementById("runList");

 runs.forEach(run=>{

   const row=document.createElement("div");

   row.innerHTML=`
     <label>
       <input type="checkbox" value="${run}" onchange="toggleRun(this)">
       ${run}
     </label>
   `;

   runList.appendChild(row);
 });

 if(runs.length){
   let res=await fetch(`/api/load?experiment=${encodeURIComponent(runs[0])}`);
   let info=await res.json();

   dbMin=info.db_min;
   dbMax=info.db_max;

   const s=document.getElementById("slider");
   s.min=dbMin;
   s.max=dbMax;
   s.value=dbMin;
   s.step=(dbMax-dbMin)/200;

   updateSliderText();

   s.addEventListener("input", ()=>{
      updateSliderText();
      refreshAll();
   });
 }
}

function updateSliderText(){
 const v=parseFloat(document.getElementById("slider").value);
 document.getElementById("sliderValue").textContent=v.toFixed(2)+" dB";
}

function toggleRun(cb){

 if(cb.checked)
   selectedRuns.add(cb.value);
 else
   selectedRuns.delete(cb.value);

 refreshAll();
}

async function refreshAll(){

 const plots=document.getElementById("plots");
 plots.innerHTML="";

 const threshold=document.getElementById("slider").value;

 const promises=[...selectedRuns].map(run=>
   fetch(`/api/heatmap?experiment=${encodeURIComponent(run)}&threshold=${threshold}`)
     .then(r=>r.json())
 );

 const results=await Promise.all(promises);

 results.forEach(renderPlot);
}

function renderPlot(data){

 if(!data.ok) return;

 const container=document.createElement("div");
 container.className="plot-card";

 const title=document.createElement("div");
 title.className="plot-title";
 title.textContent=`${data.experiment} | n=${data.n}`;

 const plot=document.createElement("div");
 plot.className="plot";

 container.appendChild(title);
 container.appendChild(plot);

 document.getElementById("plots").appendChild(container);

 Plotly.newPlot(
   plot,
   [{
      type:"heatmap",
      x:data.x,
      y:data.y,
      z:data.z,
      colorscale:"Viridis",
      zmin:0,
      zmax:data.zmax
   }],
   {
      paper_bgcolor:"#13161f",
      plot_bgcolor:"#0c0e14",
      font:{color:"#ddd"},
      margin:{l:60,r:20,t:20,b:40},
      xaxis:{
         title:"Sample Index"
      },
      yaxis:{
         title:"Angle (°)",
         range:[-180,180]
      }
   },
   {responsive:true}
 );
}

init();

</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5004)
    args = parser.parse_args()

    DATA_DIR = pathlib.Path(args.data_dir).resolve()

    app.run(host=args.host, port=args.port, debug=False)
