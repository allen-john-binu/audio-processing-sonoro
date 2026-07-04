"""
app.py  –  Ring Attractor Result Explorer
------------------------------------------------------
Run from inside ztBumpResults/:
    python app.py

Or point at another folder:
    python app.py --results-dir /path/to/results

Then open http://127.0.0.1:8050 in your browser.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-dir", type=str, default=".",
    help="Folder that contains *_bumps.csv files (default: current directory)."
)
parser.add_argument("--port", type=int, default=8050)
args, _ = parser.parse_known_args()

RESULTS_DIR = args.results_dir

# ── Discover CSV files ─────────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_bumps.csv")))
if not csv_files:
    raise FileNotFoundError(f"No *_bumps.csv files found in: {RESULTS_DIR}")

file_options = [
    {"label": os.path.basename(p), "value": p}
    for p in csv_files
]

SPIN_COLS = [f"spin_{i}" for i in range(120)]

# ── Palette ────────────────────────────────────────────────────────────────────
BG        = "#0d1117"
PANEL     = "#161b22"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
ACCENT    = "#58a6ff"
RED       = "#f85149"
GREEN     = "#3fb950"
YELLOW    = "#d29922"
PURPLE    = "#bc8cff"

# ── App ────────────────────────────────────────────────────────────────────────
app = Dash(__name__, title="Ring Attractor Explorer")

app.layout = html.Div(
    style={
        "backgroundColor": BG,
        "minHeight": "100vh",
        "fontFamily": "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
        "color": TEXT,
        "padding": "24px 32px",
        "boxSizing": "border-box",
    },
    children=[
        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            style={"marginBottom": "24px", "borderBottom": f"1px solid {BORDER}", "paddingBottom": "16px"},
            children=[
                html.H1(
                    "Ring Attractor Explorer",
                    style={"margin": 0, "fontSize": "22px", "fontWeight": "600",
                           "letterSpacing": "0.04em", "color": ACCENT},
                ),
                html.P(
                    "Select a result file to explore neural spin activity and robot trajectory.",
                    style={"margin": "6px 0 0", "fontSize": "13px", "color": MUTED},
                ),
            ],
        ),

        # ── File selector ────────────────────────────────────────────────────
        html.Div(
            style={"marginBottom": "24px", "display": "flex", "alignItems": "center", "gap": "16px"},
            children=[
                html.Label("Result file:", style={"fontSize": "13px", "color": MUTED, "whiteSpace": "nowrap"}),
                dcc.Dropdown(
                    id="file-select",
                    options=file_options,
                    value=file_options[0]["value"],
                    clearable=False,
                    style={
                        "flex": "1",
                        "backgroundColor": PANEL,
                        "color": TEXT,
                        "border": f"1px solid {BORDER}",
                        "borderRadius": "6px",
                        "fontSize": "13px",
                    },
                ),
                html.Div(
                    id="timestep-badge",
                    style={
                        "padding": "4px 12px",
                        "backgroundColor": PANEL,
                        "border": f"1px solid {BORDER}",
                        "borderRadius": "6px",
                        "fontSize": "12px",
                        "color": MUTED,
                        "whiteSpace": "nowrap",
                    },
                    children="t = —",
                ),
            ],
        ),

        # ── Raster plot ──────────────────────────────────────────────────────
        html.Div(
            style={
                "backgroundColor": PANEL,
                "border": f"1px solid {BORDER}",
                "borderRadius": "8px",
                "padding": "16px",
                "marginBottom": "16px",
            },
            children=[
                html.Div(
                    style={"fontSize": "12px", "color": MUTED, "marginBottom": "8px",
                           "textTransform": "uppercase", "letterSpacing": "0.08em"},
                    children="Neural Spin Activity  ·  click a timestep to link trajectory",
                ),
                dcc.Graph(
                    id="raster-plot",
                    config={"displayModeBar": False},
                    style={"height": "380px"},
                ),
            ],
        ),

        # ── Trajectory plot ──────────────────────────────────────────────────
        html.Div(
            style={
                "backgroundColor": PANEL,
                "border": f"1px solid {BORDER}",
                "borderRadius": "8px",
                "padding": "16px",
            },
            children=[
                html.Div(
                    style={"fontSize": "12px", "color": MUTED, "marginBottom": "8px",
                           "textTransform": "uppercase", "letterSpacing": "0.08em"},
                    children="Spatial Trajectory  ·  click a point to link raster",
                ),
                dcc.Graph(
                    id="traj-plot",
                    config={"displayModeBar": False},
                    style={"height": "420px"},
                ),
            ],
        ),

        # ── Hidden stores ────────────────────────────────────────────────────
        dcc.Store(id="store-data"),          # serialised per-file data
        dcc.Store(id="store-selected-t"),    # currently selected timestep index
    ],
)


# ── Load CSV into store ────────────────────────────────────────────────────────
@app.callback(
    Output("store-data", "data"),
    Input("file-select", "value"),
)
def load_file(path):
    df = pd.read_csv(path)

    spins = df[SPIN_COLS].values.tolist()           # list[list[int]], shape (T, 120)
    bump  = df["bump_angle"].tolist()
    t_idx = list(range(len(df)))

    robot_x    = df["robot_x"].tolist()
    robot_y    = df["robot_y"].tolist()
    speakerL_x = df["speakerL_x"].tolist()
    speakerL_y = df["speakerL_y"].tolist()
    speakerR_x = df["speakerR_x"].tolist()
    speakerR_y = df["speakerR_y"].tolist()

    return {
        "spins":      spins,
        "bump":       bump,
        "t_idx":      t_idx,
        "robot_x":    robot_x,    "robot_y":    robot_y,
        "speakerL_x": speakerL_x, "speakerL_y": speakerL_y,
        "speakerR_x": speakerR_x, "speakerR_y": speakerR_y,
        "n": len(df),
    }


# ── Sync selected timestep from either plot ───────────────────────────────────
@app.callback(
    Output("store-selected-t", "data"),
    Input("raster-plot", "clickData"),
    Input("traj-plot", "clickData"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def sync_timestep(raster_click, traj_click, data):
    if data is None:
        return None

    triggered = callback_context.triggered[0]["prop_id"]

    if "raster-plot" in triggered and raster_click:
        # x on the raster is the timestep index directly
        pt = raster_click["points"][0]
        return int(round(pt["x"]))

    if "traj-plot" in triggered and traj_click:
        pt = traj_click["points"][0]
        # customdata carries the timestep index
        if pt.get("customdata") is not None:
            return int(pt["customdata"])

    return None


# ── Raster plot ───────────────────────────────────────────────────────────────
@app.callback(
    Output("raster-plot", "figure"),
    Input("store-data", "data"),
    Input("store-selected-t", "data"),
)
def draw_raster(data, sel_t):
    if data is None:
        return go.Figure()

    spins = np.array(data["spins"])   # (T, 120)
    bump  = np.array(data["bump"])
    T     = data["n"]

    fig = go.Figure()

    # Spin heatmap — white=1 (active), black=0 (inactive)
    fig.add_trace(go.Heatmap(
        z=spins.T,                         # shape (120, T) → rows=units, cols=time
        x=list(range(T)),
        y=list(range(120)),
        colorscale=[[0, "#0d1117"], [1, "#e6edf3"]],
        showscale=False,
        hovertemplate="t=%{x}  unit=%{y}  spin=%{z}<extra></extra>",
        zmin=0, zmax=1,
    ))

    # Bump angle overlay — mapped from [-180, 180] → [0, 119]
    bump_unit = (bump + 180) / 360 * 119
    fig.add_trace(go.Scatter(
        x=list(range(T)),
        y=bump_unit.tolist(),
        mode="lines",
        line={"color": RED, "width": 2},
        name="bump angle",
        hovertemplate="t=%{x}  bump=%{customdata:.1f}°<extra></extra>",
        customdata=bump.tolist(),
    ))

    # Highlighted timestep vertical line
    shapes = []
    annotations = []
    if sel_t is not None and 0 <= sel_t < T:
        shapes.append(dict(
            type="line",
            x0=sel_t, x1=sel_t, y0=0, y1=119,
            line={"color": YELLOW, "width": 2, "dash": "dot"},
            layer="above",
        ))
        annotations.append(dict(
            x=sel_t, y=119,
            text=f"t={sel_t}",
            showarrow=False,
            font={"color": YELLOW, "size": 11},
            xanchor="left", yanchor="top",
        ))

    fig.update_layout(
        margin={"l": 52, "r": 16, "t": 12, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "JetBrains Mono, monospace", "size": 11},
        xaxis={
            "title": "Timestep",
            "color": MUTED,
            "gridcolor": BORDER,
            "showgrid": True,
            "zeroline": False,
        },
        yaxis={
            "title": "Unit index / bump angle (mapped)",
            "color": MUTED,
            "gridcolor": BORDER,
            "showgrid": False,
            "zeroline": False,
            "range": [0, 119],
        },
        legend={
            "x": 1, "y": 1, "xanchor": "right",
            "bgcolor": "rgba(0,0,0,0)", "font": {"size": 11},
        },
        shapes=shapes,
        annotations=annotations,
        clickmode="event",
    )

    return fig


# ── Trajectory plot ───────────────────────────────────────────────────────────
@app.callback(
    Output("traj-plot", "figure"),
    Output("timestep-badge", "children"),
    Input("store-data", "data"),
    Input("store-selected-t", "data"),
)
def draw_traj(data, sel_t):
    if data is None:
        return go.Figure(), "t = —"

    T          = data["n"]
    t_idx      = list(range(T))
    robot_x    = data["robot_x"]
    robot_y    = data["robot_y"]
    speakerL_x = data["speakerL_x"]
    speakerL_y = data["speakerL_y"]
    speakerR_x = data["speakerR_x"]
    speakerR_y = data["speakerR_y"]

    # Colour by timestep index (viridis-like continuous scale)
    color_vals = t_idx

    def scatter_trace(xs, ys, name, symbol, colorscale, line_color):
        return go.Scatter(
            x=xs, y=ys,
            mode="markers",
            name=name,
            marker={
                "symbol": symbol,
                "size": 8,
                "color": color_vals,
                "colorscale": colorscale,
                "showscale": False,
                "line": {"color": line_color, "width": 0.5},
            },
            customdata=t_idx,
            hovertemplate=f"<b>{name}</b><br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>t=%{{customdata}}<extra></extra>",
        )

    fig = go.Figure()

    fig.add_trace(scatter_trace(robot_x,    robot_y,    "Robot",     "circle",  "Blues",   ACCENT))
    fig.add_trace(scatter_trace(speakerL_x, speakerL_y, "Speaker L", "diamond", "Greens",  GREEN))
    fig.add_trace(scatter_trace(speakerR_x, speakerR_y, "Speaker R", "square",  "Oranges", YELLOW))

    # Add a shared colorbar for timestep
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker={
            "color": [0, T - 1],
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {
                "title": {"text": "Timestep", "font": {"color": MUTED, "size": 11}},
                "tickfont": {"color": MUTED, "size": 10},
                "thickness": 12,
                "len": 0.7,
                "x": 1.02,
            },
        },
        showlegend=False,
        hoverinfo="skip",
    ))

    # Highlight selected timestep
    shapes = []
    badge  = "t = —"
    if sel_t is not None and 0 <= sel_t < T:
        badge = f"t = {sel_t}"

        def hl_marker(xs, ys, symbol):
            return go.Scatter(
                x=[xs[sel_t]], y=[ys[sel_t]],
                mode="markers",
                marker={
                    "symbol": symbol,
                    "size": 16,
                    "color": "#ffffff",
                    "line": {"color": YELLOW, "width": 2.5},
                },
                showlegend=False,
                hoverinfo="skip",
            )

        fig.add_trace(hl_marker(robot_x,    robot_y,    "circle"))
        fig.add_trace(hl_marker(speakerL_x, speakerL_y, "diamond"))
        fig.add_trace(hl_marker(speakerR_x, speakerR_y, "square"))

    fig.update_layout(
        margin={"l": 52, "r": 80, "t": 12, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "JetBrains Mono, monospace", "size": 11},
        xaxis={
            "title": "X (m)",
            "color": MUTED,
            "gridcolor": BORDER,
            "showgrid": True,
            "zeroline": False,
            "scaleanchor": "y",
        },
        yaxis={
            "title": "Y (m)",
            "color": MUTED,
            "gridcolor": BORDER,
            "showgrid": True,
            "zeroline": False,
        },
        legend={
            "x": 0, "y": 1, "xanchor": "left",
            "bgcolor": "rgba(22,27,34,0.85)",
            "bordercolor": BORDER,
            "borderwidth": 1,
            "font": {"size": 11},
        },
        shapes=shapes,
        clickmode="event",
    )

    return fig, badge


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nFound {len(csv_files)} result file(s) in: {os.path.abspath(RESULTS_DIR)}")
    for p in csv_files:
        print(f"  • {os.path.basename(p)}")
    print(f"\nOpen http://127.0.0.1:{args.port}\n")
    app.run(debug=False, port=args.port)