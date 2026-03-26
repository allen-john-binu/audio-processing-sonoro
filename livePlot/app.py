import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, jsonify
import json
import os
import sys

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Get absolute path to data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "../dataFromReal")

print(f"\n{'='*60}")
print(f"[INFO] Flask application starting...")
print(f"[INFO] Current directory: {current_dir}")
print(f"[INFO] Data directory: {DATA_DIR}")
print(f"{'='*60}\n")

# Check if data directory exists
if not os.path.exists(DATA_DIR):
    print(f"[ERROR] Data directory not found!")
    print(f"[ERROR] Expected location: {DATA_DIR}")
    sys.exit(1)

# Scan for CSV files
def get_csv_files():
    """Get list of CSV files in data directory, sorted alphabetically"""
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        files.sort()
        print(f"[INFO] Found {len(files)} CSV files:")
        for f in files:
            print(f"       - {f}")
        return files
    except Exception as e:
        print(f"[ERROR] Failed to scan directory: {e}")
        return []

csv_files = get_csv_files()

if not csv_files:
    print(f"[ERROR] No CSV files found in {DATA_DIR}")
    sys.exit(1)

# Use first file as default
DEFAULT_FILE = csv_files[0]
print(f"[INFO] Using default file: {DEFAULT_FILE}\n")

# ============================================================
# DATA LOADING FUNCTION
# ============================================================
def load_csv_file(filename):
    """Load and parse CSV file, return all processed data"""
    try:
        # Security check: ensure filename is in data directory
        if ".." in filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid filename")
        
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"[INFO] Loading: {filename}")
        df = pd.read_csv(file_path)
        
        # Parse data
        angle_cols = df.columns[2:]
        angles = np.array([float(a) for a in angle_cols])
        spl_array = df["dB_SPL"].astype(str).str.strip("[]").astype(float).values
        intensity_matrix = df.iloc[:, 2:].values.astype(float)
        num_steps = intensity_matrix.shape[0]
        
        # Normalization
        int_min, int_max = intensity_matrix.min(), intensity_matrix.max()
        spl_min, spl_max = spl_array.min(), spl_array.max()
        
        print(f"[INFO] Shape: {df.shape}")
        print(f"[INFO] SPL range: {spl_min:.2f} - {spl_max:.2f} dB")
        print(f"[INFO] Intensity range: {int_min:.2f} - {int_max:.2f}")
        
        # Angle spacing fix
        angle_step = angles[1] - angles[0]
        y_min = angles[0] - angle_step / 2
        y_max = angles[-1] + angle_step / 2
        
        # Peak detection for full dataset
        peak_indices_full = np.argmax(intensity_matrix, axis=1)
        peak_angles_full = angles[peak_indices_full]
        time_steps_full = np.arange(num_steps)
        
        return {
            'df': df,
            'angles': angles,
            'spl_array': spl_array,
            'intensity_matrix': intensity_matrix,
            'num_steps': num_steps,
            'int_min': int_min,
            'int_max': int_max,
            'spl_min': spl_min,
            'spl_max': spl_max,
            'angle_step': angle_step,
            'y_min': y_min,
            'y_max': y_max,
            'peak_indices_full': peak_indices_full,
            'peak_angles_full': peak_angles_full,
            'time_steps_full': time_steps_full
        }
    except Exception as e:
        print(f"[ERROR] Failed to load {filename}: {e}")
        raise

# Load default file
data = load_csv_file(DEFAULT_FILE)

# Unpack data
df = data['df']
angles = data['angles']
spl_array = data['spl_array']
intensity_matrix = data['intensity_matrix']
num_steps = data['num_steps']
int_min = data['int_min']
int_max = data['int_max']
spl_min = data['spl_min']
spl_max = data['spl_max']
angle_step = data['angle_step']
y_min = data['y_min']
y_max = data['y_max']
peak_indices_full = data['peak_indices_full']
peak_angles_full = data['peak_angles_full']
time_steps_full = data['time_steps_full']

print(f"[INFO] Data preparation complete!")
print(f"{'='*60}\n")


# ============================================================
# FILTERING FUNCTION
# ============================================================
def filter_data(threshold):
    """Filter data by SPL threshold and return compressed data"""
    mask = spl_array >= threshold
    
    filtered_intensity = intensity_matrix[mask]
    filtered_spl = spl_array[mask]
    
    if len(filtered_intensity) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    peak_indices = np.argmax(filtered_intensity, axis=1)
    peak_angles = angles[peak_indices]
    new_time = np.arange(len(filtered_intensity))
    
    return filtered_intensity, peak_angles, new_time, filtered_spl


# ============================================================
# PLOTLY FIGURE GENERATION
# ============================================================

def create_filtered_figure(threshold):
    """Create Plotly figure for Graph 1 (filtered, interactive)"""
    filtered_intensity, peak_angles, time_steps, filtered_spl = filter_data(threshold)
    
    if len(filtered_intensity) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data passes this threshold", showarrow=False)
        return fig
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.85, 0.15],
        subplot_titles=("Filtered DOA Intensity + Peak", "SPL Values")
    )
    
    # ===== HEATMAP (Graph 1, top) =====
    # Use full dataset range for color scaling (not filtered data range)
    heatmap = go.Heatmap(
        z=filtered_intensity.T,
        x=time_steps,
        y=angles,
        colorscale="Inferno",
        zmin=int_min,  # Use full dataset minimum
        zmax=int_max,  # Use full dataset maximum
        colorbar=dict(title="Intensity", len=0.7, y=0.65),
        hovertemplate="Time: %{x}<br>Angle: %{y:.1f}°<br>Intensity: %{z:.2f}<extra></extra>",
        name="",
        showscale=True
    )
    fig.add_trace(heatmap, row=1, col=1)
    
    # ===== PEAK SCATTER =====
    scatter = go.Scatter(
        x=time_steps,
        y=peak_angles,
        mode="markers",
        marker=dict(
            color="cyan",
            size=6,
            line=dict(color="white", width=0.5)
        ),
        hovertemplate="Time: %{x}<br>Peak Angle: %{y:.1f}°<extra></extra>",
        name="Peak DOA",
        showlegend=True
    )
    fig.add_trace(scatter, row=1, col=1)
    
    # ===== SPL STRIP (Graph 1, bottom) =====
    # Use full dataset range for color scaling (not filtered data range)
    spl_image = filtered_spl.reshape(1, -1)
    spl_heatmap = go.Heatmap(
        z=spl_image,
        x=time_steps,
        y=[0],
        colorscale="Viridis",
        zmin=spl_min,  # Use full dataset minimum
        zmax=spl_max,  # Use full dataset maximum
        showscale=True,
        colorbar=dict(title="dB SPL", len=0.15, y=0.08),
        hovertemplate="Time: %{x}<br>SPL: %{z:.2f} dB<extra></extra>",
        name="",
        yaxis="y2"
    )
    fig.add_trace(spl_heatmap, row=2, col=1)
    
    fig.update_xaxes(title_text="Time Step (Filtered)", row=2, col=1)
    fig.update_yaxes(title_text="Angle (degrees)", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    
    fig.update_layout(
        height=600,
        title_text=f"Graph 1: Filtered DOA (SPL threshold ≥ {threshold:.2f} dB)",
        hovermode="closest",
        showlegend=True,
        template="plotly_white"
    )
    
    return fig


def create_static_figure():
    """Create Plotly figure for Graph 2 (full dataset, static)"""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.85, 0.15],
        subplot_titles=("Full DOA Intensity + Peak", "SPL Values")
    )
    
    # ===== HEATMAP (Graph 2, top) =====
    heatmap = go.Heatmap(
        z=intensity_matrix.T,
        x=time_steps_full,
        y=angles,
        colorscale="Inferno",
        zmin=int_min,
        zmax=int_max,
        colorbar=dict(title="Intensity", len=0.7, y=0.65),
        hovertemplate="Time: %{x}<br>Angle: %{y:.1f}°<br>Intensity: %{z:.2f}<extra></extra>",
        name="",
        showscale=True
    )
    fig.add_trace(heatmap, row=1, col=1)
    
    # ===== PEAK SCATTER =====
    scatter = go.Scatter(
        x=time_steps_full,
        y=peak_angles_full,
        mode="markers",
        marker=dict(
            color="cyan",
            size=6,
            line=dict(color="white", width=0.5)
        ),
        hovertemplate="Time: %{x}<br>Peak Angle: %{y:.1f}°<extra></extra>",
        name="Peak DOA",
        showlegend=True
    )
    fig.add_trace(scatter, row=1, col=1)
    
    # ===== SPL STRIP (Graph 2, bottom) =====
    spl_image = spl_array.reshape(1, -1)
    spl_heatmap = go.Heatmap(
        z=spl_image,
        x=time_steps_full,
        y=[0],
        colorscale="Viridis",
        zmin=spl_min,
        zmax=spl_max,
        showscale=True,
        colorbar=dict(title="dB SPL", len=0.15, y=0.08),
        hovertemplate="Time: %{x}<br>SPL: %{z:.2f} dB<extra></extra>",
        name=""
    )
    fig.add_trace(spl_heatmap, row=2, col=1)
    
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_yaxes(title_text="Angle (degrees)", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    
    fig.update_layout(
        height=600,
        title_text="Graph 2: Full DOA Dataset (No Filtering)",
        hovermode="closest",
        showlegend=True,
        template="plotly_white"
    )
    
    return fig


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    """Main page - render dashboard"""
    try:
        static_fig = create_static_figure()
        static_html = static_fig.to_html(full_html=False, include_plotlyjs=False)
        
        filtered_fig = create_filtered_figure(spl_min)
        filtered_html = filtered_fig.to_html(full_html=False, include_plotlyjs=False)
        
        return render_template(
            'index.html',
            csv_files=csv_files,
            current_file=DEFAULT_FILE,
            spl_min=spl_min,
            spl_max=spl_max,
            spl_init=spl_min,
            filtered_plot=filtered_html,
            static_plot=static_html
        )
    except Exception as e:
        print(f"[ERROR] Failed to render index: {e}")
        return f"<h1>Error</h1><p>{e}</p>", 500


@app.route('/get_files')
def get_files():
    """Return list of available CSV files"""
    try:
        return jsonify({
            'files': csv_files,
            'current_file': DEFAULT_FILE
        })
    except Exception as e:
        print(f"[ERROR] Failed to get files: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/load_csv', methods=['POST'])
def load_csv():
    """Load a new CSV file and return updated data and plots"""
    global df, angles, spl_array, intensity_matrix, num_steps
    global int_min, int_max, spl_min, spl_max, angle_step, y_min, y_max
    global peak_indices_full, peak_angles_full, time_steps_full, DEFAULT_FILE
    
    try:
        request_data = request.get_json()
        filename = request_data.get('filename')
        
        print(f"\n[INFO] Loading CSV file: {filename}")
        
        # Load new file
        new_data = load_csv_file(filename)
        
        # Update global variables
        df = new_data['df']
        angles = new_data['angles']
        spl_array = new_data['spl_array']
        intensity_matrix = new_data['intensity_matrix']
        num_steps = new_data['num_steps']
        int_min = new_data['int_min']
        int_max = new_data['int_max']
        spl_min = new_data['spl_min']
        spl_max = new_data['spl_max']
        angle_step = new_data['angle_step']
        y_min = new_data['y_min']
        y_max = new_data['y_max']
        peak_indices_full = new_data['peak_indices_full']
        peak_angles_full = new_data['peak_angles_full']
        time_steps_full = new_data['time_steps_full']
        DEFAULT_FILE = filename
        
        # Create new plots
        static_fig = create_static_figure()
        static_html = static_fig.to_html(full_html=False, include_plotlyjs=False)
        
        filtered_fig = create_filtered_figure(spl_min)
        filtered_html = filtered_fig.to_html(full_html=False, include_plotlyjs=False)
        
        print(f"[SUCCESS] File loaded and plots generated\n")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'spl_min': float(spl_min),
            'spl_max': float(spl_max),
            'filtered_plot': filtered_html,
            'static_plot': static_html
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/update_filtered', methods=['POST'])
def update_filtered():
    """AJAX endpoint to update Graph 1 based on slider value"""
    try:
        data = request.get_json()
        threshold = float(data.get('threshold', spl_min))
        
        print(f"[INFO] Update requested for threshold: {threshold:.2f} dB")
        
        threshold = max(spl_min, min(spl_max, threshold))
        
        fig = create_filtered_figure(threshold)
        html = fig.to_html(full_html=False, include_plotlyjs=False)
        
        print(f"[INFO] Generated HTML of length: {len(html)}")
        
        return html, 200, {'Content-Type': 'text/html; charset=utf-8'}
        
    except Exception as e:
        print(f"[ERROR] Failed to update filtered plot: {e}")
        import traceback
        traceback.print_exc()
        return f"<p style='color: red;'>Error: {str(e)}</p>", 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("[INFO] Starting Flask server on http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=True)
