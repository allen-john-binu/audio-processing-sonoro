import scipy.signal as signal
import numpy as np
import sounddevice as sd
import csv
from datetime import datetime, timedelta
import os
import sys

import matplotlib.pyplot as plt

# =========================
# Generate sweeps
# =========================
fs = 16000

all_sweeps = []

durn = 5e-3  # 5 ms

wait_time = 0.5  # seconds

t = np.linspace(0, durn, int(fs * durn), endpoint=False)
sweep = signal.chirp(t, 500, t[-1], 3500)
sweep *= signal.windows.tukey(len(sweep), 0.2)
sweep *= 0.8
sweep = np.pad(sweep, (int(fs * wait_time), int(fs * wait_time)))

sweeps_combined = sweep

target_duration = 1200  # seconds
repeats = int(np.ceil(target_duration / (len(sweeps_combined) / fs)))
long_signal = np.tile(sweeps_combined, repeats)
long_signal = long_signal[:int(fs * target_duration)]

# =========================
# Volume envelopes
# =========================
# def left_volume_envelope(t):
#     return 0.58

# def right_volume_envelope(t):
#     return 1.0

# delta = 0.10      # try: 0.025, 0.05, 0.1, 0.15
# mod_freq = 0.2  # Hz

# base_left = 0.7
# base_right = 0.7

# def left_volume_envelope(t):
#     return base_left + delta * np.sin(2 * np.pi * mod_freq * t)

# def right_volume_envelope(t):
#     return base_right - delta * np.sin(2 * np.pi * mod_freq * t)

# time_vector = np.arange(len(long_signal)) / fs
# left_volumes = np.array([left_volume_envelope(t) for t in time_vector])
# right_volumes = np.array([right_volume_envelope(t) for t in time_vector])

# =========================
# Gaussian parameters
# =========================
# sigma = 0.2  # try: 0.025, 0.05, 0.1, 0.2

# base_left = 0.6
# base_right = 0.6

# # =========================
# # Generate noise (once)
# # =========================
# np.random.seed(0)  # optional (reproducible)

# noise_left = np.random.normal(0, sigma, size=len(long_signal))
# noise_right = np.random.normal(0, sigma, size=len(long_signal))

# # =========================
# # Volume envelopes (arrays)
# # =========================
# left_volumes = np.clip(base_left + noise_left, 0.0, 1.0)
# right_volumes = np.clip(base_right + noise_right, 0.0, 1.0)
# # =========================
# Constant volumes
# =========================
# =========================
# Constant channel volumes
# =========================
base_left = 1.0

base_right = 0.0 #real left

left_volumes = np.full(len(long_signal), base_left)
right_volumes = np.full(len(long_signal), base_right)

stereo_signal = np.column_stack((
    long_signal * left_volumes,
    long_signal * right_volumes
))

# =========================
# CSV setup
# =========================
# Hardcoded folders
OUTPUT_DIR = "ztFinalLabRealWithCam/dataFromSound/"  
EMPTY_COPY_DIR = "ztFinalLabRealWithCam/dataFromReal/" 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMPTY_COPY_DIR, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python script.py <filename> [--plot-only]")
    sys.exit(1)

filename = sys.argv[1]
plot_only = "--plot-only" in sys.argv

if not filename.lower().endswith(".csv"):
    filename += ".csv"

csv_filename = os.path.join(OUTPUT_DIR, filename)
empty_csv_filename = os.path.join(EMPTY_COPY_DIR, filename)

csv_rows = []
csv_sample_interval = 1920  # 10 ms

# =========================
# Callback state
# =========================
sample_index = 0
start_wall_time = None

# =========================
# Audio callback
# =========================
def callback(outdata, frames, time, status):
    global sample_index, start_wall_time

    if status:
        print(status)

    chunk = stereo_signal[sample_index:sample_index + frames]

    if len(chunk) < frames:
        outdata[:len(chunk)] = chunk
        outdata[len(chunk):] = 0
        raise sd.CallbackStop()
    else:
        outdata[:] = chunk

    # Initialize time reference once
    if start_wall_time is None:
        start_wall_time = datetime.now() - timedelta(seconds=time.outputBufferDacTime)

    # Log samples
    for i in range(0, frames, csv_sample_interval):
        idx = sample_index + i
        if idx >= len(stereo_signal):
            break

        dac_time = time.outputBufferDacTime + (i / fs)
        wall_time = start_wall_time + timedelta(seconds=dac_time)

        csv_rows.append([
            wall_time.isoformat(),
            idx,
            f"{idx / fs:.6f}",
            f"{left_volumes[idx]:.6f}",
            f"{right_volumes[idx]:.6f}"
        ])

    sample_index += frames

# =========================
# Run with Ctrl+C handling
# =========================
print("Starting playback... Press Ctrl+C to stop and save CSV.")

try:
    with sd.OutputStream(
        samplerate=fs,
        channels=2,
        callback=callback,
        blocksize=0  # let sounddevice choose optimal
    ):
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:

    if not plot_only:
        print("Writing CSV...")

        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'sample_index',
                'time_seconds',
                'left_volume',
                'right_volume'
            ])
            writer.writerows(csv_rows)

        print(f"CSV saved: {csv_filename}")
        print(f"Total rows: {len(csv_rows)}")

        with open(empty_csv_filename, 'w', newline='') as f:
            pass        
        print(f"Empty CSV created: {empty_csv_filename}")

    else:
        print("Plot-only mode: no files saved.")
        
    plot_dont = False
    
    # =========================
    # Plot saved volume data
    # =========================
    # =========================
    # Plot saved volume data
    # =========================
if csv_rows and not plot_dont:
    times = [float(row[2]) for row in csv_rows]
    left_vals = [float(row[3]) for row in csv_rows]
    right_vals = [float(row[4]) for row in csv_rows]

    plt.figure(figsize=(12, 5))

    plt.plot(times, left_vals, label="Left Volume", color="blue")
    plt.plot(times, right_vals, label="Right Volume", color="red")

    plt.xlabel("Time (s)",
            fontsize=18)
    plt.ylabel("amplitude scaling factor",
            fontsize=18)
    plt.ylim(0, 1.2)

    plt.grid(True)
    plt.legend()

    # -----------------------------
    # Double-headed vertical arrow
    # -----------------------------
    arrow_x = 3.75

    # Actual values of the two curves at arrow_x
    left_at_x = np.interp(arrow_x, times, left_vals)
    right_at_x = np.interp(arrow_x, times, right_vals)

    arrow_bottom = min(left_at_x, right_at_x)
    arrow_top = max(left_at_x, right_at_x)

    plt.annotate(
        "",
        xy=(arrow_x, arrow_top),
        xytext=(arrow_x, arrow_bottom),
        arrowprops=dict(
            arrowstyle="<->",
            color="steelblue",
            linewidth=2.5,
            mutation_scale=18
        )
    )

    # Label "a"
    plt.text(
        arrow_x - 0.05,
        (arrow_top + arrow_bottom) / 2,
        "d",
        fontsize=32,
        va="center",
        ha="right"
    )

    print("Plot saved.")
    plt.savefig(
        "constant_modulation.pdf",
        bbox_inches="tight",
        format="pdf"
    )
    plt.show()
