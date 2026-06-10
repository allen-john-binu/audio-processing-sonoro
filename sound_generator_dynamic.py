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
fs = 48000
all_sweeps = []

durn = 5e-3  # 5 ms

t = np.linspace(0, durn, int(fs * durn), endpoint=False)
sweep = signal.chirp(t, 2000, t[-1], 24000)
sweep *= signal.windows.tukey(len(sweep), 0.2)
sweep *= 0.8
sweep = np.pad(sweep, (int(fs * 0.1), int(fs * 0.1)))

sweeps_combined = sweep

target_duration = 80  # seconds
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

# delta = 0.15      # try: 0.025, 0.05, 0.1, 0.15
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
sigma = 0.2  # try: 0.025, 0.05, 0.1, 0.2

base_left = 0.6
base_right = 0.6

# =========================
# Generate noise (once)
# =========================
np.random.seed(0)  # optional (reproducible)

noise_left = np.random.normal(0, sigma, size=len(long_signal))
noise_right = np.random.normal(0, sigma, size=len(long_signal))

# =========================
# Volume envelopes (arrays)
# =========================
left_volumes = np.clip(base_left + noise_left, 0.0, 1.0)
right_volumes = np.clip(base_right + noise_right, 0.0, 1.0)
# =========================
# Constant volumes
# =========================
# =========================
# Constant channel volumes
# =========================
# base_left = 1.0

# base_right = 1.0

# left_volumes = np.full(len(long_signal), base_left)
# right_volumes = np.full(len(long_signal), base_right)

stereo_signal = np.column_stack((
    long_signal * left_volumes,
    long_signal * right_volumes
))

# =========================
# CSV setup
# =========================
# Hardcoded folders
OUTPUT_DIR = "ztLabCollection/dataFromSound"  
EMPTY_COPY_DIR = "ztLabCollection/dataFromReal" 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMPTY_COPY_DIR, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
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
    print("Writing CSV...")

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'sample_index', 'time_seconds', 'left_volume', 'right_volume'])
        writer.writerows(csv_rows)

    print(f"CSV saved: {csv_filename}")
    print(f"Total rows: {len(csv_rows)}")
    
    # Create an empty CSV with the same filename
    with open(empty_csv_filename, 'w', newline='') as f:
        pass

    print(f"Empty CSV created: {empty_csv_filename}")
    
        # =========================
    # Plot saved volume data
    # =========================
    if csv_rows:
        timestamps = [row[0] for row in csv_rows]
        left_vals = [float(row[3]) for row in csv_rows]
        right_vals = [float(row[4]) for row in csv_rows]

        plt.figure(figsize=(12, 5))
        plt.plot(left_vals, label="Left Volume", color="blue")
        plt.plot(right_vals, label="Right Volume", color="red")

        plt.title("Input Volume Envelopes")
        plt.xlabel("Logged Sample")
        plt.ylabel("Volume")
        plt.grid(True)
        plt.legend()
        plt.show()
