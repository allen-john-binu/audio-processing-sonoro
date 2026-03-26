"""
Created on Wed May  8 09:44:27 2024
@author: theja

Modified version that:
1. Plays sound directly without saving files
2. Generates CSV with global timestamp, left level, and right level
"""
import scipy.signal as signal 
import numpy as np 
import matplotlib.pyplot as plt 
import sounddevice as sd
import csv
from datetime import datetime
import time

# %%
# Make sweeps
durns = np.array([3, 4, 5, 8, 10]) * 1e-3
fs = 192000  # Hz
all_sweeps = []

for durn in durns:
    t = np.linspace(0, durn, int(fs * durn))
    start_f, end_f = 100, 24e3
    sweep = signal.chirp(t, start_f, t[-1], end_f)
    sweep *= signal.windows.tukey(sweep.size, 0.2)
    sweep *= 0.8
    sweep_padded = np.pad(sweep, pad_width=[int(fs*0.1)]*2, constant_values=[0, 0])
    all_sweeps.append(sweep_padded)

sweeps_combined = np.concatenate(all_sweeps)

# Create target 10-minute signal
target_duration = 0.5 * 60  # 10 minutes in seconds
current_duration = len(sweeps_combined) / fs
repeats = int(np.ceil(target_duration / current_duration))
long_signal = np.tile(sweeps_combined, repeats)
long_signal = long_signal[:int(fs * target_duration)]

# %%
# Create stereo signal with different left/right volumes
left_volume = 1.0
right_volume = 0.2
stereo_signal = np.column_stack((long_signal * left_volume, long_signal * right_volume))

# %%
# Create CSV with timestamp and sound levels
csv_filename = 'sound_levels.csv'

# Determine sampling interval for CSV (every N samples)
# Options: every sample (slow, large file) or decimated
csv_sample_interval = 1920  # Every 1920 samples = 10ms at 192kHz

# Create header
csv_data = []
csv_data.append(['timestamp_iso', 'sample_index', 'time_seconds', 'left_level', 'right_level'])

# Extract and record levels
start_time = datetime.now()
for idx in range(0, len(stereo_signal), csv_sample_interval):
    elapsed_seconds = idx / fs
    current_time = datetime.now()
    time_iso = current_time.isoformat()
    left_level = abs(stereo_signal[idx, 0])
    right_level = abs(stereo_signal[idx, 1])
    
    csv_data.append([
        time_iso,
        idx,
        f'{elapsed_seconds:.6f}',
        f'{left_level:.6f}',
        f'{right_level:.6f}'
    ])

# Write CSV file
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print(f"CSV file created: {csv_filename}")
print(f"Total rows: {len(csv_data) - 1}")  # Subtract header
print(f"Time interval between entries: {csv_sample_interval / fs * 1000:.2f} ms ({csv_sample_interval / fs:.6f} seconds)")

# %%
# Play the stereo sound directly
print(f"\nPlaying {target_duration}s of audio...")
print(f"Sample rate: {fs} Hz")
print(f"Left volume: {left_volume}, Right volume: {right_volume}")

sd.play(stereo_signal, fs)
sd.wait()

print("Playback finished!")