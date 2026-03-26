import scipy.signal as signal 
import numpy as np 
import sounddevice as sd
import csv
from datetime import datetime

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

# Create target duration signal
target_duration = 0.5 * 60  # 30 seconds (0.5 * 60)
current_duration = len(sweeps_combined) / fs
repeats = int(np.ceil(target_duration / current_duration))
long_signal = np.tile(sweeps_combined, repeats)
long_signal = long_signal[:int(fs * target_duration)]

# %%
# Define dynamic volume envelopes for left and right channels
# These functions take time (in seconds) and return volume (0.0 to 1.0)

def left_volume_envelope(t):
    """
    Example: Left volume increases linearly from 0.2 to 1.0 over time
    """
    return np.clip(0.2 + (t / target_duration) * 0.8, 0.0, 1.0)

def right_volume_envelope(t):
    """
    Example: Right volume is a sine wave modulation
    """
    return 0.2 + 0.15 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave

# You can customize these functions. Examples:
# Constant volume:
#   return 0.5
# Step change at specific time:
#   return 0.3 if t < 15 else 0.7
# Exponential fade:
#   return 1.0 * np.exp(-t / 10)
# Cosine fade:
#   return 0.5 * (1 + np.cos(np.pi * t / target_duration))

# %%
# Apply dynamic volume envelopes to the signal
time_vector = np.arange(len(long_signal)) / fs
left_volumes = np.array([left_volume_envelope(t) for t in time_vector])
right_volumes = np.array([right_volume_envelope(t) for t in time_vector])

# Create stereo signal with dynamic volumes
stereo_signal = np.column_stack((
    long_signal * left_volumes,
    long_signal * right_volumes
))

# %%
# Create CSV with timestamp and sound levels
current_datetime = datetime.now()
timestamp_str = current_datetime.strftime('%Y%m%d_%H%M%S')
csv_filename = f'dataFromSound/sound_levels_{timestamp_str}.csv'

# Determine sampling interval for CSV (every N samples)
csv_sample_interval = 1920  # Every 1920 samples = 10ms at 192kHz

# Create header
csv_data = []
csv_data.append(['timestamp_iso', 'sample_index', 'time_seconds', 'left_level', 'right_level', 'left_volume', 'right_volume'])

# Extract and record levels
for idx in range(0, len(stereo_signal), csv_sample_interval):
    elapsed_seconds = idx / fs
    current_time = datetime.now()
    time_iso = current_time.isoformat()
    left_level = abs(stereo_signal[idx, 0])
    right_level = abs(stereo_signal[idx, 1])
    left_vol = left_volumes[idx]
    right_vol = right_volumes[idx]
    
    csv_data.append([
        time_iso,
        idx,
        f'{elapsed_seconds:.6f}',
        f'{left_level:.6f}',
        f'{right_level:.6f}',
        f'{left_vol:.6f}',
        f'{right_vol:.6f}'
    ])

# Write CSV file
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print(f"CSV file created: {csv_filename}")
print(f"Total rows: {len(csv_data) - 1}")  # Subtract header
print(f"Time interval between entries: {csv_sample_interval / fs * 1000:.2f} ms ({csv_sample_interval / fs:.6f} seconds)")
print(f"Left volume range: {left_volumes.min():.4f} to {left_volumes.max():.4f}")
print(f"Right volume range: {right_volumes.min():.4f} to {right_volumes.max():.4f}")

# %%
# Play the stereo sound directly
print(f"\nPlaying {target_duration:.1f}s of audio with dynamic volumes...")
print(f"Sample rate: {fs} Hz")

sd.play(stereo_signal, fs)
sd.wait()

print("Playback finished!")
