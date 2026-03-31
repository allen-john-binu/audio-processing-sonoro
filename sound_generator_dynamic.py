import scipy.signal as signal
import numpy as np
import sounddevice as sd
import csv
from datetime import datetime, timedelta

# =========================
# Generate sweeps
# =========================
durns = np.array([3, 4, 5, 8, 10]) * 1e-3
fs = 192000
all_sweeps = []

for durn in durns:
    t = np.linspace(0, durn, int(fs * durn), endpoint=False)
    sweep = signal.chirp(t, 100, t[-1], 24000)
    sweep *= signal.windows.tukey(len(sweep), 0.2)
    sweep *= 0.8
    sweep = np.pad(sweep, (int(fs * 0.1), int(fs * 0.1)))
    all_sweeps.append(sweep)

sweeps_combined = np.concatenate(all_sweeps)

target_duration = 120  # seconds
repeats = int(np.ceil(target_duration / (len(sweeps_combined) / fs)))
long_signal = np.tile(sweeps_combined, repeats)
long_signal = long_signal[:int(fs * target_duration)]

# =========================
# Volume envelopes
# =========================
def left_volume_envelope(t):
    return 0.70

def right_volume_envelope(t):
    return 1.0

time_vector = np.arange(len(long_signal)) / fs
left_volumes = np.array([left_volume_envelope(t) for t in time_vector])
right_volumes = np.array([right_volume_envelope(t) for t in time_vector])

stereo_signal = np.column_stack((
    long_signal * left_volumes,
    long_signal * right_volumes
))

# =========================
# CSV setup
# =========================
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f'dataFromSound/sound_levels_{timestamp_str}.csv'

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