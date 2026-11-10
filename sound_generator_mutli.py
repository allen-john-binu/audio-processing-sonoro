
import scipy.signal as signal
import numpy as np
import sounddevice as sd
import csv
from datetime import datetime, timedelta
import os
import sys
import matplotlib.pyplot as plt

# =========================
# Audio parameters
# =========================
fs = 48000.0

durn = 5e-3       # Chirp duration: 5 ms
wait_time = 0.4   # Padding before and after chirp: 400 ms

# Speaker alternation parameters
play_duration = 3.0
silent_duration = 0
cycle_duration = 2 * (play_duration + silent_duration)

target_duration = 30  # Total playback duration: 1200 seconds

base_left = 1.0
base_right = 1.0

# =========================
# Generate chirp waveform
# =========================
t = np.linspace(
    0, durn, int(fs * durn), endpoint=False
)

sweep = signal.chirp(t, 500, t[-1], 3500)
sweep *= signal.windows.tukey(len(sweep), 0.2)

sweep = np.pad(
    sweep,
    (int(fs * wait_time), int(fs * wait_time))
)

# Repeat the existing chirp waveform for the full duration.
repeats = int(np.ceil(
    
    target_duration / (len(sweep) / fs)
))

long_signal = np.tile(sweep, repeats)
long_signal = long_signal[:int(fs * target_duration)]

# =========================
# Speaker switching
# =========================
play_samples = int(play_duration * fs)
silent_samples = int(silent_duration * fs)

cycle_samples = int(cycle_duration * fs)
target_samples = len(long_signal)

def get_channel_volumes(sample_indices):
    """
    Left:  play for 3 s, then remain silent for 9 s.
    Right: remain silent for 6 s, play for 3 s,
           then remain silent for 3 s.
    """
    phase = sample_indices % cycle_samples

    left_active = phase < play_samples

    right_active = (
        (phase >= play_samples + silent_samples)
        & (phase < play_samples + silent_samples + play_samples)
    )

    left_volumes = left_active.astype(float) * base_left
    right_volumes = right_active.astype(float) * base_right

    return left_volumes, right_volumes

# =========================
# CSV setup
# =========================
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
csv_sample_interval = 1920  # Log every 10 ms

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

    remaining = target_samples - sample_index
    n = min(frames, remaining)

    # If no samples remain, stop playback.
    if n <= 0:
        outdata.fill(0)
        raise sd.CallbackStop()

    # Sample indices for this audio block.
    indices = sample_index + np.arange(n)

    # Get the corresponding chirp samples.
    mono_chunk = long_signal[sample_index:sample_index + n]

    # Determine which speaker is active at each sample.
    left_volumes, right_volumes = get_channel_volumes(indices)

    # Initialize output as silence.
    outdata.fill(0)

    # Write the audio to the appropriate channels.
    outdata[:n, 0] = mono_chunk * left_volumes
    outdata[:n, 1] = mono_chunk * right_volumes

    # Initialize the wall-clock time reference.
    if start_wall_time is None:
        start_wall_time = datetime.now() - timedelta(
            seconds=time.outputBufferDacTime
        )

    # =========================
    # Log every 10 ms
    # =========================
    # Align logging samples globally, rather than restarting
    # the 10 ms interval at every callback.
    first_offset = (
        -sample_index
    ) % csv_sample_interval

    for i in range(first_offset, n, csv_sample_interval):
        idx = sample_index + i

        dac_time = time.outputBufferDacTime + (i / fs)
        wall_time = start_wall_time + timedelta(seconds=dac_time)

        phase = idx % cycle_samples

        left_volume = (
            base_left if phase < play_samples else 0.0
        )

        right_start = play_samples + silent_samples
        right_end = right_start + play_samples

        right_volume = (
            base_right
            if right_start <= phase < right_end
            else 0.0
        )

        csv_rows.append([
            wall_time.isoformat(),
            idx,
            f"{idx / fs:.6f}",
            f"{left_volume:.6f}",
            f"{right_volume:.6f}"
        ])

    sample_index += n

    # Stop automatically when the target duration is reached.
    if n < frames or sample_index >= target_samples:
        raise sd.CallbackStop()

# =========================
# Run playback
# =========================
print("Starting playback...")
print("Sequence: Left 3 s -> Silence 3 s -> Right 3 s -> Silence 3 s")
print("Press Ctrl+C to stop and save CSV.")

try:
    with sd.OutputStream(
        samplerate=fs,
        channels=2,
        callback=callback,
        blocksize=0
    ) as stream:

        while stream.active:
            sd.sleep(100)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:

    if not plot_only:
        print("Writing CSV...")

        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "sample_index",
                "time_seconds",
                "left_volume",
                "right_volume"
            ])
            writer.writerows(csv_rows)

        print(f"CSV saved: {csv_filename}")
        print(f"Total rows: {len(csv_rows)}")

        # Create the corresponding empty CSV file.
        with open(empty_csv_filename, "w", newline=""):
            pass

        print(f"Empty CSV created: {empty_csv_filename}")

    else:
        print("Plot-only mode: no files saved.")

# =========================
# Plot logged volume data
# =========================
if csv_rows:
    times = [float(row[2]) for row in csv_rows]
    left_vals = [float(row[3]) for row in csv_rows]
    right_vals = [float(row[4]) for row in csv_rows]

    plt.figure(figsize=(12, 5))

    plt.plot(
        times, left_vals,
        label="Left Volume", color="blue"
    )
    plt.plot(
        times, right_vals,
        label="Right Volume", color="red"
    )

    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Amplitude scaling factor", fontsize=18)
    plt.ylim(-0.1, 1.2)

    plt.grid(True)
    plt.legend()

    # Double-headed arrow showing the difference
    # between the two volume curves at t = 3.75 s.
    arrow_x = 3.75

    left_at_x = np.interp(arrow_x, times, left_vals)
    right_at_x = np.interp(arrow_x, times, right_vals)

    arrow_bottom = min(left_at_x, right_at_x)
    arrow_top = max(left_at_x, right_at_x)

    if arrow_top > arrow_bottom:
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

        plt.text(
            arrow_x - 0.05,
            (arrow_top + arrow_bottom) / 2,
            "d",
            fontsize=32,
            va="center",
            ha="right"
        )

    plt.savefig(
        "constant_modulation.pdf",
        bbox_inches="tight",
        format="pdf"
    )

    print("Plot saved: constant_modulation.pdf")

