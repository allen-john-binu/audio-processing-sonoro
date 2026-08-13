import scipy.signal as signal
import numpy as np
import sounddevice as sd
import csv
from datetime import datetime, timedelta
import os
import sys

import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

fs = 48000

chirp_duration = 5e-3       # 5 ms
wait_time = 0.6             # 400 ms
target_duration = 300        # 5 minutes

chirp_start_freq = 100
chirp_end_freq = 2900

# Channel volumes
base_left = 1.0
base_right = 1.0


# ============================================================
# Generate one chirp
# ============================================================

num_chirp_samples = int(fs * chirp_duration)

t = np.linspace(
    0,
    chirp_duration,
    num_chirp_samples,
    endpoint=False
)

sweep = signal.chirp(
    t,
    chirp_start_freq,
    chirp_duration,
    chirp_end_freq
)

# Apply Tukey window
sweep *= signal.windows.tukey(
    len(sweep),
    0.2
)

# Overall amplitude
sweep *= 0.8


# ============================================================
# Generate alternating LEFT / RIGHT signal
# ============================================================
#
# Sequence:
#
#   LEFT  chirp
#   silence
#   RIGHT chirp
#   silence
#   LEFT  chirp
#   silence
#   RIGHT chirp
#   ...
#
# Only one channel contains sound at a time.
# ============================================================

wait_samples = int(fs * wait_time)

silence = np.zeros(wait_samples)

# One LEFT event
left_event = np.concatenate([
    sweep * base_left,
    silence
])

# One RIGHT event
right_event = np.concatenate([
    sweep * base_right,
    silence
])

# Alternating stereo events
left_stereo_event = np.column_stack([
    left_event,
    np.zeros(len(left_event))
])

right_stereo_event = np.column_stack([
    np.zeros(len(right_event)),
    right_event
])

# One complete LEFT -> RIGHT cycle
alternating_cycle = np.vstack([
    left_stereo_event,
    right_stereo_event
])


# ============================================================
# Repeat until target duration
# ============================================================

cycle_samples = len(alternating_cycle)

repeats = int(
    np.ceil(
        (target_duration * fs) / cycle_samples
    )
)

stereo_signal = np.tile(
    alternating_cycle,
    (repeats, 1)
)

# Trim to exactly target duration
stereo_signal = stereo_signal[
    :int(fs * target_duration)
]

print("Generated stereo signal:")
print(f"  Sample rate:       {fs} Hz")
print(f"  Target duration:   {target_duration} s")
print(f"  Total samples:     {len(stereo_signal)}")
print(f"  Actual duration:   {len(stereo_signal) / fs:.3f} s")
print(f"  Left volume:       {base_left}")
print(f"  Right volume:      {base_right}")
print()


# ============================================================
# Volume arrays for CSV logging
# ============================================================

left_volumes = np.zeros(len(stereo_signal))
right_volumes = np.zeros(len(stereo_signal))

# Determine which samples belong to LEFT / RIGHT events
event_length = len(left_event)

cycle_length = event_length * 2

for cycle_start in range(
    0,
    len(stereo_signal),
    cycle_length
):

    # LEFT event
    left_start = cycle_start
    left_end = min(
        cycle_start + event_length,
        len(stereo_signal)
    )

    left_volumes[left_start:left_end] = base_left

    # RIGHT event
    right_start = cycle_start + event_length
    right_end = min(
        right_start + event_length,
        len(stereo_signal)
    )

    if right_start < len(stereo_signal):
        right_volumes[right_start:right_end] = base_right


# ============================================================
# CSV setup
# ============================================================

OUTPUT_DIR = "findEqualWithFoamRing/datacollection/dataFromSound"
EMPTY_COPY_DIR = "findEqualWithFoamRing/datacollection/"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    EMPTY_COPY_DIR,
    exist_ok=True
)


# ============================================================
# Command-line arguments
# ============================================================

if len(sys.argv) < 2:
    print(
        "Usage: python script.py <filename.csv> "
        "[--plot-only]"
    )
    sys.exit(1)

filename = sys.argv[1]

plot_only = "--plot-only" in sys.argv

if not filename.lower().endswith(".csv"):
    filename += ".csv"

csv_filename = os.path.join(
    OUTPUT_DIR,
    filename
)

empty_csv_filename = os.path.join(
    EMPTY_COPY_DIR,
    filename
)


# ============================================================
# CSV logging
# ============================================================

csv_rows = []

# 1920 samples at 48 kHz = 40 ms
csv_sample_interval = 1920


# ============================================================
# Callback state
# ============================================================

sample_index = 0
start_wall_time = None


# ============================================================
# Audio callback
# ============================================================

def callback(outdata, frames, time, status):

    global sample_index
    global start_wall_time

    if status:
        print(status)

    # --------------------------------------------------------
    # Get next audio chunk
    # --------------------------------------------------------

    chunk = stereo_signal[
        sample_index:sample_index + frames
    ]

    if len(chunk) < frames:

        outdata[:len(chunk)] = chunk

        outdata[len(chunk):] = 0

        raise sd.CallbackStop()

    else:

        outdata[:] = chunk


    # --------------------------------------------------------
    # Initialize wall-clock reference
    # --------------------------------------------------------

    if start_wall_time is None:

        start_wall_time = (
            datetime.now()
            - timedelta(
                seconds=time.outputBufferDacTime
            )
        )


    # --------------------------------------------------------
    # Log volume information
    # --------------------------------------------------------

    for i in range(
        0,
        frames,
        csv_sample_interval
    ):

        idx = sample_index + i

        if idx >= len(stereo_signal):
            break

        dac_time = (
            time.outputBufferDacTime
            + (i / fs)
        )

        wall_time = (
            start_wall_time
            + timedelta(seconds=dac_time)
        )

        csv_rows.append([
            wall_time.isoformat(),
            idx,
            f"{idx / fs:.6f}",
            f"{left_volumes[idx]:.6f}",
            f"{right_volumes[idx]:.6f}"
        ])


    # --------------------------------------------------------
    # Advance sample index
    # --------------------------------------------------------

    sample_index += frames


# ============================================================
# Run playback
# ============================================================

print(
    "Starting alternating LEFT/RIGHT playback..."
)

print(
    "Sequence: LEFT -> RIGHT -> LEFT -> RIGHT -> ..."
)

print(
    "Press Ctrl+C to stop and save CSV."
)

print()


try:

    with sd.OutputStream(
        samplerate=fs,
        channels=2,
        callback=callback,
        blocksize=0
    ):

        while True:
            sd.sleep(1000)


except KeyboardInterrupt:

    print("\nInterrupted by user.")


finally:

    # ========================================================
    # Save CSV
    # ========================================================

    if not plot_only:

        print("Writing CSV...")

        with open(
            csv_filename,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "timestamp",
                "sample_index",
                "time_seconds",
                "left_volume",
                "right_volume"
            ])

            writer.writerows(csv_rows)


        print(
            f"CSV saved: {csv_filename}"
        )

        print(
            f"Total rows: {len(csv_rows)}"
        )


        # ----------------------------------------------------
        # Create empty CSV copy
        # ----------------------------------------------------

        with open(
            empty_csv_filename,
            "w",
            newline=""
        ):
            pass


        print(
            f"Empty CSV created: {empty_csv_filename}"
        )


    else:

        print(
            "Plot-only mode: no files saved."
        )


# ============================================================
# Plot logged volume data
# ============================================================

if csv_rows:

    times = [
        float(row[2])
        for row in csv_rows
    ]

    left_vals = [
        float(row[3])
        for row in csv_rows
    ]

    right_vals = [
        float(row[4])
        for row in csv_rows
    ]


    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        times,
        left_vals,
        label="Left Volume",
        color="blue"
    )

    plt.plot(
        times,
        right_vals,
        label="Right Volume",
        color="red"
    )

    plt.title(
        "Alternating Left / Right Volume"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Volume"
    )

    plt.ylim(
        -0.5,
        1.2
    )

    plt.grid(True)

    plt.legend()

    print(
        "Plot saved."
    )

    plt.savefig(
        "noise_modulation.pdf",
        bbox_inches="tight"
    )

    plt.show()