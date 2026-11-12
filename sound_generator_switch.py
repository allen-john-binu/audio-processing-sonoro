import scipy.signal as signal
import numpy as np
import sounddevice as sd

# =========================
# Audio parameters
# =========================
fs = 48000

durn = 5e-3             # Chirp duration: 5 ms
gap_duration = 1.0     # Time between consecutive ticks
target_duration = 6000  # Total playback duration: 10 minutes

# =========================
# Speaker volume controls
# =========================
base_left = 0.50 #real left
base_right = 1.0 

# =========================
# Generate chirp waveform
# =========================
t = np.linspace(
    0, durn, int(fs * durn), endpoint=False
)

sweep = signal.chirp(t, 500, t[-1], 3500)
sweep *= signal.windows.tukey(len(sweep), 0.2)

sweep = sweep.astype(np.float32)

chirp_samples = len(sweep)

# =========================
# Timing parameters
# =========================
gap_samples = int(gap_duration * fs)

interval_samples = gap_samples

target_samples = int(target_duration * fs)

# =========================
# Callback state
# =========================
sample_index = 0

# =========================
# Audio callback
# =========================
def callback(outdata, frames, time, status):

    global sample_index

    if status:
        print(status)

    remaining = target_samples - sample_index
    n = min(frames, remaining)

    # Initialize output as silence
    outdata.fill(0)

    if n <= 0:
        raise sd.CallbackStop()

    # Global sample indices
    indices = sample_index + np.arange(n)

    # Determine which 2-second interval is active
    interval_number = indices // interval_samples

    # Position within each interval
    local_position = indices % interval_samples

    # Tick is active during the first 5 ms of each interval
    tick_active = local_position < chirp_samples

    # Corresponding chirp sample positions
    chirp_indices = local_position[tick_active]

    # Even intervals: Left speaker
    left_active = tick_active & (interval_number % 2 == 0)

    # Odd intervals: Right speaker
    right_active = tick_active & (interval_number % 2 == 1)

    # Write left channel
    outdata[left_active, 0] = (
        sweep[local_position[left_active]] * base_left
    )

    # Write right channel
    outdata[right_active, 1] = (
        sweep[local_position[right_active]] * base_right
    )

    sample_index += n

    # Stop after 10 minutes
    if sample_index >= target_samples:
        raise sd.CallbackStop()


# =========================
# Run playback
# =========================
print("Starting playback...")
print("Left tick -> 2 seconds -> Right tick -> 2 seconds")
print("Repeating for 10 minutes.")
print(f"Left volume: {base_left}")
print(f"Right volume: {base_right}")

try:

    with sd.OutputStream(
        samplerate=fs,
        channels=2,
        dtype="float32",
        callback=callback,
        blocksize=0
    ) as stream:

        while stream.active:
            sd.sleep(100)

except KeyboardInterrupt:
    print("\nPlayback interrupted by user.")

print("Playback finished.")