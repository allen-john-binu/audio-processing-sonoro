import numpy as np
import scipy.signal as signal
from scipy.io.wavfile import write

# =========================
# Parameters
# =========================
fs = 48000
duration_seconds = 4 * 60  # 4 minutes

# =========================
# Generate sweep
# =========================
durn = 5e-3  # 5 ms

t = np.linspace(0, durn, int(fs * durn), endpoint=False)

sweep = signal.chirp(
    t,
    f0=2000,
    f1=24000,
    t1=t[-1]
)

sweep *= signal.windows.tukey(len(sweep), 0.2)
sweep *= 0.8

# 100 ms silence before and after
sweep = np.pad(
    sweep,
    (int(fs * 0.1), int(fs * 0.1))
)

# =========================
# Repeat to 4 minutes
# =========================
repeats = int(np.ceil(duration_seconds / (len(sweep) / fs)))

long_signal = np.tile(sweep, repeats)
long_signal = long_signal[: int(fs * duration_seconds)]

# =========================
# Constant volumes
# =========================
left_gain = 1.0
right_gain = 1.0

stereo_signal = np.column_stack((
    long_signal * left_gain,
    long_signal * right_gain
))

# =========================
# Convert to 16-bit PCM
# =========================
audio_int16 = np.int16(
    np.clip(stereo_signal, -1.0, 1.0) * 32767
)

# =========================
# Save WAV
# =========================
output_file = "sweep_4min_L1.0_R1.0.wav"

write(output_file, fs, audio_int16)

print(f"Saved: {output_file}")
print(f"Duration: {duration_seconds} seconds")
print(f"Sample rate: {fs} Hz")