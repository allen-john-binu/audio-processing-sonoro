import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.signal import chirp
import subprocess

# =====================
# Parameters
# =====================

f_start = 200        # start frequency (Hz)
f_end = 1200         # end frequency (Hz)
volume = 0.2

chirp_duration = 0.3
pause_duration = 0.2
repetitions = 10

sample_rate = 44100

wav_file = "soundGenerate/chirp1.wav"
mp3_file = "soundGenerate/chirp1.mp3"

# =====================
# Generate one chirp
# =====================

t = np.linspace(0, chirp_duration, int(sample_rate * chirp_duration), False)

chirp_signal = chirp(t, f0=f_start, f1=f_end, t1=chirp_duration, method='linear')
chirp_signal *= volume

# silence gap
pause = np.zeros(int(sample_rate * pause_duration))

# =====================
# Build full signal
# =====================

signal = np.array([])

for i in range(repetitions):
    print(f"Generating chirp {i+1}/{repetitions}", end="\r")
    signal = np.concatenate((signal, chirp_signal, pause))

# =====================
# Play sound
# =====================

# print("\nPlaying sound...")
# sd.play(signal, sample_rate)
# sd.wait()

# =====================
# Save WAV
# =====================

audio_int16 = np.int16(signal * 32767)
wav.write(wav_file, sample_rate, audio_int16)

# =====================
# Convert to MP3
# =====================

subprocess.run(["ffmpeg", "-y", "-i", wav_file, mp3_file])

print("Saved:", mp3_file)