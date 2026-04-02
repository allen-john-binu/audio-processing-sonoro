import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters (edit freely)
# =========================
delta = 0.025    # try: 0.025, 0.05, 0.1, 0.15
mod_freq = 0.2     # Hz

base_left = 0.4
base_right = 0.82

duration = 20      # seconds
fs_plot = 1000     # samples per second

# =========================
# Time vector
# =========================
t = np.linspace(0, duration, int(fs_plot * duration))

# =========================
# Envelope functions (UPDATED)
# =========================
left = base_left + delta * np.sin(2 * np.pi * mod_freq * t)
right = base_right - delta * np.sin(2 * np.pi * mod_freq * t)

# =========================
# Plot
# =========================
plt.figure()

plt.plot(t, left, label="Left Channel")
plt.plot(t, right, label="Right Channel")

# Baselines
plt.axhline(base_left, linestyle='--', linewidth=1, label="Left Base")
plt.axhline(base_right, linestyle='--', linewidth=1, label="Right Base")

# Optional: sum (should be constant)
total = left + right
plt.plot(t, total, linestyle=':', label="Left + Right (constant)")

plt.title(f"Opposite Phase Envelopes (delta={delta}, freq={mod_freq} Hz)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()
plt.grid()

plt.show()