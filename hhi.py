import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
sigma = 0.1      # try: 0.025, 0.05, 0.1, 0.2
base_left = 0.4
base_right = 0.8

duration = 20
fs_plot = 1000

np.random.seed(0)  # reproducible

# =========================
# Time vector
# =========================
t = np.linspace(0, duration, int(fs_plot * duration))

# =========================
# Independent Gaussian noise
# =========================
noise_left = np.random.normal(0, sigma, size=len(t))
noise_right = np.random.normal(0, sigma, size=len(t))

# =========================
# Envelopes
# =========================
left = base_left + noise_left
right = base_right + noise_right

# Optional safety clipping
left = np.clip(left, 0.0, 1.0)
right = np.clip(right, 0.0, 1.0)

# =========================
# Plot
# =========================
plt.figure()

plt.plot(t, left, label="Left Channel")
plt.plot(t, right, label="Right Channel")

plt.axhline(base_left, linestyle='--', linewidth=1, label="Left Base")
plt.axhline(base_right, linestyle='--', linewidth=1, label="Right Base")

plt.title(f"Independent Gaussian Modulation (sigma={sigma})")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()
plt.grid()

plt.show()