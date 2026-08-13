import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hi.csv")

# Single row of intensity values
curve = df.iloc[27, 2:].astype(float).values

# ------------------------------------------------------------------
# Min-Max normalization
max_val = 6.8100e-06
minmax = curve / max_val

# ------------------------------------------------------------------
# Percentile values for log normalization
percentiles = {
    90.0: 6.685895823093989e-06,
    95.0: 2.1446301022338682e-05,
    96.0: 3.1865620945254505e-05,
    97.0: 4.277084089897525e-05,
    98.0: 5.8852725240544924e-05,
    99.0: 8.37786395947876e-05,
    99.5: 1.1127479673528774e-04,
    99.9: 1.9249963464094855e-04,
    100.0: 2.932412228901015e-04
}

# Compute log(curve) once
log_curve = np.log1p(curve)
log_min = log_curve.min()

# ------------------------------------------------------------------
# Plot 1: Original curve
plt.figure(figsize=(10, 4))
plt.plot(curve, lw=2)
plt.title("Original Curve")
plt.xlabel("Sample")
plt.ylabel("Value")
plt.grid(True)

# ------------------------------------------------------------------
# Plot 2: Compare normalizations
plt.figure(figsize=(12, 6))

# Min-Max in black
plt.plot(minmax, color='black', lw=3, label='Min-Max')

# Colormap for percentile curves
colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))

for color, (pct, value) in zip(colors, percentiles.items()):
    log_max = np.log1p(value)
    log_norm = (log_curve - log_min) / (log_max - log_min)

    plt.plot(
        log_norm,
        color=color,
        lw=2,
        label=f'Log ({pct:g}%)'
    )

plt.title("Min-Max vs Log Normalization Using Different Percentiles")
plt.xlabel("Sample")
plt.ylabel("Normalized Value")
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.legend(title="Normalization", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()