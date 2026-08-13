import numpy as np
import matplotlib.pyplot as plt

# Global min and max
global_min = 0.0000000003   # 3e-10
global_max = 0.0002932412

# Small epsilon to avoid log(0)
epsilon = 1e-12

# Generate values across the range
x = np.linspace(global_min, global_max, 1000)

# Log normalization
y = (np.log(x + epsilon) - np.log(global_min + epsilon)) / (
    np.log(global_max + epsilon) - np.log(global_min + epsilon)
)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, linewidth=2)
plt.xlim(0, 1e-6)
plt.xlabel("Original Value")
plt.ylabel("Log Normalized Value")
plt.title("Log Normalization")
plt.grid(True)
plt.tight_layout()
plt.show()