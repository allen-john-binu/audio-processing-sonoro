import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 15,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ============================================================
# Parameters
# ============================================================
fs = 48000

durn = 5e-3          # Chirp duration = 5 ms
wait_time = 0.01    # Waiting time = 10 ms

f_start = 2000      # 2 kHz
f_end = 24000       # 24 kHz


# ============================================================
# Generate one chirp
# ============================================================
t_chirp = np.linspace(
    0,
    durn,
    int(fs * durn),
    endpoint=False
)

chirp = signal.chirp(
    t_chirp,
    f_start,
    durn,
    f_end,
    method="linear"
)

# Same window and amplitude as your original generator
chirp *= signal.windows.tukey(len(chirp), 0.2)
chirp *= 0.8


# ============================================================
# Create one chirp + waiting interval
# ============================================================
silence = np.zeros(int(fs * wait_time))

cycle = np.concatenate([
    chirp,
    silence
])


# ============================================================
# Three chirps
# ============================================================
three_chirps = np.tile(cycle, 3)

t = np.arange(len(three_chirps)) / fs


# ============================================================
# Plot
# ============================================================
# plt.rcParams.update({
#     "font.family": "DejaVu Sans",
#     "font.size": 18,
#     "axes.labelsize": 12,
#     "axes.titlesize": 13,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

fig, ax = plt.subplots(figsize=(10, 4.5))

ax.plot(
    t * 1000,          # Convert seconds → milliseconds
    three_chirps,
    linewidth=0.9
)

# ============================================================
# Axis labels
# ============================================================
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")

ax.set_title(
    "Three consecutive chirp stimuli",
    fontweight="bold"
)

# 3 × (5 ms chirp + 10 ms wait) = 45 ms
ax.set_xlim(0, 45)
ax.set_ylim(-0.9, 0.9)

ax.grid(
    True,
    linewidth=0.4,
    alpha=0.35
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# Mark chirps
# ============================================================
chirp_starts = [0, 15, 30]

for i, start in enumerate(chirp_starts):

    ax.annotate(
        f"Chirp {i + 1}",
        xy=(start + 2.5, 0.55),
        xytext=(start + 2.5, 0.82),
        ha="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.8
        )
    )


# ============================================================
# Mark waiting intervals
# ============================================================

# ============================================================
# Save
# ============================================================
plt.tight_layout()

plt.savefig(
    "three_chirps_10ms_wait.pdf",
    bbox_inches="tight", format="pdf"
)

plt.savefig(
    "three_chirps_10ms_wait.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()