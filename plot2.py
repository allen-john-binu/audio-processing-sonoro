import numpy as np
import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

# Negative angle robot is on the left, positive angle robot is on the right.

ANGLE_START = 2

# -----------------------------
# 1. Load CSV
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python3 plot_bump_stats.py <path_to_csv>")
    sys.exit(1)

csv_path = pathlib.Path(sys.argv[1])
if not csv_path.exists():
    print(f"Error: file not found — {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

angle_cols = df.columns[ANGLE_START:]
angles = np.array([float(a) for a in angle_cols])

spl_array = df["dB_SPL"].astype(str).str.strip("[]").astype(float).values
intensity_matrix = df.iloc[:, ANGLE_START:].values.astype(float)

# -----------------------------
# 2. Angle spacing fix
# -----------------------------
angle_step = angles[1] - angles[0]
y_min = angles[0] - angle_step / 2
y_max = angles[-1] + angle_step / 2

# -----------------------------
# 3. Figure setup
# -----------------------------
fig, (ax, ax_diff) = plt.subplots(
    1, 2,
    figsize=(16, 8),
    gridspec_kw={"width_ratios": [3, 1]}
)

plt.subplots_adjust(wspace=0.3)

threshold_init = spl_array.min()

# -----------------------------
# 4. Filtering function
# -----------------------------
def filter_data(threshold):
    mask = spl_array >= threshold

    filtered_intensity = intensity_matrix[mask]
    filtered_spl = spl_array[mask]

    peak_angles_1 = []
    peak_angles_2 = []
    intensity_diffs = []

    for row in filtered_intensity:
        peaks, _ = signal.find_peaks(row)

        if len(peaks) >= 2:
            # Two highest peaks
            top2 = peaks[np.argsort(row[peaks])[-2:]]
            p1, p2 = top2

            a1, a2 = angles[p1], angles[p2]
            i1, i2 = row[p1], row[p2]

            peak_angles_1.append(a1)
            peak_angles_2.append(a2)
            intensity_diffs.append(abs(i1 - i2))

        elif len(peaks) == 1:
            p1 = peaks[0]

            peak_angles_1.append(angles[p1])
            peak_angles_2.append(np.nan)
            intensity_diffs.append(np.nan)

        else:
            peak_angles_1.append(np.nan)
            peak_angles_2.append(np.nan)
            intensity_diffs.append(np.nan)

    new_time = np.arange(len(filtered_intensity))

    return (
        filtered_intensity,
        np.array(peak_angles_1),
        np.array(peak_angles_2),
        np.array(intensity_diffs),
        new_time,
        filtered_spl,
    )


# -----------------------------
# 5. Initial data
# -----------------------------
(
    filtered_intensity,
    peak1,
    peak2,
    intensity_diffs,
    time_steps,
    filtered_spl,
) = filter_data(threshold_init)

# -----------------------------
# 6. Heatmap
# -----------------------------
im = ax.imshow(
    filtered_intensity.T,
    aspect="auto",
    cmap="inferno",
    origin="lower",
    extent=[-0.5, len(time_steps) - 0.5, y_min, y_max],
)

scatter1 = ax.scatter(
    time_steps,
    peak1,
    color="cyan",
    s=25,
    label="Primary Peak",
)

scatter2 = ax.scatter(
    time_steps,
    peak2,
    color="lime",
    s=25,
    label="Secondary Peak",
)

ax.set_xlabel("Time Step")
ax.set_ylabel("Angle (degrees)")
ax.set_title("DOA Heatmap with Peak Locations")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Intensity")

ax.legend(loc="upper right", frameon=True)

# -----------------------------
# 7. Peak intensity difference plot
# -----------------------------
intensity_plot = np.nan_to_num(intensity_diffs, nan=0.0)

valid = ~np.isnan(intensity_diffs)
invalid = np.isnan(intensity_diffs)

ax_diff.scatter(
    time_steps[valid],
    intensity_plot[valid],
    color="tab:blue",
    s=25,
    label="Two Peaks Found",
)

ax_diff.scatter(
    time_steps[invalid],
    intensity_plot[invalid],
    color="red",
    s=25,
    label="Less than Two Peaks",
)

ax_diff.set_title("Peak Intensity Difference")
ax_diff.set_xlabel("Time Step")
ax_diff.set_ylabel(r"$|I_1-I_2|$")
ax_diff.set_xlim(-0.5, len(time_steps) - 0.5)

# Set y-limits nicely
if np.any(valid):
    ymax = np.nanmax(intensity_diffs)
    ax_diff.set_ylim(0, ymax * 1.1)

ax_diff.grid(True, alpha=0.3)
ax_diff.legend(loc="upper right", frameon=True)

# -----------------------------
# 8. Save figure
# -----------------------------
plt.tight_layout()

output_pdf = csv_path.with_suffix(".pdf")
plt.savefig(output_pdf, format="pdf", dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_pdf}")

# -----------------------------
# 9. Show
# -----------------------------
plt.show()