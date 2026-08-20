import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Hardcode your CSV paths
# ============================================================
intensity_csv = r"../mergedData/intensity_difference_mean.csv"
angle_csv = r"./speaker_angle_at_b_per_file.csv"


# ============================================================
# Load data
# ============================================================
df_intensity = pd.read_csv(intensity_csv)
df_angle = pd.read_csv(angle_csv)


# ============================================================
# Merge using filename
# ============================================================
df = pd.merge(
    df_intensity,
    df_angle,
    left_on="file",
    right_on="filename",
    how="inner"
)


# ============================================================
# Plot each prefix separately
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

for prefix, group in df.groupby("prefix"):
    ax.scatter(
        group["intensity_difference_mean"],
        group["angle_deg"],
        s=55,
        alpha=0.8,
        label=prefix
    )


ax.set_xlabel("Intensity Difference Mean")
ax.set_ylabel("Angle (deg)")
ax.set_title("Intensity Difference Mean vs Angle")

ax.grid(True, alpha=0.3)
ax.legend(
    title="Prefix",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.show()