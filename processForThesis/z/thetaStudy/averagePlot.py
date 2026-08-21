import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Settings
# ============================================================

# Input CSV file
csv_file = "./speaker_angle_at_b_per_file.csv"

# Mapping from group suffix to volume difference
volume_difference = {
    "AA": 0.00,
    "AB": 0.05,
    "AC": 0.10,
    "AD": 0.15,
    "AE": 0.20,
    "AF": 0.25,
    "AG": 0.30,
    "AH": 0.35,
    "AI": 0.40,
}

# ============================================================
# Read data
# ============================================================

df = pd.read_csv(csv_file)

# Extract AA, AB, ..., AI from the prefix
df["group"] = df["prefix"].str.split("_").str[-1]

# Map each group to its volume difference
df["volume_difference"] = df["group"].map(volume_difference)

# Check that all groups were mapped
if df["volume_difference"].isna().any():
    missing = df.loc[df["volume_difference"].isna(), "group"].unique()
    raise ValueError(f"Unknown groups found in CSV: {missing}")

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

groups = list(volume_difference.keys())

# 9 distinct colours
colors = plt.cm.tab10(range(len(groups)))

for group, color in zip(groups, colors):

    group_data = df[df["group"] == group]

    x = volume_difference[group]
    y = group_data["angle_deg"]

    # --------------------------------------------------------
    # Individual measurements
    # All 8 points are at exactly the same x-value
    # --------------------------------------------------------
    ax.scatter(
        [x] * len(y),
        y,
        color=color,
        s=45,
        alpha=0.8,
        label=group
    )

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------
    mean_angle = y.mean()

    ax.scatter(
        x,
        mean_angle,
        color=color,
        marker="x",
        s=140,
        linewidths=3,
        zorder=5
    )

# ============================================================
# Formatting
# ============================================================

ax.set_xlabel("Volume difference")
ax.set_ylabel("Angle (deg)")
ax.set_title("Angle vs. Volume Difference")

ax.set_xticks(list(volume_difference.values()))

ax.grid(True, alpha=0.25)

ax.legend(title="Group")

plt.tight_layout()
plt.show()