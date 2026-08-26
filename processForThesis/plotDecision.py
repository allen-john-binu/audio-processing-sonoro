import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================


INPUT_CSV = "./zt/mergedData/decisionResults.csv"
OUTPUT_FIG = "volume_difference_vs_decision_zt.pdf"

# Only process filenames starting with this text
FILE_PREFIX = "19_08_"


# Volume difference for each condition
# volume_diff = {
#     "AA": 0.00,
#     "AB": 0.05,
#     "AC": 0.10,
#     "AD": 0.15,
#     "AE": 0.20,
#     "AF": 0.25,
#     "AG": 0.30,
#     "AH": 0.35,
#     "AI": 0.40,
# }

volume_diff = {
    "AA": 0.0,
    "AB": 0.1,
    "AC": 0.2,
    "AD": 0.3,
}


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(INPUT_CSV)

# Keep only filenames starting with the hardcoded prefix
df = df[
    df["filename"].str.startswith(
        FILE_PREFIX,
        na=False
    )
].copy()


# ============================================================
# Extract AA, AB, ..., AI from filename
# ============================================================

df["type"] = df["filename"].str.extract(
    rf"^{FILE_PREFIX}([A-Z]{{2}})_"
)

# Clean decision column
df["decision"] = (
    df["decision"]
    .str.strip()
    .str.lower()
)

# Add volume difference
df["volume_difference"] = df["type"].map(volume_diff)

# Remove anything that doesn't correspond to a known type
df = df.dropna(
    subset=["type", "volume_difference"]
)


# ============================================================
# Calculate percentage of RIGHT decisions
# ============================================================

summary = (
    df.groupby(
        ["type", "volume_difference"],
        sort=False
    )
    .agg(
        n=("decision", "size"),
        n_right=(
            "decision",
            lambda x: (x == "right").sum()
        )
    )
    .reset_index()
)

summary["right_ratio"] = (
    summary["n_right"]
    / summary["n"]
    * 100
)

summary = summary.sort_values(
    "volume_difference"
)


# ============================================================
# Expected decision
#
# 0 volume difference -> 50% right
# >0 volume difference -> 100% right
# ============================================================

summary["expected_ratio"] = 100.0

summary.loc[
    summary["volume_difference"] == 0,
    "expected_ratio"
] = 50.0


# ============================================================
# Print results
# ============================================================

print("\nSummary:")
print(
    summary[
        [
            "type",
            "volume_difference",
            "n",
            "n_right",
            "right_ratio",
            "expected_ratio",
        ]
    ].to_string(index=False)
)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

# Observed
ax.plot(
    summary["volume_difference"],
    summary["right_ratio"],
    marker="o",
    markersize=8,
    linewidth=2,
    label="Observed"
)

# Expected
ax.plot(
    summary["volume_difference"],
    summary["expected_ratio"],
    marker="o",
    markersize=6,
    linewidth=2,
    linestyle="--",
    label="Expected"
)


# Formatting
ax.set_xlabel(
    "Absolute amplitude difference, d",
    fontsize=13
)

ax.set_ylabel(
    "Right speaker decision (%)",
    fontsize=13
)

ax.set_title(
    "Speaker Volume Difference vs. Decision",
    fontsize=15
)

ax.set_xlim(-0.02, 0.42)
ax.set_ylim(-5, 105)

ax.set_xticks(
    list(volume_diff.values())
)

ax.set_yticks(
    range(0, 101, 10)
)

ax.grid(
    True,
    alpha=0.25
)

ax.legend(
    frameon=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FIG,
    dpi=300,
    bbox_inches="tight"
)

plt.show()