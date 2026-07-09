import pandas as pd
import matplotlib.pyplot as plt

# Read raw simulation data
df = pd.read_csv("parameter_study_results.csv")

# Compute mean phi for each parameter combination
mean_df = (
    df.groupby(["v", "h_b", "nf"], as_index=False)
      .agg(mean_phi=("phi", "mean"))
)

# -------------------------------------------------
# 3D scatter plot
# -------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    mean_df["h_b"],
    mean_df["nf"],
    mean_df["v"],
    c=mean_df["mean_phi"],
    cmap="coolwarm",
    s=15,
    alpha=0.8
)

ax.set_xlabel(r"$h_b$")
ax.set_ylabel("NORMALIZING_FACTOR")
ax.set_zlabel("v")
ax.set_title("Mean Phi")

cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label("Mean Phi (deg)")

plt.tight_layout()
plt.show()