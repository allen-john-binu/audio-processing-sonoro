import pandas as pd

# Load DOA data
df = pd.read_csv("hi.csv")

# Metadata columns to preserve
meta_cols = ["timestamp", "dB_SPL"]

# Angle columns
angles = [float(c) for c in df.columns if c not in meta_cols]

# Output dataframe
sym_df = df.copy()

# Average opposite angles
for angle in angles:
    if angle < 0:
        pos = str(-angle)
        neg = str(angle)

        avg = (df[neg] + df[pos]) / 2

        sym_df[neg] = avg
        sym_df[pos] = avg

# 0° remains unchanged

# Save
sym_df.to_csv("hi_symmetric_average.csv", index=False)

print("Saved symmetric DOA to hi_symmetric.csv")