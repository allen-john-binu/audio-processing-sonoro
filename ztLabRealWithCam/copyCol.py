import pandas as pd

# ==========================
# Configuration
# ==========================

# Source CSV (contains the column to copy)
source_csv = r"./dataFromRing/16_07_ED_05.csv"

# Destination CSV (will receive the column)
destination_csv = r"./dataFromSound/16_07_ED_05.csv"

# Name of the column to copy
column_name = "timestamp"

# Output CSV (use destination_csv to overwrite the destination file)
output_csv = destination_csv

# ==========================
# Read CSVs
# ==========================

source_df = pd.read_csv(source_csv)
destination_df = pd.read_csv(destination_csv)

# Check if the column exists in the source
if column_name not in source_df.columns:
    raise ValueError(f"Column '{column_name}' not found in source CSV.")

# Ensure both files have the same number of rows
if len(source_df) != len(destination_df):
    raise ValueError(
        f"Row count mismatch: source has {len(source_df)} rows, "
        f"destination has {len(destination_df)} rows."
    )

# Delete the column from destination if it already exists
if column_name in destination_df.columns:
    destination_df.drop(columns=[column_name], inplace=True)

# Copy the column from source to destination
destination_df[column_name] = source_df[column_name]

# Save (overwrites the destination CSV)
destination_df.to_csv(output_csv, index=False)

print(f"Column '{column_name}' copied successfully.")
print(f"Updated destination saved to: {output_csv}")