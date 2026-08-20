from pathlib import Path
import pandas as pd

# Folder containing the CSV files
folder = Path("./mergedData/")

for csv_file in folder.glob("*.csv"):
    print(f"Processing: {csv_file.name}")

    df = pd.read_csv(csv_file)

    if "phi_in_degrees" not in df.columns:
        print("  Skipping: 'phi_in_degrees' column not found")
        continue

    # Remove rows where phi_in_degrees is exactly -500
    df = df[df["phi_in_degrees"] != -500]

    # Overwrite the original file
    df.to_csv(csv_file, index=False)

    # Output final row count
    print(f"  Final row count: {len(df)}")