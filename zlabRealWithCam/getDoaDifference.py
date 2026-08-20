from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Folder containing the CSV files
folder = Path("./mergedData/")

# Output file
output_file = folder / "intensity_difference_mean.csv"

# DOA columns: doa_0 through doa_119
doa_columns = [f"doa_{i}" for i in range(120)]

results = []

# Natural filename sorting
import re

def natural_sort_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]

csv_files = sorted(
    [f for f in folder.glob("*.csv") if f.name != output_file.name],
    key=natural_sort_key
)

for csv_file in csv_files:
    print(f"Processing: {csv_file.name}")

    df = pd.read_csv(csv_file)

    # Check that all required DOA columns exist
    missing_columns = [col for col in doa_columns if col not in df.columns]

    if missing_columns:
        print(f"  Skipping: missing columns {missing_columns}")
        continue

    intensity_differences = []

    # Process every row
    for _, row in df[doa_columns].iterrows():

        values = row.to_numpy(dtype=float)

        # Find all peaks
        peak_indices, _ = find_peaks(values)

        # Ignore rows with fewer than 2 peaks
        if len(peak_indices) < 2:
            continue

        # Get the intensities at the detected peaks
        peak_values = values[peak_indices]

        # Select the two highest-intensity peaks
        two_highest = np.sort(peak_values)[-2:]

        peak1 = two_highest[0]
        peak2 = two_highest[1]

        # Absolute intensity difference
        difference = abs(peak1 - peak2)

        intensity_differences.append(difference)

    # Calculate mean across all qualifying rows
    if intensity_differences:
        mean_difference = np.mean(intensity_differences)
    else:
        mean_difference = np.nan

    results.append({
        "file": csv_file.name,
        "intensity_difference_mean": mean_difference,
        "num_rows_with_at_least_2_peaks": len(intensity_differences)
    })

    print(f"  Qualifying rows: {len(intensity_differences)}")
    print(f"  Mean intensity difference: {mean_difference}")

# Save results
result_df = pd.DataFrame(results)

result_df.to_csv(output_file, index=False)

print(f"\nSaved results to: {output_file}")
print(result_df)