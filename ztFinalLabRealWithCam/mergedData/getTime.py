import os
import pandas as pd

# Folder containing your CSV files
folder_path = "./"

# Required headers
required_columns = [
    "doa_acquisition_time",
    "ring_processing_time"
]

results = []
skipped = []

for filename in os.listdir(folder_path):
    if not filename.lower().endswith(".csv"):
        continue

    file_path = os.path.join(folder_path, filename)

    try:
        df = pd.read_csv(file_path)

        # Skip if either required column is missing
        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            skipped.append({
                "file": filename,
                "reason": f"Missing: {', '.join(missing_columns)}"
            })
            continue

        # Convert values to numeric; invalid values become NaN
        df["doa_acquisition_time"] = pd.to_numeric(
            df["doa_acquisition_time"], errors="coerce"
        )
        df["ring_processing_time"] = pd.to_numeric(
            df["ring_processing_time"], errors="coerce"
        )

        results.append({
            "file": filename,
            "doa_acquisition_time_mean": df["doa_acquisition_time"].mean(),
            "ring_processing_time_mean": df["ring_processing_time"].mean(),
        })

    except Exception as e:
        skipped.append({
            "file": filename,
            "reason": f"Error reading file: {e}"
        })

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save results
output_file = os.path.join(folder_path, "means.csv")
results_df.to_csv(output_file, index=False)

# Print results
print("\n=== Results ===")
print(results_df.to_string(index=False))

print(f"\nResults saved to: {output_file}")

# Print skipped files
if skipped:
    skipped_df = pd.DataFrame(skipped)

    skipped_file = os.path.join(folder_path, "skipped.csv")
    skipped_df.to_csv(skipped_file, index=False)

    print("\n=== Skipped Files ===")
    print(skipped_df.to_string(index=False))
    print(f"\nSkipped-file details saved to: {skipped_file}")