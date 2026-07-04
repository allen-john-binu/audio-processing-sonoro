import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PROCESS_DATA_DIR = "./ztProcessData"
OUTPUT_CSV = "global_min_max_summary.csv"
OUTPUT_PLOT = "global_min_max_plot.png"

# ------------------------------------------------------------------
# PROCESS FILES
# ------------------------------------------------------------------
results = []

overall_values = []

files = sorted(glob.glob(os.path.join(PROCESS_DATA_DIR, "*.csv")))

if not files:
    raise FileNotFoundError(
        f"No CSV files found in {PROCESS_DATA_DIR}"
    )

for filepath in files:

    filename = os.path.splitext(os.path.basename(filepath))[0]

    all_values = []

    with open(filepath, newline="") as f:
        reader = csv.reader(f)

        header = next(reader)

        # Same logic as original script
        robot_x_idx = header.index("robot_x")

        for row in reader:

            try:
                # DOA columns only
                values = [
                    float(x)
                    for x in row[2:robot_x_idx]
                ]

                all_values.extend(values)

            except ValueError:
                continue

    if len(all_values) == 0:
        print(f"Skipping {filename} (no valid values)")
        continue

    global_min = min(all_values)
    global_max = max(all_values)

    overall_values.extend(all_values)

    results.append({
        "filename": filename,
        "global_min": global_min,
        "global_max": global_max
    })

    print(
        f"{filename:40s} "
        f"min={global_min:.6f} "
        f"max={global_max:.6f}"
    )

# ------------------------------------------------------------------
# SUMMARY CSV
# ------------------------------------------------------------------
df = pd.DataFrame(results)

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSummary saved to: {OUTPUT_CSV}")

# ------------------------------------------------------------------
# OVERALL GLOBAL VALUES ACROSS ALL FILES
# ------------------------------------------------------------------
overall_global_min = min(overall_values)
overall_global_max = max(overall_values)

print("\n===================================")
print(f"Overall Global Min: {overall_global_min:.10f}")
print(f"Overall Global Max: {overall_global_max:.10f}")
print("===================================\n")

# ------------------------------------------------------------------
# PLOT
# ------------------------------------------------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    df["filename"],
    df["global_min"],
    marker="o",
    label="Global Min"
)

plt.plot(
    df["filename"],
    df["global_max"],
    marker="o",
    label="Global Max"
)

plt.xlabel("CSV File")
plt.ylabel("Value")
plt.title("Global Min and Global Max per CSV File")

plt.xticks(rotation=45, ha="right")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(OUTPUT_PLOT, dpi=300)

print(f"Plot saved to: {OUTPUT_PLOT}")

plt.show()