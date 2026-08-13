import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

PROCESS_DATA_DIR = "./dataFromReal"
OUTPUT_DIR = "./analysis_timesteps"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD ALL FILES
# ============================================================

results = []
overall_values = []          # keeps all DOA values for histograms/statistics
overall_row_maxima = []      # one value per timestep

files = sorted(glob.glob(os.path.join(PROCESS_DATA_DIR, "*.csv")))

if not files:
    raise FileNotFoundError(f"No CSV files found in {PROCESS_DATA_DIR}")

for filepath in files:

    filename = os.path.splitext(os.path.basename(filepath))[0]

    values = []
    row_maxima = []

    with open(filepath, newline="") as f:

        reader = csv.reader(f)
        next(reader)

        for row in reader:

            try:
                doa = np.asarray(
                    [float(x) for x in row[2:]],
                    dtype=np.float64
                )

                values.extend(doa)

                # One timestep = one sample
                row_maxima.append(np.max(doa))

            except ValueError:
                continue

    if len(values) == 0:
        continue

    values = np.asarray(values)
    row_maxima = np.asarray(row_maxima)

    overall_values.extend(values)
    overall_row_maxima.extend(row_maxima)

    results.append({
        "filename": filename,
        "global_min": np.min(values),
        "global_max": np.max(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "n_timesteps": len(row_maxima)
    })

overall_values = np.asarray(overall_values)
overall_row_maxima = np.asarray(overall_row_maxima)

print(f"\nLoaded {len(overall_values):,} DOA values")
print(f"Loaded {len(overall_row_maxima):,} timesteps")

# ============================================================
# 1. PER FILE SUMMARY
# ============================================================

summary_df = pd.DataFrame(results)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "global_summary.csv"
)

summary_df.to_csv(summary_csv, index=False)

print(summary_df)

# ============================================================
# 2. OVERALL STATISTICS
# ============================================================

overall_stats = {
    "Global Min": np.min(overall_values),
    "Global Max": np.max(overall_values),
    "Mean": np.mean(overall_values),
    "Median": np.median(overall_values),
    "Std": np.std(overall_values)
}

stats_df = pd.DataFrame(
    overall_stats.items(),
    columns=["Statistic", "Value"]
)

stats_df.to_csv(
    os.path.join(OUTPUT_DIR, "overall_statistics.csv"),
    index=False
)

print(stats_df)

# ============================================================
# 3. PERCENTILE ANALYSIS
# ============================================================

percentiles = [
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    99.5,
    99.9,
    100
]

percentile_values = [
    np.percentile(overall_row_maxima, p)
    for p in percentiles
]

percentile_df = pd.DataFrame({
    "Percentile": percentiles,
    "Value": percentile_values
})

percentile_df.to_csv(
    os.path.join(OUTPUT_DIR, "percentiles.csv"),
    index=False
)

print("\nPercentiles")
print(percentile_df)

# ============================================================
# 4. HISTOGRAM (LINEAR)
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(
    overall_values,
    bins=200
)

plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Histogram")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "histogram_linear.png"),
    dpi=300
)

plt.close()

# ============================================================
# 5. HISTOGRAM (LOG SCALE)
# ============================================================

positive = overall_values[overall_values > 0]

plt.figure(figsize=(8,5))

plt.hist(
    np.log10(positive),
    bins=200
)

plt.xlabel("log10(Value)")
plt.ylabel("Count")
plt.title("Histogram (log10)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "histogram_log.png"),
    dpi=300
)

plt.close()

# ============================================================
# 6. BOXPLOT
# ============================================================

plt.figure(figsize=(10,2))

plt.boxplot(
    positive,
    vert=False
)

plt.xscale("log")

plt.title("Boxplot (Log Scale)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "boxplot_log.png"),
    dpi=300
)

plt.close()

# ============================================================
# 7. CDF
# ============================================================

sorted_values = np.sort(positive)

cdf = np.arange(len(sorted_values)) / len(sorted_values)

plt.figure(figsize=(8,5))

plt.plot(sorted_values, cdf)

plt.xscale("log")

plt.xlabel("Value")

plt.ylabel("CDF")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "cdf.png"),
    dpi=300
)

plt.close()

# ============================================================
# 8. MIN/MAX PER FILE
# ============================================================

plt.figure(figsize=(14,6))

plt.plot(
    summary_df["filename"],
    summary_df["global_min"],
    marker="o",
    label="Global Min"
)

plt.plot(
    summary_df["filename"],
    summary_df["global_max"],
    marker="o",
    label="Global Max"
)

plt.xticks(rotation=45, ha="right")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "min_max_per_file.png"),
    dpi=300
)

plt.close()

# ============================================================
# 9. SATURATION ANALYSIS
# ============================================================

candidate_percentiles = [
    99,
    99.5,
    99.9
]

rows = []

for p in candidate_percentiles:

    threshold = np.percentile(overall_row_maxima, p)

    clipped = np.sum(overall_row_maxima > threshold)

    rows.append({
        "Percentile": p,
        "Threshold": threshold,
        "Rows Clipped": clipped,
        "Percent Rows Clipped":
            clipped / len(overall_row_maxima) * 100
    })

sat_df = pd.DataFrame(rows)

sat_df.to_csv(
    os.path.join(OUTPUT_DIR, "saturation_analysis.csv"),
    index=False
)

print("\nSaturation Analysis")
print(sat_df)

print("\nAnalysis saved to:", OUTPUT_DIR)