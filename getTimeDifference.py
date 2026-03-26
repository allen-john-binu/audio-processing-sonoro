import os
import pandas as pd
import matplotlib.pyplot as plt

folder_path = "./dataFromReal"

file_means = {}
all_diffs = []

# Loop through all CSV files
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        
        # Read file
        df = pd.read_csv(file_path)
        
        # Convert to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort just in case
        df = df.sort_values('timestamp')
        
        # Compute differences
        diffs = df['timestamp'].diff().dropna()
        
        # Convert to seconds (or keep timedelta if you prefer)
        diffs_seconds = diffs.dt.total_seconds()
        
        # Mean difference for this file
        mean_diff = diffs_seconds.mean()
        
        file_means[file] = mean_diff
        all_diffs.extend(diffs_seconds)

# Overall mean
overall_mean = sum(all_diffs) / len(all_diffs)

print("Mean per file:")
for file, mean in file_means.items():
    print(f"{file}: {mean:.2f} seconds")

print(f"\nOverall mean: {overall_mean:.2f} seconds")

# 📊 Plot
plt.figure()
plt.plot(list(file_means.keys()), list(file_means.values()), marker='o')
plt.xticks(rotation=45)
plt.xlabel("File")
plt.ylabel("Mean Time Difference (seconds)")
plt.title("Mean Timestamp Difference per File")
plt.tight_layout()
plt.show()