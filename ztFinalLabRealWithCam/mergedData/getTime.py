import os
import pandas as pd

folder_path = "./"

doa_means = []
ring_means = []

for filename in os.listdir(folder_path):
    if not filename.lower().endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(folder_path, filename))

    if "doa_acquisition_time" not in df.columns or \
       "ring_processing_time" not in df.columns:
        continue

    doa = pd.to_numeric(df["doa_acquisition_time"], errors="coerce")
    ring = pd.to_numeric(df["ring_processing_time"], errors="coerce")

    doa_means.append(doa.mean())
    ring_means.append(ring.mean())

print("doa_acquisition_time:", sum(doa_means) / len(doa_means))
print("ring_processing_time:", sum(ring_means) / len(ring_means))