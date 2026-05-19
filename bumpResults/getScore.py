import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# ============================================================
# ARGUMENT PARSER
# ============================================================

parser = argparse.ArgumentParser(
    description="Compare symmetry scores across experiment CSV files."
)

parser.add_argument(
    "folder",
    type=str,
    help="Path to folder containing CSV files"
)

parser.add_argument(
    "--angle",
    type=float,
    default=None,
    help="Only process timestamps where angle column > threshold"
)

args = parser.parse_args()

FOLDER_PATH = args.folder

# ============================================================
# VALIDATE FOLDER
# ============================================================

if not os.path.exists(FOLDER_PATH):
    raise ValueError(f"Folder does not exist: {FOLDER_PATH}")

# ============================================================
# CONFIG
# ============================================================

CLIP_ANGLE = 90

W_MEAN = 0.4
W_SD = 0.3
W_COUNT = 0.3

EPS = 1e-6

# ============================================================
# GET CSV FILES
# ============================================================


def natural_sort_key(path):
    filename = os.path.basename(path)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]

csv_files = sorted(
    glob.glob(os.path.join(FOLDER_PATH, "*.csv")),
    key=natural_sort_key
)


if len(csv_files) == 0:
    raise ValueError("No CSV files found.")

print(f"\nFound {len(csv_files)} CSV files.\n")

# ============================================================
# STORAGE
# ============================================================

run_names = []

mean_scores_all = []
sd_scores_all = []
count_scores_all = []
final_scores_all = []

# ============================================================
# PROCESS FILES
# ============================================================

for file_path in csv_files:

    file_name = os.path.basename(file_path)

    print(f"Processing: {file_name}")

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(file_path)

    # ========================================================
    # FILTER BY ANGLE THRESHOLD
    # ========================================================

    if args.angle is not None:

        if "angle" not in df.columns:
            raise ValueError(
                f"'angle' column not found in {file_name}"
            )

        original_count = len(df)

        df = df[df["angle"] > args.angle]

        filtered_count = len(df)

        print(
            f"  Angle filter > {args.angle} : "
            f"{filtered_count}/{original_count} timestamps kept"
        )

        # skip empty files after filtering
        if filtered_count == 0:
            print("  No timestamps passed filter. Skipping.\n")
            continue

    # --------------------------------------------------------
    # First column = timestamp
    # Remaining columns = bump angles
    # --------------------------------------------------------

    angle_cols = df.columns[1:]

    # --------------------------------------------------------
    # Per-timestamp score storage
    # --------------------------------------------------------

    mean_scores = []
    sd_scores = []
    count_scores = []
    final_scores = []

    # ========================================================
    # PROCESS TIMESTAMPS
    # ========================================================

    for _, row in df.iterrows():

        angles = row[angle_cols].astype(float).values

        # ----------------------------------------------------
        # Clip angles
        # ----------------------------------------------------

        angles = np.clip(angles, -CLIP_ANGLE, CLIP_ANGLE)

        # ----------------------------------------------------
        # Split positive / negative
        # ----------------------------------------------------

        positive = angles[angles > 0]
        negative = np.abs(angles[angles < 0])

        nP = len(positive)
        nN = len(negative)

        # ----------------------------------------------------
        # Edge case
        # ----------------------------------------------------

        if nP == 0 or nN == 0:

            S_mean = 0
            S_sd = 0
            S_count = 0
            S_final = 0

        else:

            # =================================================
            # Mean symmetry score
            # =================================================

            muP = np.mean(positive)
            muN = np.mean(negative)

            S_mean = 1 - abs(muP - muN) / (muP + muN + EPS)

            # =================================================
            # SD symmetry score
            # =================================================

            sdP = np.std(positive, ddof=0)
            sdN = np.std(negative, ddof=0)

            S_sd = 1 - abs(sdP - sdN) / (sdP + sdN + EPS)

            # =================================================
            # Count balance score
            # =================================================

            S_count = (2 * min(nP, nN)) / (nP + nN)

            # =================================================
            # Final score
            # =================================================

            S_final = (
                W_MEAN * S_mean +
                W_SD * S_sd +
                W_COUNT * S_count
            )

        mean_scores.append(S_mean)
        sd_scores.append(S_sd)
        count_scores.append(S_count)
        final_scores.append(S_final)

    # ========================================================
    # FILE-LEVEL SCORES
    # ========================================================

    run_name = os.path.splitext(file_name)[0]

    run_names.append(run_name)

    mean_scores_all.append(np.mean(mean_scores))
    sd_scores_all.append(np.mean(sd_scores))
    count_scores_all.append(np.mean(count_scores))
    final_scores_all.append(np.mean(final_scores))

# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame({
    "Run": run_names,
    "MeanScore": mean_scores_all,
    "SDScore": sd_scores_all,
    "CountScore": count_scores_all,
    "FinalScore": final_scores_all
})

# ============================================================
# SORT BY FINAL SCORE
# ============================================================

results_df = results_df.sort_values(
    by="FinalScore",
    ascending=False
).reset_index(drop=True)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n================ RESULTS ================\n")

print(results_df.to_string(index=False))

# ============================================================
# BEST RUN
# ============================================================

best_run = results_df.iloc[0]["Run"]
best_score = results_df.iloc[0]["FinalScore"]

print("\n=========================================")
print(f"BEST RUN   : {best_run}")
print(f"BEST SCORE : {best_score:.4f}")
print("=========================================\n")

# ============================================================
# PLOT
# ============================================================

x = np.arange(len(results_df))

plt.figure(figsize=(14, 7))

plt.plot(x,
         results_df["MeanScore"],
         marker='o',
         label="Mean Score")

plt.plot(x,
         results_df["SDScore"],
         marker='o',
         label="SD Score")

plt.plot(x,
         results_df["CountScore"],
         marker='o',
         label="Count Score")

plt.plot(x,
         results_df["FinalScore"],
         marker='o',
         linewidth=3,
         label="Final Score")

plt.xticks(x,
           results_df["Run"],
           rotation=45)

plt.xlabel("Experiment Run")
plt.ylabel("Score")
plt.title("Experiment Comparison Scores")

plt.ylim(0, 1)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()