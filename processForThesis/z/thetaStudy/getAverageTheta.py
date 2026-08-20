"""
Using the output of line_map_reconstruction.py (line_map_reconstruction.csv),
compute, for each file, the absolute angle at the reconstructed corner point
B between the speakerR and speakerL (mode) positions:

    angle = angle(speakerR - B, speakerL - B)

reported in degrees, in the range [0, 180] (i.e. unsigned / "absolute" angle
- no notion of clockwise vs counter-clockwise).

The per-file angles are then averaged per "type" (the date_group prefix,
e.g. "09_07_AA"), giving one averaged value per type.

Input:
    line_map_reconstruction.csv, with (at least) columns:
        prefix, filename, Bx, By,
        speakerL_x_mode, speakerL_y_mode,
        speakerR_x_mode, speakerR_y_mode

Output:
    - PER_FILE_OUTPUT_CSV: one row per file, with the computed angle_deg
    - PER_TYPE_OUTPUT_CSV: one row per prefix/type, with the averaged angle_deg
      and the number of files that contributed to the average

Usage:
    python speaker_angle_at_b.py
"""

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Config - edit these to match your setup
INPUT_CSV = "./line_map_reconstruction.csv"
PER_FILE_OUTPUT_CSV = "./speaker_angle_at_b_per_file.csv"
PER_TYPE_OUTPUT_CSV = "./speaker_angle_at_b_per_type_avg.csv"
# ------------------------------------------------------------------ #


def angle_at_vertex(vertex: np.ndarray, point1: np.ndarray, point2: np.ndarray):
    """
    Absolute (unsigned) angle in degrees, at `vertex`, between the rays
    vertex->point1 and vertex->point2. Returns np.nan if either ray has
    zero length (i.e. a point coincides with the vertex).
    """
    v1 = point1 - vertex
    v2 = point2 - vertex
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # guard against float rounding
    return float(np.degrees(np.arccos(cos_angle)))


def main():
    df = pd.read_csv(INPUT_CSV)

    required_cols = ["prefix", "filename", "Bx", "By",
                      "speakerL_x_mode", "speakerL_y_mode",
                      "speakerR_x_mode", "speakerR_y_mode"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{INPUT_CSV} is missing required columns: {missing}")

    angles = []
    for _, row in df.iterrows():
        B = np.array([row["Bx"], row["By"]], dtype=float)
        speakerL = np.array([row["speakerL_x_mode"], row["speakerL_y_mode"]], dtype=float)
        speakerR = np.array([row["speakerR_x_mode"], row["speakerR_y_mode"]], dtype=float)

        if np.any(np.isnan(B)) or np.any(np.isnan(speakerL)) or np.any(np.isnan(speakerR)):
            print(f"{row['filename']}: NaN in B/speakerL/speakerR, angle set to NaN.")
            angles.append(np.nan)
            continue

        angle_deg = angle_at_vertex(B, speakerR, speakerL)
        if np.isnan(angle_deg):
            print(f"{row['filename']}: speaker point coincides with B, angle set to NaN.")
        angles.append(angle_deg)

    per_file_df = df[["prefix", "index", "filename"]].copy() if "index" in df.columns \
        else df[["prefix", "filename"]].copy()
    per_file_df["angle_deg"] = angles
    per_file_df.to_csv(PER_FILE_OUTPUT_CSV, index=False)
    print(f"Saved: {PER_FILE_OUTPUT_CSV} ({len(per_file_df)} rows)")

    per_type_df = (
        per_file_df.groupby("prefix")["angle_deg"]
        .agg(avg_angle_deg="mean", n_files="count")
        .reset_index()
        .sort_values("prefix")
    )
    per_type_df.to_csv(PER_TYPE_OUTPUT_CSV, index=False)
    print(f"Saved: {PER_TYPE_OUTPUT_CSV} ({len(per_type_df)} rows)")


if __name__ == "__main__":
    main()