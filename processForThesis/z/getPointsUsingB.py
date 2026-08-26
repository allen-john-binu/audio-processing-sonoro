#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_five_nearest(points_csv, experiment_folder, output_csv):
    points_csv = Path(points_csv)
    experiment_folder = Path(experiment_folder)
    output_csv = Path(output_csv)

    if not points_csv.is_file():
        raise FileNotFoundError(
            f"Points CSV does not exist: {points_csv}"
        )

    if not experiment_folder.is_dir():
        raise NotADirectoryError(
            f"Experiment folder does not exist: {experiment_folder}"
        )

    # ------------------------------------------------------------
    # Read the points CSV
    # ------------------------------------------------------------
    points_df = pd.read_csv(points_csv)

    required_points_columns = {
        "Bx",
        "By",
        "filename",
    }

    missing_columns = required_points_columns - set(points_df.columns)

    if missing_columns:
        raise ValueError(
            f"Points CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    result_rows = []

    # ------------------------------------------------------------
    # Process every row in the points CSV
    # ------------------------------------------------------------
    for points_row_number, points_row in points_df.iterrows():

        source_filename = str(points_row["filename"])
        experiment_path = experiment_folder / source_filename

        # --------------------------------------------------------
        # Check experiment CSV exists
        # --------------------------------------------------------
        if not experiment_path.is_file():
            raise FileNotFoundError(
                f"Experiment CSV not found for points CSV row "
                f"{points_row_number + 1}: {experiment_path}"
            )

        # --------------------------------------------------------
        # Get Bx / By
        # --------------------------------------------------------
        try:
            bx = float(points_row["Bx"])
            by = float(points_row["By"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid Bx/By in points CSV row "
                f"{points_row_number + 1}: "
                f"Bx={points_row['Bx']!r}, "
                f"By={points_row['By']!r}"
            )

        if not np.isfinite(bx) or not np.isfinite(by):
            raise ValueError(
                f"Bx/By are not finite in points CSV row "
                f"{points_row_number + 1}: "
                f"Bx={bx}, By={by}"
            )

        # --------------------------------------------------------
        # Read experiment CSV
        # --------------------------------------------------------
        experiment_df = pd.read_csv(experiment_path)

        required_experiment_columns = {
            "robot_x",
            "robot_y",
            "angle_1",
        }

        missing_experiment_columns = (
            required_experiment_columns - set(experiment_df.columns)
        )

        if missing_experiment_columns:
            raise ValueError(
                f"Experiment CSV {experiment_path} is missing "
                f"required columns: "
                f"{sorted(missing_experiment_columns)}"
            )

        experiment_df = experiment_df.copy()

        # --------------------------------------------------------
        # Convert relevant columns to numeric
        # --------------------------------------------------------
        experiment_df["robot_x"] = pd.to_numeric(
            experiment_df["robot_x"],
            errors="coerce"
        )

        experiment_df["robot_y"] = pd.to_numeric(
            experiment_df["robot_y"],
            errors="coerce"
        )

        experiment_df["angle_1"] = pd.to_numeric(
            experiment_df["angle_1"],
            errors="coerce"
        )

        # --------------------------------------------------------
        # Keep ONLY rows where:
        #
        #   robot_x is valid
        #   robot_y is valid
        #   angle_1 != -1000
        #
        # Note that NaN angle_1 is also excluded because it cannot
        # satisfy the required valid angle condition.
        # --------------------------------------------------------
        valid_mask = (
            experiment_df["robot_x"].notna()
            & experiment_df["robot_y"].notna()
            & np.isfinite(experiment_df["robot_x"])
            & np.isfinite(experiment_df["robot_y"])
            & experiment_df["angle_1"].notna()
            & np.isfinite(experiment_df["angle_1"])
            & (experiment_df["angle_1"] != -1000)
        )

        valid_experiment_df = experiment_df.loc[valid_mask].copy()

        # --------------------------------------------------------
        # We require at least 5 eligible rows
        # --------------------------------------------------------
        if len(valid_experiment_df) < 5:
            raise ValueError(
                f"Experiment CSV {experiment_path} has only "
                f"{len(valid_experiment_df)} eligible rows after "
                f"excluding invalid robot coordinates and "
                f"angle_1 == -1000. At least 5 are required."
            )

        # --------------------------------------------------------
        # Calculate Euclidean distance:
        #
        # sqrt((robot_x - Bx)^2 + (robot_y - By)^2)
        # --------------------------------------------------------
        dx = valid_experiment_df["robot_x"] - bx
        dy = valid_experiment_df["robot_y"] - by

        valid_experiment_df["_distance_to_B"] = np.sqrt(
            dx * dx + dy * dy
        )

        # --------------------------------------------------------
        # Select the 5 nearest eligible points.
        #
        # Stable sorting preserves original CSV order when two
        # rows have exactly the same distance.
        # --------------------------------------------------------
        nearest_five = (
            valid_experiment_df
            .sort_values(
                "_distance_to_B",
                ascending=True,
                kind="stable"
            )
            .head(3)
        )

        # --------------------------------------------------------
        # Create output rows
        # --------------------------------------------------------
        for _, experiment_row in nearest_five.iterrows():

            output_row = {}

            # ----------------------------------------------------
            # All columns from original points CSV row
            # ----------------------------------------------------
            for column in points_df.columns:
                output_row[column] = points_row[column]

            # ----------------------------------------------------
            # All columns from experiment CSV row
            #
            # Prefix experiment columns if a column name already
            # exists in the points CSV.
            # ----------------------------------------------------
            for column in experiment_df.columns:

                if column == "_distance_to_B":
                    continue

                if column in output_row:
                    output_column = f"experiment_{column}"
                else:
                    output_column = column

                output_row[output_column] = experiment_row[column]

            # ----------------------------------------------------
            # Additional information
            # ----------------------------------------------------
            output_row["source_filename"] = source_filename
            output_row["distance_to_B"] = experiment_row[
                "_distance_to_B"
            ]

            result_rows.append(output_row)

    # ------------------------------------------------------------
    # Create result DataFrame
    # ------------------------------------------------------------
    result_df = pd.DataFrame(result_rows)

    # ------------------------------------------------------------
    # Write result CSV
    # ------------------------------------------------------------
    result_df.to_csv(
        output_csv,
        index=False
    )

    print("Done.")
    print(f"Points CSV rows processed: {len(points_df)}")
    print(f"Output rows: {len(result_df)}")
    print(f"Expected output rows: {len(points_df) * 5}")
    print(f"Output file: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For every row in a points CSV, find the five experiment "
            "CSV rows closest to (Bx, By), excluding rows where "
            "angle_1 == -1000 or robot coordinates are invalid."
        )
    )

    parser.add_argument(
        "points_csv",
        help="Path to the points CSV"
    )

    parser.add_argument(
        "experiment_folder",
        help="Folder containing the experiment CSV files"
    )

    parser.add_argument(
        "output_csv",
        help="Path for the resulting CSV"
    )

    args = parser.parse_args()

    find_five_nearest(
        points_csv=args.points_csv,
        experiment_folder=args.experiment_folder,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
    
    
    # python3 getPointsUsingB.py \
    # ./thetaStudy/line_map_reconstruction.csv \
    # ./mergedData \
    # result.csv