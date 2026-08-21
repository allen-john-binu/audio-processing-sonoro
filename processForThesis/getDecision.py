#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


OUTPUT_FILENAME = "decisionResults.csv"

REQUIRED_COLUMNS = [
    "robot_x",
    "robot_y",
    "speakerL_x",
    "speakerL_y",
    "speakerR_x",
    "speakerR_y",
]


def calculate_distance(x1, y1, x2, y2):
    """Calculate ordinary 2D Euclidean distance."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def error_result(csv_path, message):
    """Print the exact error and return an error decision."""
    print(f"ERROR: {csv_path.name}: {message}")
    return csv_path.name, "error"


def process_csv(csv_path):
    """
    Process one CSV file and return:
        (filename, decision)
    """

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Check whether the CSV has a header
            if reader.fieldnames is None:
                return error_result(csv_path, "CSV file has no header.")

            # Check required columns
            missing_columns = [
                column
                for column in REQUIRED_COLUMNS
                if column not in reader.fieldnames
            ]

            if missing_columns:
                return error_result(
                    csv_path,
                    f"Missing required column(s): {', '.join(missing_columns)}",
                )

            # Read the last row
            last_row = None
            row_number = 0

            for row_number, row in enumerate(reader, start=2):
                last_row = row

            # Empty CSV after header
            if last_row is None:
                return error_result(
                    csv_path,
                    "CSV file contains a header but no data rows.",
                )

            # Convert required values to floats
            values = {}

            for column in REQUIRED_COLUMNS:
                raw_value = last_row.get(column)

                if raw_value is None:
                    return error_result(
                        csv_path,
                        f"Column '{column}' is missing from the last row.",
                    )

                if raw_value.strip() == "":
                    return error_result(
                        csv_path,
                        f"Column '{column}' is empty in the last row "
                        f"(row {row_number}).",
                    )

                try:
                    values[column] = float(raw_value)
                except ValueError:
                    return error_result(
                        csv_path,
                        f"Invalid numeric value in column '{column}' "
                        f"in the last row (row {row_number}): "
                        f"{raw_value!r}",
                    )

            # Check for NaN / infinity
            for column, value in values.items():
                if not math.isfinite(value):
                    return error_result(
                        csv_path,
                        f"Column '{column}' contains a non-finite value "
                        f"in the last row (row {row_number}): {value!r}",
                    )

            # Extract coordinates
            robot_x = values["robot_x"]
            robot_y = values["robot_y"]

            speakerL_x = values["speakerL_x"]
            speakerL_y = values["speakerL_y"]

            speakerR_x = values["speakerR_x"]
            speakerR_y = values["speakerR_y"]

            # Calculate distances
            distance_left = calculate_distance(
                robot_x,
                robot_y,
                speakerL_x,
                speakerL_y,
            )

            distance_right = calculate_distance(
                robot_x,
                robot_y,
                speakerR_x,
                speakerR_y,
            )

            # Make decision
            if distance_left < distance_right:
                decision = "left"
            elif distance_right < distance_left:
                decision = "right"
            else:
                decision = "equal"

            return csv_path.name, decision

    except UnicodeDecodeError as e:
        return error_result(
            csv_path,
            f"Unable to decode CSV as UTF-8: {e}",
        )

    except csv.Error as e:
        return error_result(
            csv_path,
            f"CSV parsing error: {e}",
        )

    except OSError as e:
        return error_result(
            csv_path,
            f"File I/O error: {e}",
        )

    except Exception as e:
        return error_result(
            csv_path,
            f"Unexpected {type(e).__name__}: {e}",
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process CSV files and determine whether the robot "
            "is closer to the left or right speaker."
        )
    )

    parser.add_argument(
        "input_folder",
        type=Path,
        help="Path to the folder containing the CSV files",
    )

    args = parser.parse_args()
    input_folder = args.input_folder

    if not input_folder.exists():
        parser.error(f"Folder does not exist: {input_folder}")

    if not input_folder.is_dir():
        parser.error(f"Path is not a folder: {input_folder}")

    output_path = input_folder / OUTPUT_FILENAME

    # Process every CSV except decisionResults.csv
    csv_files = sorted(
        path
        for path in input_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and path.name != OUTPUT_FILENAME
    )

    results = []

    print(f"Found {len(csv_files)} CSV files.")
    print()

    for csv_path in csv_files:
        filename, decision = process_csv(csv_path)
        results.append((filename, decision))

    # Overwrite/create decisionResults.csv
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(["filename", "decision"])

        for filename, decision in results:
            writer.writerow([filename, decision])

    # Summary
    error_count = sum(
        1 for _, decision in results if decision == "error"
    )

    print()
    print(f"Processed {len(results)} CSV files.")
    print(f"Errors: {error_count}")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()