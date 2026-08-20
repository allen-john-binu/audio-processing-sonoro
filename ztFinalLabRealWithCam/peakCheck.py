import csv
from pathlib import Path
from scipy.signal import find_peaks


INPUT_FOLDER = Path("./processedData/")
OUTPUT_FOLDER = Path("./peakCheck/")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def index_to_angle(index):
    return 180 - index * 3


def process_csv(input_file, output_file):
    output_rows = []
    all_both = True

    with open(input_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_index, row in enumerate(reader):

            doa_values = []

            for i in range(120):
                try:
                    doa_values.append(float(row[f"doa_{i}"]))
                except (ValueError, TypeError):
                    doa_values.append(0.0)

            peak_indices, _ = find_peaks(doa_values)

            peak_indices = sorted(
                peak_indices,
                key=lambda i: doa_values[i],
                reverse=True
            )

            angle_0 = float(row["angle_0"])
            angle_1 = float(row["angle_1"])
            phi_in_degrees = float(row["phi_in_degrees"])

            if len(peak_indices) < 2:
                output_rows.append({
                    "row_index": row_index,
                    "angle_0": angle_0,
                    "angle_1": angle_1,
                    "phi_in_degrees": phi_in_degrees,
                    "highest_peak_index": "",
                    "highest_peak_value": "",
                    "highest_peak_angle": "",
                    "second_peak_index": "",
                    "second_peak_value": "",
                    "second_peak_angle": "",
                    "highest_matches_angle_0": "",
                    "second_matches_angle_1": "",
                    "result": "fewerThan2Peaks",
                })

                all_both = False
                continue

            highest_index = peak_indices[0]
            highest_value = doa_values[highest_index]
            highest_angle = index_to_angle(highest_index)

            second_index = peak_indices[1]
            second_value = doa_values[second_index]
            second_angle = index_to_angle(second_index)

            highest_matches = highest_angle == angle_0
            second_matches = second_angle == angle_1

            if highest_matches and second_matches:
                result = "both"
            elif highest_matches:
                result = "onlyHighest"
            elif second_matches:
                result = "onlySecond"
            else:
                result = "neither"

            if result != "both":
                all_both = False

            output_rows.append({
                "row_index": row_index,
                "angle_0": angle_0,
                "angle_1": angle_1,
                "phi_in_degrees": phi_in_degrees,
                "highest_peak_index": highest_index,
                "highest_peak_value": highest_value,
                "highest_peak_angle": highest_angle,
                "second_peak_index": second_index,
                "second_peak_value": second_value,
                "second_peak_angle": second_angle,
                "highest_matches_angle_0": highest_matches,
                "second_matches_angle_1": second_matches,
                "result": result,
            })

    fieldnames = [
        "row_index",
        "angle_0",
        "angle_1",
        "phi_in_degrees",
        "highest_peak_index",
        "highest_peak_value",
        "highest_peak_angle",
        "second_peak_index",
        "second_peak_value",
        "second_peak_angle",
        "highest_matches_angle_0",
        "second_matches_angle_1",
        "result",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return all_both


# Process all CSV files
for csv_file in sorted(INPUT_FOLDER.glob("*.csv")):
    output_file = OUTPUT_FOLDER / csv_file.name

    all_both = process_csv(csv_file, output_file)

    print(f"Processed: {csv_file.name}")
    print(f"  All rows are both: {all_both}")


print("\nDone!")