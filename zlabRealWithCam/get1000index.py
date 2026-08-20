import csv
from pathlib import Path

# Folder containing your CSV files
INPUT_FOLDER = Path("./mergedData/")

# Output CSV
OUTPUT_FILE = "1000indexResult.csv"


def find_first_two(csv_file):
    occurrences = []

    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_index, row in enumerate(reader):
            try:
                angle_1 = float(row["angle_1"])
            except (ValueError, TypeError):
                continue

            if angle_1 == -1000:
                occurrences.append(row_index)

                # Stop after finding the first 2
                if len(occurrences) == 2:
                    break

    if len(occurrences) == 2:
        first = occurrences[0]
        second = occurrences[1]

        if first >= 23 and second >= 23:
            above_23 = "both"
        elif first < 23 and second >= 23:
            above_23 = "onlySecond"
        else:
            above_23 = ""

        return first, second, second - first, above_23

    return (
        occurrences[0] if len(occurrences) > 0 else None,
        None,
        None,
        "",
    )


results = []

for csv_file in sorted(INPUT_FOLDER.glob("*.csv")):

    occurrence_1, occurrence_2, difference, above_23 = find_first_two(csv_file)

    results.append({
        "filename": csv_file.name,
        "occurrence_1": occurrence_1,
        "occurrence_2": occurrence_2,
        "difference": difference,
        "above_23": above_23,
    })


with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "occurrence_1",
            "occurrence_2",
            "difference",
            "above_23",
        ],
    )

    writer.writeheader()
    writer.writerows(results)

print(f"Done. Results written to: {OUTPUT_FILE}")