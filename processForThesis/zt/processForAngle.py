import csv
from pathlib import Path

# Input and output folders
INPUT_FOLDER = Path("./mergedData/")
OUTPUT_FOLDER = Path("./processedData/")

# Create output folder if it doesn't exist
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def process_csv(input_file, output_file):
    with open(input_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Keep the original column order
        fieldnames = reader.fieldnames
        rows_to_keep = []

        for original_index, row in enumerate(reader):
            try:
                angle_1 = float(row["angle_1"])
            except (ValueError, TypeError):
                # Keep rows where angle_1 cannot be converted
                rows_to_keep.append(row)
                continue

            if angle_1 == -1000:

                if original_index >= 23:
                    # First -1000 at index >= 23:
                    # stop processing completely.
                    break

                else:
                    # -1000 before index 23:
                    # remove only this row.
                    continue

            # Normal row
            rows_to_keep.append(row)

    # Write processed CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_keep)


# Process every CSV
for csv_file in sorted(INPUT_FOLDER.glob("*.csv")):
    output_file = OUTPUT_FOLDER / csv_file.name

    process_csv(csv_file, output_file)

    print(f"Processed: {csv_file.name}")

print("\nDone!")