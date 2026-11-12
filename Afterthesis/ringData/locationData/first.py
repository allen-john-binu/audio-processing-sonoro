from vicon_dssdk import ViconDataStream
import csv
import time
import os

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------

VICON_SERVER = "localhost:801"

OBJECTS = [
    "sonoRo",
    "soundRight",
    "soundLeft"
]

CSV_FILE = "vicon_positions.csv"

# -------------------------------------------------
# CONNECT TO VICON
# -------------------------------------------------

client = ViconDataStream.Client()

print("Connecting to Vicon Tracker...")

while not client.IsConnected():
    try:
        client.Connect(VICON_SERVER)
    except ViconDataStream.DataStreamException as e:
        print("Connection failed:", e)
        time.sleep(1)

print("Connected!")

# Enable the data we need
client.EnableSegmentData()

# -------------------------------------------------
# OPEN CSV FILE
# -------------------------------------------------

with open(CSV_FILE, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    # Header
    header = ["Frame", "Time"]

    for obj in OBJECTS:
        header.extend([
            obj + "_X",
            obj + "_Y",
            obj + "_Z"
        ])

    writer.writerow(header)

    print("Recording to:", os.path.abspath(CSV_FILE))
    print("Press Ctrl+C to stop.")
    print()

    start_time = time.time()

    try:

        while client.IsConnected():

            # Wait for next Vicon frame
            if not client.GetFrame():
                continue

            frame_number = client.GetFrameNumber()
            elapsed_time = time.time() - start_time

            row = [
                frame_number,
                elapsed_time
            ]

            # Get position of each object
            # Get position of each object
            for obj in OBJECTS:

                try:
                    translation, occluded = client.GetSegmentGlobalTranslation(
                        obj,
                        obj
                    )

                    if occluded:
                        # Object not visible in this frame
                        x = ""
                        y = ""
                        z = ""

                    else:
                        x = translation[0]
                        y = translation[1]
                        z = translation[2]

                except Exception as e:
                    print("Error reading", obj, ":", e)
                    x = ""
                    y = ""
                    z = ""

                row.extend([x, y, z])
            
            # Write the row
            writer.writerow(row)

            # Make sure data is written immediately
            csvfile.flush()

            print(
                "Frame:",
                frame_number,
                " | ",
                row[2:]
            )

    except KeyboardInterrupt:
        print("\nRecording stopped.")

print("CSV file saved:")
print(os.path.abspath(CSV_FILE))
