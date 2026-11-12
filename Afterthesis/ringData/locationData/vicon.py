from vicon_dssdk import ViconDataStream
import csv
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import sys

# ============================================================
# SETTINGS
# ============================================================

VICON_SERVER = "localhost:801"

experiment_name = sys.argv[1]

date_prefix = datetime.now().strftime("%d_%m")

CSV_FILE = f"{date_prefix}_{experiment_name}.csv"

OBJECTS = [
    ("sonoRo", "sonoRo"),
    ("soundLeft", "soundLeft"),
    ("soundRight", "soundRight")
]


# ============================================================
# CONNECT TO VICON
# ============================================================

client = ViconDataStream.Client()

print("Connecting to Vicon Tracker...")

while not client.IsConnected():

    try:
        client.Connect(VICON_SERVER)

    except ViconDataStream.DataStreamException as e:
        print("Connection failed:", e)
        time.sleep(1)

print("Connected!")

client.EnableSegmentData()


# ============================================================
# DATA FOR PLOT
# ============================================================

robot_x = []
robot_y = []

speakerL_x = []
speakerL_y = []

speakerR_x = []
speakerR_y = []


# ============================================================
# OPEN CSV
# ============================================================

with open(CSV_FILE, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    # Same headers as your old collector
    writer.writerow([
        "timestamp",
        "robot_x",
        "robot_y",
        "speakerL_x",
        "speakerL_y",
        "speakerR_x",
        "speakerR_y"
    ])

    csvfile.flush()

    print()
    print("Recording to:")
    print(os.path.abspath(CSV_FILE))
    print()
    print("Press Ctrl+C to stop.")
    print()

    try:

        while client.IsConnected():

            # ------------------------------------------------
            # Get next Vicon frame
            # ------------------------------------------------

            if not client.GetFrame():
                continue

            frame = client.GetFrameNumber()

            timestamp = datetime.now().isoformat(
                timespec="milliseconds"
            )

            # ------------------------------------------------
            # ROBOT
            # ------------------------------------------------

            result = client.GetSegmentGlobalTranslation(
                "sonoRo",
                "sonoRo"
            )

            print("Robot:", result)

            translation, occluded = result

            if occluded:
                print("WARNING: sonoRo is occluded")

            x_robot = translation[0]
            y_robot = translation[1]

            # ------------------------------------------------
            # SPEAKER LEFT
            # ------------------------------------------------

            result = client.GetSegmentGlobalTranslation(
                "soundLeft",
                "soundLeft"
            )

            print("Speaker L:", result)

            translation, occluded = result

            if occluded:
                print("WARNING: soundLeft is occluded")

            x_left = translation[0]
            y_left = translation[1]

            # ------------------------------------------------
            # SPEAKER RIGHT
            # ------------------------------------------------

            result = client.GetSegmentGlobalTranslation(
                "soundRight",
                "soundRight"
            )

            print("Speaker R:", result)

            translation, occluded = result

            if occluded:
                print("WARNING: soundRight is occluded")

            x_right = translation[0]
            y_right = translation[1]

            # ------------------------------------------------
            # SAVE FOR PLOT
            # ------------------------------------------------

            robot_x.append(x_robot)
            robot_y.append(y_robot)

            speakerL_x.append(x_left)
            speakerL_y.append(y_left)

            speakerR_x.append(x_right)
            speakerR_y.append(y_right)

            # ------------------------------------------------
            # WRITE CSV
            # -----------------------------------------------
            writer.writerow([
                timestamp,
                x_robot,
                y_robot,
                x_left,
                y_left,
                x_right,
                y_right
            ])

            csvfile.flush()

            # ------------------------------------------------
            # CONSOLE
            # ------------------------------------------------

            print(
                f"Frame {frame} | "
                f"{timestamp} | "
                f"Robot=({x_robot:.2f}, {y_robot:.2f}) | "
                f"L=({x_left:.2f}, {y_left:.2f}) | "
                f"R=({x_right:.2f}, {y_right:.2f})"
            )

    except KeyboardInterrupt:

        print()
        print("Recording stopped.")


# ============================================================
# PLOT
# ============================================================

print()
print("Creating plot...")

plt.figure(figsize=(8, 6))

if robot_x:
    plt.plot(
        robot_x,
        robot_y,
        label="Robot"
    )

if speakerL_x:
    plt.plot(
        speakerL_x,
        speakerL_y,
        label="Target 1"
    )

if speakerR_x:
    plt.plot(
        speakerR_x,
        speakerR_y,
        label="Target 2"
    )

plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")

plt.title("Vicon Trajectories")

plt.legend()
plt.grid(True)
plt.axis("equal")

plt.show()


# ============================================================
# DONE
# ============================================================

print()
print("CSV file saved:")
print(os.path.abspath(CSV_FILE))