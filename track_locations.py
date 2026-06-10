import asyncio
import csv
import os
import sys
import time
from datetime import datetime
import xml.etree.ElementTree as ET

import numpy as np
import qtm


# ============================================================
# CONFIGURATION
# ============================================================

QTM_ADDRESS = "10.126.17.54"
PASSWORD = "robot609"

SAVE_DIR = "ztLabCollection/dataFromQualis"   # <-- change as needed

BODY_NAMES = {
    "robat": "robot",
    "speakerL": "speakerL",
    "speakerR": "speakerR",
}

qtm_start_estimate = []
qtm_estimate_len = 5


# ============================================================
# UTILITIES
# ============================================================

def create_body_index(xml_string):
    xml = ET.fromstring(xml_string)

    body_to_index = {}
    for index, body in enumerate(xml.findall("*/Body/Name")):
        body_to_index[body.text.strip()] = index

    return body_to_index


# ============================================================
# MAIN LOGGER
# ============================================================

async def collect_data(output_csv):

    global qtm_start_estimate

    connection = await qtm.connect(QTM_ADDRESS)

    if connection is None:
        print("Failed to connect to QTM.")
        return

    async with qtm.TakeControl(connection, PASSWORD):
        await connection.new()

    xml_string = await connection.get_parameters(parameters=["6d"])
    body_index = create_body_index(xml_string)

    print("Detected bodies:")
    for k in body_index:
        print(f"  {k}")

    frame_count = 0
    last_report_time = time.time()

    csv_file = open(output_csv, "w", newline="")
    writer = csv.writer(csv_file)

    writer.writerow([
        "timestamp",
        "robot_x",
        "robot_y",
        "speakerL_x",
        "speakerL_y",
        "speakerR_x",
        "speakerR_y",
    ])

    def on_packet(packet):
        nonlocal frame_count, last_report_time

        now_ns = time.time_ns()

        timestamp = packet.timestamp
        rel_ns = timestamp * 1000

        if len(qtm_start_estimate) < qtm_estimate_len:
            inst_start_ns = now_ns - rel_ns
            qtm_start_estimate.append(inst_start_ns)
        else:
            inst_start_ns = int(np.mean(qtm_start_estimate))

        abs_timestamp_ns = inst_start_ns + rel_ns

        iso_timestamp = datetime.fromtimestamp(
            abs_timestamp_ns / 1e9
        ).isoformat(timespec="microseconds")

        info, bodies = packet.get_6d_euler()

        row = {
            "robot_x": np.nan,
            "robot_y": np.nan,
            "speakerL_x": np.nan,
            "speakerL_y": np.nan,
            "speakerR_x": np.nan,
            "speakerR_y": np.nan,
        }

        for qtm_name, csv_prefix in BODY_NAMES.items():

            if qtm_name not in body_index:
                continue

            idx = body_index[qtm_name]

            try:
                position, rotation = bodies[idx]

                x = float(position.x)
                y = float(position.y)

                row[f"{csv_prefix}_x"] = x
                row[f"{csv_prefix}_y"] = y

            except Exception:
                pass

        writer.writerow([
            iso_timestamp,
            row["robot_x"],
            row["robot_y"],
            row["speakerL_x"],
            row["speakerL_y"],
            row["speakerR_x"],
            row["speakerR_y"],
        ])

        frame_count += 1

        now = time.time()
        if now - last_report_time >= 1.0:
            print(f"Rate: {frame_count} Hz")

            csv_file.flush()

            frame_count = 0
            last_report_time = now

    try:
        print(f"\nLogging to:\n{output_csv}\n")
        print("Press Ctrl+C to stop.\n")

        await connection.stream_frames(
            components=["6deuler"],
            on_packet=on_packet
        )

        print("Streaming started...")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass

    except KeyboardInterrupt:
        print("\nStopping logger...")

    finally:
        csv_file.flush()
        csv_file.close()

        try:
            await connection.stream_frames_stop()
        except Exception:
            pass

        connection.disconnect()

        print("CSV saved.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python collect_data.py <filename>"
        )
        sys.exit(1)

    filename = sys.argv[1]

    if not filename.endswith(".csv"):
        filename += ".csv"

    os.makedirs(SAVE_DIR, exist_ok=True)

    output_csv = os.path.join(SAVE_DIR, filename)

    asyncio.run(collect_data(output_csv))