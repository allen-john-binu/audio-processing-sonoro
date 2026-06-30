import asyncio
import time
import numpy as np
import qtm
import xml.etree.ElementTree as ET
from collections import deque

# ============================================================
# CONFIGURATION
# ============================================================

QTM_ADDRESS = "10.126.17.54"
PASSWORD = "robot609"

BODY_LIST = ["speakerL", "speakerR", "robat"]

pose_message_buffer = deque(maxlen=5)

qtm_start_estimate = []
qtm_estimate_len = 5


# ============================================================
# UTILITIES
# ============================================================

def create_body_index(xml_string):
    """Extract a name-to-index dictionary from 6DOF settings XML."""
    xml = ET.fromstring(xml_string)

    body_to_index = {}
    for index, body in enumerate(xml.findall("*/Body/Name")):
        body_to_index[body.text.strip()] = index

    return body_to_index


# ============================================================
# MAIN RECEIVER
# ============================================================

async def pos_receiver(
    qtm_address=QTM_ADDRESS,
    password=PASSWORD,
):
    global pose_message_buffer
    global qtm_start_estimate

    connection = await qtm.connect(qtm_address)

    if connection is None:
        print("Failed to connect to QTM.")
        return

    print(f"Connected to {qtm_address}")

    async with qtm.TakeControl(connection, password):
        await connection.new()

    # Get body information
    xml_string = await connection.get_parameters(parameters=["6d"])
    body_index = create_body_index(xml_string)

    print("Detected bodies:")
    for body in body_index:
        print(f"  {body}")

    wanted_list = [b for b in BODY_LIST]

    def on_packet(packet):
        global pose_message_buffer
        global qtm_start_estimate

        now_ns = time.time_ns()

        # ------------------------------------------------------
        # Timestamp handling
        # ------------------------------------------------------
        timestamp = packet.timestamp
        rel_ns = timestamp * 1000

        if len(qtm_start_estimate) < qtm_estimate_len:
            inst_start_ns = now_ns - rel_ns
            qtm_start_estimate.append(inst_start_ns)
        else:
            inst_start_ns = int(np.mean(qtm_start_estimate))

        abs_timestamp_ns = inst_start_ns + rel_ns

        # ------------------------------------------------------
        # Get body data
        # ------------------------------------------------------
        info, bodies = packet.get_6d_euler()

        pose_message = {
            "timestamp": abs_timestamp_ns
        }

        for wanted_body in wanted_list:

            if wanted_body not in body_index:
                continue

            idx = body_index[wanted_body]

            if idx >= len(bodies):
                continue

            try:
                position, rotation = bodies[idx]

                x = float(position.x)
                y = float(position.y)
                angle = float(rotation.a1)

                pose_message[wanted_body] = (
                    x,
                    y,
                    angle
                )

            except Exception:
                continue

        # ------------------------------------------------------
        # Ensure all required bodies exist
        # ------------------------------------------------------
        required = ["speakerL", "speakerR", "robat"]

        if not all(k in pose_message for k in required):
            return

        # ------------------------------------------------------
        # Extract positions
        # ------------------------------------------------------
        s1 = np.array(
            pose_message["speakerL"][:2]
        )

        s2 = np.array(
            pose_message["speakerR"][:2]
        )

        robot = np.array(
            pose_message["robat"][:2]
        )

        # ------------------------------------------------------
        # Compute geometry
        # ------------------------------------------------------
        midpoint = (s1 + s2) / 2.0
        speaker_vec = s2 - s1
        mid_to_robot_vec = robot - midpoint

        norm1 = np.linalg.norm(speaker_vec)
        norm2 = np.linalg.norm(mid_to_robot_vec)

        if norm1 < 1e-9 or norm2 < 1e-9:
            return

        dot_product = np.dot(
            speaker_vec,
            mid_to_robot_vec
        )

        cos_theta = dot_product / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle_deg = np.degrees(
            np.arccos(cos_theta)
        )

        deviation = abs(90.0 - angle_deg)

        tolerance = 1.0

        if deviation < tolerance:
            print(
                f"Perpendicular "
                f"(angle={angle_deg:.2f}°, "
                f"deviation={deviation:.2f}°)"
            )
        else:
            print(
                f"Not perpendicular "
                f"(angle={angle_deg:.2f}°, "
                f"deviation={deviation:.2f}°)"
            )

        pose_message_buffer.append(
            pose_message
        )

    # ----------------------------------------------------------
    # Start streaming continuously
    # ----------------------------------------------------------
    try:
        print("\nStarting stream...\n")

        await connection.stream_frames(
            components=["6deuler"],
            on_packet=on_packet
        )

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping stream...")

    finally:
        try:
            await connection.stream_frames_stop()
        except Exception:
            pass

        connection.disconnect()
        print("Disconnected from QTM.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(
        pos_receiver(
            qtm_address=QTM_ADDRESS,
            password=PASSWORD
        )
    )