import asyncio
import numpy as np
import qtm
import xml.etree.ElementTree as ET

        #    self.v = 0.6 faster
        #     self.h_0 = 0.051
        #     self.h_b = 0.045
            
        #             self.v = 0.5 slower

        # self.h_0 = 0.051

        # self.h_b = 0.045



QTM_ADDRESS = "10.126.17.54"
PASSWORD = "robot609"


def create_body_index(xml_string):
    xml = ET.fromstring(xml_string)

    body_to_index = {}

    for index, body in enumerate(xml.findall("*/Body/Name")):
        body_to_index[body.text.strip()] = index

    return body_to_index


def calculate_perpendicular_angle(sonoRo, speakerL, speakerR):

    # --------------------------------------------------------
    # Midpoint between speakerL and speakerR
    # --------------------------------------------------------

    midpoint = (
        (speakerL[0] + speakerR[0]) / 2.0,
        (speakerL[1] + speakerR[1]) / 2.0
    )

    # --------------------------------------------------------
    # Vector along speakerL -> speakerR
    # --------------------------------------------------------

    speaker_vector = np.array([
        speakerR[0] - speakerL[0],
        speakerR[1] - speakerL[1]
    ])

    # --------------------------------------------------------
    # Vector midpoint -> sonoRo
    # --------------------------------------------------------

    sono_vector = np.array([
        sonoRo[0] - midpoint[0],
        sonoRo[1] - midpoint[1]
    ])

    # --------------------------------------------------------
    # Calculate angle between the two vectors
    # --------------------------------------------------------

    speaker_length = np.linalg.norm(speaker_vector)
    sono_length = np.linalg.norm(sono_vector)

    if speaker_length == 0 or sono_length == 0:
        return None, None

    dot_product = np.dot(speaker_vector, sono_vector)

    cos_angle = dot_product / (speaker_length * sono_length)

    # Protect against numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cos_angle))

    # --------------------------------------------------------
    # Convert to the smaller angle between the two lines
    # This gives a value between 0 and 90 degrees.
    # --------------------------------------------------------

    if angle > 90:
        angle = 180 - angle

    deviation = abs(angle - 90.0)

    return angle, deviation


async def collect_data():

    connection = await qtm.connect(QTM_ADDRESS)

    if connection is None:
        print("Failed to connect to QTM.")
        return

    async with qtm.TakeControl(connection, PASSWORD):
        await connection.new()

    xml_string = await connection.get_parameters(parameters=["6d"])
    body_index = create_body_index(xml_string)

    print("Detected bodies:")
    for body in body_index:
        print(f"  {body}")

    required_bodies = [
        "sonoRo",
        "speakerL",
        "speakerR"
    ]

    for body in required_bodies:
        if body not in body_index:
            print(f"ERROR: Body '{body}' not found.")
            connection.disconnect()
            return

    # Latest calculated values
    latest_angle = None
    latest_deviation = None

    def on_packet(packet):
        nonlocal latest_angle, latest_deviation

        try:

            info, bodies = packet.get_6d_euler()

            # ------------------------------------------------
            # Get positions
            # ------------------------------------------------

            sonoRo_position, _ = bodies[
                body_index["sonoRo"]
            ]

            speakerL_position, _ = bodies[
                body_index["speakerL"]
            ]

            speakerR_position, _ = bodies[
                body_index["speakerR"]
            ]

            # ------------------------------------------------
            # Convert to XY coordinates
            # ------------------------------------------------

            sonoRo = (
                float(sonoRo_position.x),
                float(sonoRo_position.y)
            )

            speakerL = (
                float(speakerL_position.x),
                float(speakerL_position.y)
            )

            speakerR = (
                float(speakerR_position.x),
                float(speakerR_position.y)
            )

            # ------------------------------------------------
            # Calculate perpendicularity
            # ------------------------------------------------

            latest_angle, latest_deviation = \
                calculate_perpendicular_angle(
                    sonoRo,
                    speakerL,
                    speakerR
                )

        except Exception as e:
            print(f"\nError processing frame: {e}")

    try:

        await connection.stream_frames(
            components=["6deuler"],
            on_packet=on_packet
        )

        print("\nStreaming started.")
        print("Checking perpendicularity every 0.5 seconds.")
        print("Ideal angle = 90 degrees.")
        print("Press Ctrl+C to stop.\n")

        while True:

            await asyncio.sleep(0.5)

            if latest_angle is not None:

                print(
                    f"Angle: {latest_angle:7.2f}°   "
                    f"Deviation from 90°: "
                    f"{latest_deviation:6.2f}°"
                )

            else:
                print("Waiting for QTM data...")

    except KeyboardInterrupt:

        print("\nStopping...")

    finally:

        try:
            await connection.stream_frames_stop()
        except Exception:
            pass

        connection.disconnect()

        print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(collect_data())