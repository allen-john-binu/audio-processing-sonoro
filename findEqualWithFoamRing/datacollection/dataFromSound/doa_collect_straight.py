#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drive the Thymio straight at a constant speed while continuously estimating
DOA from the mic array and logging each estimate to a CSV file.
Press Ctrl+C to stop the robot, close the file, and exit.
"""

import csv
import sys
import time
import queue
import argparse
from datetime import datetime

import numpy as np
import sounddevice as sd

# TODO: adjust these to your actual module names
from utilities import das_filter_UCA, get_card, calc_dBrms
from robot_control import RobotMove  # module containing the RobotMove class

# ------------------------- Configuration -------------------------
FS = 16000                       # sampling rate (Hz)
MIC_NAME = "reSpeaker"           # substring identifying the mic array
N_MICS = 4                       # number of mics fed to the DOA estimator
ARRAY_RADIUS = 0.04625           # UCA radius (m)
ARRAY_SHIFT = 45                 # array geometry rotation offset (deg)
BANDPASS = [1000, 2000]         # Hz band used for DOA estimation
BLOCKSIZE = 256              # samples per audio block (~128 ms @ 16 kHz)
THETA = np.linspace(0, 360, 361)
DB_RMS_THRESHOLD = -50.0

FORWARD_SPEED = 30
TURN_SPEED = 100                 # required by RobotMove's constructor, unused here
# --------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drive the Thymio straight while logging DOA + SPL to CSV."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default=f"doa_log_{datetime.now():%Y%m%d_%H%M%S}.csv",
        help="Output CSV filename (default: doa_log_<timestamp>.csv)",
    )
    return parser.parse_args()

audio_q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

def get_spatial_spectrum(mic_block):

    mono = np.mean(mic_block, axis=1)
    db_rms = calc_dBrms(mono)

    if db_rms < DB_RMS_THRESHOLD:
        print("with less dm")
        return None

    theta, spatial_resp, _, _, _ = das_filter_UCA(
        mic_block,
        FS,
        N_MICS,
        ARRAY_RADIUS,
        ARRAY_SHIFT,
        BANDPASS,
        np.linspace(0, 360, 361),
        show=False,
    )

    return theta, spatial_resp, db_rms


def main():
    device_idx = get_card(sd.query_devices(), MIC_NAME)
    print("Device index:", device_idx)
    print(sd.query_devices(device_idx))
    if device_idx is None:
        raise RuntimeError(f"Microphone array '{MIC_NAME}' not found.")
    print(f"Mic array found (device index {device_idx})")

    robot = RobotMove(
        forward_speed=FORWARD_SPEED,
        turn_speed=TURN_SPEED,
        left_sensor_threshold=200,
        right_sensor_threshold=200,
        critical_level=-3,
        trigger_level=-40,
        ring=None,
        ground_sensors_bool=False,
    )
    print("Thymio connected.")
    
    print(sd.__version__)
    print(sd.get_portaudio_version())

    stream = sd.InputStream(
        device=device_idx,
        samplerate=FS,
        channels=6,
        dtype="float32",
        blocksize=BLOCKSIZE,      # same as working script
        latency="low",
        callback=audio_callback,
    )

    args = parse_args()
    CSV_PATH = args.filename
    csv_file = open(CSV_PATH, "w", newline="")
    
    writer = csv.writer(csv_file)
    header = (
        ["timestamp", "elapsed_s", "db_rms"]
        + [f"{int(t)}" for t in THETA]
    )

    writer.writerow(header)

    stream.start()
    
    robot.move_forward()
    start_time = time.time()
    print(f"Driving straight, logging DOA to '{CSV_PATH}'. Press Ctrl+C to stop.\n")

    try:
        while True:
            try:
                block = audio_q.get(timeout=1.0)
            except queue.Empty:
                print("Queue empty")
                print("Stream active:", stream.active)
                continue

            mic_block = block[:, 2:6]

            result = get_spatial_spectrum(mic_block)

            if result is None:
                print("im here")
                continue

            theta, spatial_resp, db_rms = result

            elapsed = time.time() - start_time

            row = (
                [
                    datetime.now().isoformat(),
                    elapsed,
                    db_rms,
                ]
                + spatial_resp.tolist()
            )

            writer.writerow(row)
            csv_file.flush()

            print(
                f"t={elapsed:6.2f}s   dB={db_rms:6.1f}"
            )

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.stop()
        stream.stop()
        stream.close()
        csv_file.close()
        print(f"Saved log to '{CSV_PATH}'")


if __name__ == "__main__":
    main()