###################################################################################################
# SINGLE-SEQUENCE ROBAT / THYMIO CONTROL
#
# Processing sequence:
#
#   1. Acquire/process one DOA measurement
#   2. Check dB SPL threshold
#   3. Run ring attractor using that DOA
#   4. Compute bump angle
#   5. Turn robot according to bump angle
#   6. Move forward for 0.5 seconds
#   7. Stop
#   8. Wait 0.2 seconds
#   9. Acquire/process the NEXT DOA
#  10. Repeat
#
# IMPORTANT:
# AudioProcessor itself is still responsible for microphone acquisition.
# The robot-control/DOA-processing path below is strictly sequential.
###################################################################################################

from thymiodirect import Connection
from thymiodirect import Thymio

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf

import argparse
import queue
import datetime
import time
import random
import os
import csv
import threading
import math
import logging
import signal as os_signal

import pandas as pd
import netifaces as ni

from random import seed as py_seed

from functions.get_card import get_card
from functions.save_data_to_csv import save_data_to_csv
from functions.utilities import (
    pascal_to_dbspl,
    calc_native_freqwise_rms,
    interpolate_freq_response
)
from functions.save_data_to_xml import save_data_to_xml
from functions.matched_filter import matched_filter
from functions.detect_peaks import detect_peaks

from AudioProcessor import AudioProcessor

import utils


###################################################################################################
# RANDOM SEED
###################################################################################################

seed = 300

np.random.seed(seed)
py_seed(seed)


###################################################################################################
# LOGGING
###################################################################################################

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_FILE = os.path.join(
    LOG_DIR,
    f"target_tracking_{log_timestamp}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ROBAT")

logger.info("============================================================")
logger.info("Starting single-sequence ROBAT control system")
logger.info("Log file: %s", LOG_FILE)
logger.info("============================================================")


###################################################################################################
# PARAMETERS
###################################################################################################

timestamp_queue = queue.Queue()

timestamp_bool = True
recording_bool = True

behavior = "dynamic_movement"

# -----------------------------------------------------------------------------------------------
# DOA / SOUND PARAMETERS
# -----------------------------------------------------------------------------------------------

trigger_level = 80       # dB SPL threshold
critical_level = 99     # dB SPL

c = 343
fs = 48000

rec_samplerate = 48000

input_buffer_time = 0.04
block_size = int(input_buffer_time * fs)

channels = 5

mic_spacing = 0.018

nfft = 512

ref = channels // 2

method = "DAS"

avar_theta = None
theta_values = []

theta_das = np.linspace(-90, 90, 61)

N_peaks = 2


# -----------------------------------------------------------------------------------------------
# CHIRP PARAMETERS
# -----------------------------------------------------------------------------------------------

duration_out = 10e-3

if behavior == "attraction":
    silence_post = 10
    amplitude = 0

elif behavior == "repulsion":
    silence_post = 10
    amplitude = 0

elif behavior == "dynamic_movement":
    silence_post = 10
    amplitude = 0

else:
    silence_post = 10
    amplitude = 0


t = np.linspace(
    0,
    duration_out,
    int(fs * duration_out)
)

start_f = 24e3
end_f = 2e3

sweep = signal.chirp(
    t,
    start_f,
    t[-1],
    end_f
)

sweep *= signal.windows.tukey(
    sweep.size,
    0.2
)

sweep *= 0.8

silence_samples_post = int(
    silence_post * fs / 1000
)

silence_vec_post = np.zeros(
    (silence_samples_post,)
)

post_silence_sig = np.concatenate(
    (
        sweep,
        silence_vec_post
    )
)

full_sig = post_silence_sig

stereo_sig = np.hstack(
    [
        full_sig.reshape(-1, 1),
        full_sig.reshape(-1, 1)
    ]
)

data = amplitude * np.float32(stereo_sig)

out_blocksize = int(len(data))

logger.info(
    "Audio output block size: %d",
    out_blocksize
)


###################################################################################################
# MICROPHONE FILTER PARAMETERS
###################################################################################################

auto_hipas_freq = int(
    343 / (
        2 * (
            mic_spacing *
            (channels - 1)
        )
    )
)

auto_lowpas_freq = int(
    343 / (
        2 * mic_spacing
    )
)

highpass_freq = auto_hipas_freq
lowpass_freq = auto_lowpas_freq

logger.info(
    "Automatic high-pass frequency: %d Hz",
    highpass_freq
)

logger.info(
    "Automatic low-pass frequency: %d Hz",
    lowpass_freq
)

freq_range = [
    start_f,
    end_f
]

cutoff = auto_hipas_freq

sos = signal.butter(
    1,
    cutoff,
    "hp",
    fs=fs,
    output="sos"
)


###################################################################################################
# MICROPHONE SENSITIVITY
###################################################################################################

script_dir = os.path.dirname(
    os.path.abspath(__file__)
)

sensitivity_path = os.path.join(
    script_dir,
    "Knowles_SPH0645LM4H-B_sensitivity.csv"
)

sensitivity = pd.read_csv(
    sensitivity_path
)

analyzed_buffer_time = 0.01

block_size_analyzed_buffer = int(
    analyzed_buffer_time * fs
)

in_sig = np.zeros(
    block_size_analyzed_buffer,
    dtype=np.float32
)

centrefreqs, freqrms = calc_native_freqwise_rms(
    in_sig,
    fs
)

freqs = np.array(
    sensitivity.iloc[:, 0]
)

sens_freqwise_rms = np.array(
    sensitivity.iloc[:, 1]
)

interp_sensitivity = interpolate_freq_response(
    [
        freqs,
        sens_freqwise_rms
    ],
    centrefreqs
)

frequency_band = [
    2e3,
    20e3
]

tgtmic_relevant_freqs = np.logical_and(
    centrefreqs >= frequency_band[0],
    centrefreqs <= frequency_band[1]
)


###################################################################################################
# THYMIO PARAMETERS
###################################################################################################

speed = 80
turn_speed = 100

left_sensor_threshold = 300
right_sensor_threshold = 300


###################################################################################################
# RING ATTRACTOR
###################################################################################################

class RA:

    def __init__(self):

        self.Ns = 120

        self.v = 0.5

        self.h_0 = 0.051

        self.h_b = 0.045

        self.h_ext = None

        self.beta = 400.0

        self.sigma_ang = (
            2 * np.pi / self.Ns
        )

        self.thetas = np.linspace(
            -np.pi,
            np.pi,
            self.Ns,
            endpoint=False
        )

        self.spins = np.random.choice(
            [1, 0],
            size=self.Ns
        )

        self.updates_per_step = int(
            round(self.Ns * 10)
        )

        logger.info(
            "Ring attractor initialized: Ns=%d, beta=%.2f, updates_per_step=%d",
            self.Ns,
            self.beta,
            self.updates_per_step
        )


###################################################################################################
# DOA NORMALIZATION
###################################################################################################

def doa_to_row(
    doa_estimate,
    global_min=0.0000000005,
    global_max=0.0000068100,
    NORMALIZING_FACTOR=0.1900,
):
    """
    Replicates the normalization used by the original ring-attractor
    implementation.

    Input:
        120-element DOA spectrum.

    Output:
        normalized 120-element external field.
    """

    denom = global_max - global_min

    norm_arr = []

    for v in doa_estimate:

        if v == 0.0:

            norm_arr.append(0.0)

        else:

            if denom == 0:

                norm = 0.0

            else:

                norm = (
                    (v - global_min)
                    / denom
                )

            norm_arr.append(
                norm * NORMALIZING_FACTOR
            )

    return norm_arr


###################################################################################################
# RING CSV LOGGER
###################################################################################################

def log_ring_state(
    time_step,
    angles,
    doa_estimate,
    phi_in_degrees,
    spins,
    state,
    doa_acquisition_time=None,
    ring_processing_time=None,
    base_folder="ring_results",
    session_name="target_tracking",
    max_angles=2
):
    """
    Logs ring-attractor state.

    Additional timing fields:
        doa_acquisition_time:
            Time required to obtain the considered DOA.

        ring_processing_time:
            Time required to process that DOA through the
            ring attractor and calculate the bump angle.
    """

    # ---------- INIT ----------
    if "file_path" not in state:

        os.makedirs(
            base_folder,
            exist_ok=True
        )

        timestamp = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        file_name = f"{session_name}_{timestamp}.csv"

        file_path = os.path.join(
            base_folder,
            file_name
        )

        with open(
            file_path,
            mode="w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            header = [
                "timestamp",
                "time_step",
                "phi_in_degrees",

                # NEW TIMING FIELDS
                "doa_acquisition_time",
                "ring_processing_time",
            ]

            header += [
                f"angle_{i}"
                for i in range(max_angles)
            ]

            header += [
                f"doa_{i}"
                for i in range(len(doa_estimate))
            ]

            header += [
                f"spin_{i}"
                for i in range(len(spins))
            ]

            writer.writerow(header)

        state["file_path"] = file_path

        logger.info(
            "Ring-state CSV created: %s",
            file_path
        )

    # ---------- APPEND ----------

    padded_angles = list(
        angles[:max_angles]
    )

    if len(padded_angles) < max_angles:

        padded_angles.extend(
            [-1000] *
            (
                max_angles -
                len(padded_angles)
            )
        )

    row = [
        datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        ),

        time_step,

        phi_in_degrees,

        # NEW TIMING FIELDS
        doa_acquisition_time,
        ring_processing_time,
    ]

    row += padded_angles
    row += list(doa_estimate)
    row += list(spins)

    with open(
        state["file_path"],
        mode="a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(row)
###################################################################################################
# ROBOT CLASS
###################################################################################################

class RobotMove:

    def __init__(
        self,
        forward_speed,
        turn_speed,
        left_sensor_threshold,
        right_sensor_threshold,
        critical_level,
        trigger_level,
        ring,
        ground_sensors_bool=True
    ):

        self.forward_speed = forward_speed
        self.turn_speed = turn_speed

        self.left_sensor_threshold = (
            left_sensor_threshold
        )

        self.right_sensor_threshold = (
            right_sensor_threshold
        )

        self.ground_sensors_bool = (
            ground_sensors_bool
        )

        self.critical_level = critical_level
        self.trigger_level = trigger_level

        self.running = True

        self.turning_angle = 5

        self.ring = ring

        self.heading = 0

        self.max_count = 0

        self.time_step = 0

        self.log_ring_state = {}

        logger.info(
            "Initializing Thymio Robot"
        )

        port = Connection.serial_default_port()

        th = Thymio(
            serial_port=port,
            on_connect=lambda node_id:
                logger.info(
                    "Thymio %s connected",
                    node_id
                )
        )

        th.connect()

        self.robot = th[
            th.first_node()
        ]

        logger.info(
            "Robot connected"
        )

        if self.ground_sensors_bool:

            logger.info(
                "Initializing ground sensors..."
            )

            time.sleep(2)

            logger.info(
                "ground.delta L/R = %s",
                self.robot[
                    "prox.ground.delta"
                ]
            )

            logger.info(
                "ground.reflected L/R = %s",
                self.robot[
                    "prox.ground.reflected"
                ]
            )

        self.stop_bool = False



    def wait_until_path_clear(self, poll_interval=0.02):
        """
        Hard safety gate.

        If an obstacle is detected in front of the robot:
        - stop the motors
        - do not process a DOA
        - do not run the ring attractor
        - do not turn
        - do not move

        This function blocks until the obstacle is removed or the
        program is stopped.
        """

        obstacle_logged = False

        while self.running:

            if self.obstacle_in_front():

                if not obstacle_logged:
                    logger.warning(
                        "OBSTACLE DETECTED IN FRONT - "
                        "STOPPING ALL PROCESSING"
                    )
                    obstacle_logged = True

                self.stop()

                time.sleep(poll_interval)

                continue

            if obstacle_logged:

                logger.info(
                    "OBSTACLE CLEARED - "
                    "RESUMING PROCESSING"
                )

            return True

        return False
    ################################################################################################
    # TURN TIME
    ################################################################################################

    def angle_to_time(
        self,
        angle,
        speed
    ):

        A = 612.33
        B = -0.94

        t = A * speed ** B

        return (
            t *
            abs(angle) /
            360
        )


    ################################################################################################
    # SAFETY
    ################################################################################################

    def check_stop_all_motion(self):

        if self.robot is not None:

            ground = self.robot[
                "prox.ground.delta"
            ]

            if (
                ground[0] < 10
                or
                ground[1] < 10
            ):

                logger.warning(
                    "Robot safety stop: ground sensor detected lift"
                )

                return True

        else:

            logger.error(
                "Robot object is None"
            )

            self.stop()

        return False


    def obstacle_in_front(self, threshold=3000):
        """
        Return True when an obstacle is detected by the front
        horizontal proximity sensors.
        """

        if self.robot is None:
            return False

        front = self.robot["prox.horizontal"]

        detected = (
            front[0] > threshold or
            front[1] > threshold or
            front[2] > threshold or
            front[3] > threshold or
            front[4] > threshold
        )

        return detected


    ################################################################################################
    # MOTOR CONTROL
    ################################################################################################

    def move_forward(self):

        if self.robot is None:

            logger.error(
                "Cannot move forward: robot is None"
            )

            self.stop()

            return False

        if self.check_stop_all_motion():

            self.stop_bool = True

            self.stop()

            return False

        self.robot["leds.top"] = [
            0,
            255,
            0
        ]

        self.robot["leds.bottom.right"] = [
            0,
            255,
            0
        ]

        self.robot["leds.bottom.left"] = [
            0,
            255,
            0
        ]

        self.robot[
            "motor.left.target"
        ] = self.forward_speed

        self.robot[
            "motor.right.target"
        ] = self.forward_speed

        logger.info(
            "Robot moving forward: speed=%d",
            self.forward_speed
        )

        return True


    def move_with_speed(
        self,
        vL,
        vR
    ):

        if self.robot is None:

            self.stop()

            return

        if self.check_stop_all_motion():

            self.stop_bool = True

            self.stop()

            return

        self.robot[
            "motor.left.target"
        ] = vL

        self.robot[
            "motor.right.target"
        ] = vR


    ################################################################################################
    # ROTATION
    ################################################################################################

    def rotate_right(
        self,
        angle
    ):

        if self.robot is None:

            self.stop()

            return

        counter = self.angle_to_time(
            angle,
            self.turn_speed
        )

        logger.info(
            "Rotating right: %.2f degrees, %.3f seconds",
            angle,
            counter
        )

        start_time = time.time()

        while (
            time.time() - start_time
            < counter
        ):

            if self.check_stop_all_motion():

                self.stop_bool = True

                break

            self.robot[
                "motor.left.target"
            ] = (
                self.turn_speed
                +
                self.turn_speed // 2
            )

            self.robot[
                "motor.right.target"
            ] = -self.turn_speed

        self.stop()


    def rotate_left(
        self,
        angle
    ):

        if self.robot is None:

            self.stop()

            return

        counter = self.angle_to_time(
            angle,
            self.turn_speed
        )

        logger.info(
            "Rotating left: %.2f degrees, %.3f seconds",
            angle,
            counter
        )

        start_time = time.time()

        while (
            time.time() - start_time
            < counter
        ):

            if self.check_stop_all_motion():

                self.stop_bool = True

                break

            self.robot[
                "motor.left.target"
            ] = -self.turn_speed

            self.robot[
                "motor.right.target"
            ] = (
                self.turn_speed
                +
                self.turn_speed // 2
            )

        self.stop()


    def smooth_rotate_right(
        self,
        angle
    ):

        if self.robot is None:

            self.stop()

            return

        counter = self.angle_to_time(
            angle,
            self.turn_speed
        )

        logger.info(
            "Smooth rotating right: %.2f degrees, %.3f seconds",
            angle,
            counter
        )

        start_time = time.time()

        while (
            time.time() - start_time
            < counter
        ):

            if self.check_stop_all_motion():

                self.stop_bool = True

                break

            self.robot[
                "motor.left.target"
            ] = (
                self.turn_speed
                +
                self.turn_speed // 2
            )

            self.robot[
                "motor.right.target"
            ] = (
                -self.turn_speed
                +
                self.turn_speed // 2
            )

        self.stop()


    def smooth_rotate_left(
        self,
        angle
    ):

        if self.robot is None:

            self.stop()

            return

        counter = self.angle_to_time(
            angle,
            self.turn_speed
        )

        logger.info(
            "Smooth rotating left: %.2f degrees, %.3f seconds",
            angle,
            counter
        )

        start_time = time.time()

        while (
            time.time() - start_time
            < counter
        ):

            if self.check_stop_all_motion():

                self.stop_bool = True

                break

            self.robot[
                "motor.left.target"
            ] = (
                -self.turn_speed
                +
                self.turn_speed // 2
            )

            self.robot[
                "motor.right.target"
            ] = (
                self.turn_speed
                +
                self.turn_speed // 2
            )

        self.stop()


    ################################################################################################
    # STOP
    ################################################################################################

    def stop(self):

        if self.robot is None:
            return

        self.robot[
            "motor.left.target"
        ] = 0

        self.robot[
            "motor.right.target"
        ] = 0

        self.robot["leds.top"] = [
            0,
            0,
            0
        ]

        self.robot[
            "leds.bottom.right"
        ] = [
            0,
            0,
            0
        ]

        self.robot[
            "leds.bottom.left"
        ] = [
            0,
            0,
            0
        ]

        self.robot[
            "leds.circle"
        ] = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ]


###################################################################################################
# RING ATTRACTOR PROCESSING
###################################################################################################

def run_ring_attractor(
    ring,
    doa_estimate
):
    """
    Run the ring attractor and return:

        bump_angle
        normalized DOA
        ring processing time
    """

    ring_start_time = time.perf_counter()

    logger.info(
        "Starting ring-attractor processing"
    )

    row_from_doa = doa_to_row(
        doa_estimate=doa_estimate
    )

    ring.h_ext = row_from_doa

    for _ in range(
        ring.updates_per_step
    ):

        i = np.random.randint(
            0,
            ring.Ns
        )

        delta_H = utils.compute_delta_H(
            ring,
            i
        )

        if delta_H < 0:

            ring.spins[i] = (
                1 -
                ring.spins[i]
            )

        else:

            p = np.exp(
                -ring.beta *
                delta_H
            )

            if np.random.rand() < p:

                ring.spins[i] = (
                    1 -
                    ring.spins[i]
                )

    active_indices = np.where(
        ring.spins == 1
    )[0]

    n_active = len(
        active_indices
    )

    if n_active > 0:

        phi = np.angle(
            np.sum(
                np.exp(
                    1j *
                    ring.thetas[
                        ring.spins == 1
                    ]
                )
            )
        )

    else:

        phi = 0.0

    phi_in_degrees = math.degrees(
        phi
    )

    ring_processing_time = (
        time.perf_counter() -
        ring_start_time
    )

    logger.info(
        "Ring attractor complete: "
        "bump angle=%.2f degrees | "
        "active neurons=%d | "
        "processing time=%.6f s",
        phi_in_degrees,
        n_active,
        ring_processing_time
    )

    return (
        phi_in_degrees,
        row_from_doa,
        ring_processing_time
    )
###################################################################################################
# SINGLE DOA -> ACTION CYCLE
###################################################################################################

def process_one_doa_cycle(
    audio_processor,
    robot_move,
    ring,
    args,
    counter
):
    """
    Performs EXACTLY ONE processing cycle.

    No queue is used for robot decisions.

    Sequence:

        DOA
        -> threshold
        -> ring attractor
        -> bump angle
        -> turn
        -> forward 0.5 sec
        -> stop
        -> wait 0.2 sec
    """
    logger.info(
        "------------------------------------------------------------"
    )
    
    # ---------------------------------------------------------------------------------------------
    # HARD OBSTACLE GATE
    # ---------------------------------------------------------------------------------------------

    if not robot_move.wait_until_path_clear():

        logger.info(
            "Processing stopped because robot is no longer running."
        )

        return counter


    logger.info(
        "Path clear. Requesting next DOA measurement."
    )
    
    
    # ---------------------------------------------------------------------------------------------
    # GET ONE DOA
    # ---------------------------------------------------------------------------------------------

    timestamp = datetime.datetime.timestamp(
        datetime.datetime.now(
            datetime.timezone.utc
        )
    )

    doa_start_time = time.perf_counter()

    args.angle, dB_SPL_level, doa_estimate = (
        audio_processor.update_das_with_doa()
    )

    doa_acquisition_time = (
        time.perf_counter() -
        doa_start_time
    )

    logger.info(
        "DOA acquisition/processing time: %.6f s",
        doa_acquisition_time
    )
    
    if robot_move.obstacle_in_front():

        logger.warning(
            "Obstacle detected immediately after DOA acquisition. "
            "Discarding this DOA."
        )

        robot_move.stop()

        return counter

    db_spl = float(
        dB_SPL_level[0]
    )

    counter += 1

    logger.info(
        "DOA measurement #%d: raw angle=%s, dB_SPL=%.2f",
        counter,
        args.angle,
        db_spl
    )

    timestamp_queue.put(
        [
            timestamp,
            db_spl,
            args.angle,
            "doa_measurement"
        ]
    )


    # ---------------------------------------------------------------------------------------------
    # CREATE 120-ELEMENT DOA ARRAY
    # ---------------------------------------------------------------------------------------------

    full_array_doa = np.zeros(
        120
    )

    angles = np.arange(
        90,
        -91,
        -3
    )

    for angle, value in zip(
        angles,
        doa_estimate
    ):

        idx = int(
            (angle + 180) // 3
        )

        if 0 <= idx < len(
            full_array_doa
        ):

            full_array_doa[idx] = value


    # ---------------------------------------------------------------------------------------------
    # THRESHOLD
    # ---------------------------------------------------------------------------------------------

    if db_spl <= trigger_level:

        logger.info(
            "DOA #%d rejected: %.2f dB SPL <= threshold %.2f dB SPL",
            counter,
            db_spl,
            trigger_level
        )

        robot_move.time_step += 1

        row_from_doa = doa_to_row(
            doa_estimate=full_array_doa
        )

        return counter


    # ---------------------------------------------------------------------------------------------
    # VALID DOA ABOVE THRESHOLD
    # ---------------------------------------------------------------------------------------------

    logger.info(
        "DOA #%d accepted: %.2f dB SPL > threshold %.2f dB SPL",
        counter,
        db_spl,
        trigger_level
    )
    
    if robot_move.obstacle_in_front():

        logger.warning(
            "Obstacle detected before ring-attractor processing. "
            "Discarding current DOA."
        )

        robot_move.stop()

        return counter


    # ---------------------------------------------------------------------------------------------
    # RUN RING ATTRACTOR
    # ---------------------------------------------------------------------------------------------

    bump_angle, row_from_doa, ring_processing_time = (
        run_ring_attractor(
            ring=ring,
            doa_estimate=full_array_doa
        )
    )

    robot_move.heading = bump_angle

    robot_move.time_step += 1


    # ---------------------------------------------------------------------------------------------
    # SAVE RING STATE
    # ---------------------------------------------------------------------------------------------

    if isinstance(
        args.angle,
        (list, tuple, np.ndarray)
    ):

        detected_angles = list(
            args.angle
        )

    else:

        detected_angles = [
            args.angle
        ]


    log_ring_state(
        time_step=robot_move.time_step,
        angles=detected_angles,
        doa_estimate=row_from_doa,
        phi_in_degrees=bump_angle,
        spins=ring.spins,
        state=robot_move.log_ring_state,
        doa_acquisition_time=doa_acquisition_time,
        ring_processing_time=ring_processing_time,
        session_name="target_tracking"
    )


    # ---------------------------------------------------------------------------------------------
    # OUTPUT CSV
    # ---------------------------------------------------------------------------------------------

    row = (
        list(ring.spins)
        +
        [bump_angle]
        +
        detected_angles
    )

    with open(
        "output.csv",
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(row)


    logger.info(
        "Cycle #%d: DOA=%s | dB_SPL=%.2f | bump_angle=%.2f",
        counter,
        detected_angles,
        db_spl,
        bump_angle
    )


    # ---------------------------------------------------------------------------------------------
    # TURN
    # ---------------------------------------------------------------------------------------------

    if robot_move.check_stop_all_motion():

        logger.warning(
            "Safety condition detected before turn"
        )

        robot_move.stop()

        return counter


    if bump_angle < 0:

        logger.info(
            "Bump angle negative: rotating RIGHT by %.2f degrees",
            abs(bump_angle)
        )

        robot_move.smooth_rotate_right(
            abs(bump_angle)
        )

    elif bump_angle > 0:

        logger.info(
            "Bump angle positive: rotating LEFT by %.2f degrees",
            bump_angle
        )

        robot_move.smooth_rotate_left(
            bump_angle
        )

    else:

        logger.info(
            "Bump angle is zero: no rotation required"
        )


    robot_move.heading = 0


    # ---------------------------------------------------------------------------------------------
    # MOVE FOR EXACTLY 0.5 SECONDS
    # ---------------------------------------------------------------------------------------------

    logger.info(
        "Starting forward movement for 0.2 seconds"
    )

    if robot_move.move_forward():

        movement_start = time.time()

        while (
            time.time() -
            movement_start
            <
            0.5
        ):

            if robot_move.check_stop_all_motion():

                logger.warning(
                    "Safety stop during 0.5-second movement"
                )

                break

            # Keep this loop responsive to safety sensors.
            time.sleep(0.005)

    else:

        logger.warning(
            "Forward movement was not started"
        )


    # ---------------------------------------------------------------------------------------------
    # STOP
    # ---------------------------------------------------------------------------------------------

    robot_move.stop()

    logger.info(
        "Forward movement complete. Robot STOPPED."
    )


    # ---------------------------------------------------------------------------------------------
    # WAIT 8 SECONDS BEFORE NEXT DOA
    # ---------------------------------------------------------------------------------------------

    logger.info(
        "Waiting 5 seconds before requesting next DOA"
    )

    time.sleep(
        5
    )

    logger.info(
        "8-second wait complete"
    )


    return counter


###################################################################################################
# COMMAND LINE
###################################################################################################

def int_or_str(text):

    try:

        return int(text)

    except ValueError:

        return text


###################################################################################################
# MAIN
###################################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Single-sequence ROBAT Thymio control"
        )
    )

    parser.add_argument(
        "-l",
        "--list-devices",
        action="store_true",
        help="Show audio devices and exit"
    )

    parser.add_argument(
        "filename",
        nargs="?",
        metavar="FILENAME"
    )

    parser.add_argument(
        "-d",
        "--device",
        type=int_or_str
    )

    parser.add_argument(
        "-r",
        "--samplerate",
        type=int
    )

    parser.add_argument(
        "-c",
        "--channels",
        type=int,
        default=channels
    )

    parser.add_argument(
        "-t",
        "--subtype",
        type=str
    )

    args = parser.parse_args()


    ################################################################################################
    # AUDIO DEVICE
    ################################################################################################

    if args.list_devices:

        print(
            sd.query_devices()
        )

        raise SystemExit(0)


    logger.info(
        "Audio recording enabled: %s",
        recording_bool
    )

    logger.info(
        "Timestamp saving enabled: %s",
        timestamp_bool
    )


    ################################################################################################
    # GET AUDIO CARD
    ################################################################################################

    usb_fireface_index = get_card(
        sd.query_devices()
    )

    logger.info(
        "USB Fireface index: %s",
        usb_fireface_index
    )


    ################################################################################################
    # NETWORK
    ################################################################################################

    ni.ifaddresses(
        "wlan0"
    )

    raspi_local_ip = (
        ni.ifaddresses("wlan0")[2][0]["addr"]
    )

    logger.info(
        "Raspberry Pi local IP: %s",
        raspi_local_ip
    )


    ################################################################################################
    # AUDIO ARGUMENTS
    ################################################################################################

    args.buffer = np.zeros(
        (
            block_size,
            channels
        )
    )

    startime = datetime.datetime.now()

    args.samplerate = fs

    args.rec_samplerate = rec_samplerate

    args.angle = 0


    ################################################################################################
    # DATA DIRECTORIES
    ################################################################################################

    time1 = startime.strftime(
        "%Y-%m-%d"
    )

    save_path = (
        "/home/thymio/robat_py/"
        "robat_swarm/passive_robat/"
    )

    folder_name = str(
        raspi_local_ip
    )

    folder_path = os.path.join(
        save_path,
        "Data",
        folder_name,
        time1
    )


    ################################################################################################
    # RECORDING FILE
    ################################################################################################

    if recording_bool:

        os.makedirs(
            folder_path,
            exist_ok=True
        )

        time2 = startime.strftime(
            "_%Y-%m-%d__%H-%M-%S"
        )

        name = (
            "MULTIWAV_"
            +
            str(raspi_local_ip)
            +
            str(time2)
            +
            ".wav"
        )

        args.filename = os.path.join(
            folder_path,
            name
        )

        logger.info(
            "Recording file: %s",
            args.filename
        )

        if args.samplerate is None:

            device_info = sd.query_devices(
                args.device,
                "input"
            )

            args.samplerate = int(
                device_info[
                    "default_samplerate"
                ]
            )


    ################################################################################################
    # TIMESTAMP FILE
    ################################################################################################

    if timestamp_bool:

        os.makedirs(
            folder_path,
            exist_ok=True
        )

        folder_path_data = os.path.join(
            save_path,
            "Data",
            folder_name,
            time1
        )

        os.makedirs(
            folder_path_data,
            exist_ok=True
        )

        time2 = startime.strftime(
            "_%Y-%m-%d__%H-%M-%S"
        )

        timestamp_filename = (
            "TIMESTAMPS_"
            +
            str(raspi_local_ip)
            +
            str(time2)
            +
            ".csv"
        )

        args.timestamp_filename = os.path.join(
            folder_path_data,
            timestamp_filename
        )

        logger.info(
            "Timestamp file: %s",
            args.timestamp_filename
        )


    ################################################################################################
    # INITIALIZE RING
    ################################################################################################

    ring = RA()


    ################################################################################################
    # INITIALIZE AUDIO PROCESSOR
    ################################################################################################

    logger.info(
        "Initializing AudioProcessor"
    )

    audio_processor = AudioProcessor(
        fs,
        channels,
        block_size,
        analyzed_buffer_time,
        data,
        args,
        trigger_level,
        critical_level,
        mic_spacing,
        ref,
        highpass_freq,
        lowpass_freq,
        theta_das,
        N_peaks,
        usb_fireface_index,
        args.subtype,
        interp_sensitivity,
        tgtmic_relevant_freqs,
        args.filename,
        args.rec_samplerate,
        sos,
        sweep
    )

    logger.info(
        "AudioProcessor initialized"
    )


    ################################################################################################
    # INITIALIZE ROBOT
    ################################################################################################

    robot_move = RobotMove(
        forward_speed=speed,
        turn_speed=turn_speed,
        left_sensor_threshold=left_sensor_threshold,
        right_sensor_threshold=right_sensor_threshold,
        critical_level=critical_level,
        trigger_level=trigger_level,
        ring=ring,
        ground_sensors_bool=False
    )


    ################################################################################################
    # START AUDIO ACQUISITION
    #
    # IMPORTANT:
    #
    # This is the only continuously running acquisition operation.
    #
    # The old code also had a separate robot movement thread:
    #
    #     move_thread = threading.Thread(
    #         target=robot_move.move_for_target
    #     )
    #
    # That thread has been completely removed.
    #
    # The main loop below owns:
    #
    #     DOA -> threshold -> ring -> turn -> move -> stop -> wait
    #
    ################################################################################################

    inputstream_thread = None

    if recording_bool:

        logger.info(
            "Starting audio recording acquisition"
        )

        inputstream_thread = threading.Thread(
            target=audio_processor.continuos_recording,
            daemon=True
        )

        inputstream_thread.start()

    else:

        logger.info(
            "Starting audio input acquisition"
        )

        inputstream_thread = threading.Thread(
            target=audio_processor.input_stream,
            daemon=True
        )

        inputstream_thread.start()


    ################################################################################################
    # TIMESTAMP STORAGE
    ################################################################################################

    time2 = startime.strftime(
        "_%Y-%m-%d__%H-%M-%S"
    )

    os.makedirs(
        folder_path,
        exist_ok=True
    )

    name = (
        "TIMESTAMPS_"
        +
        str(raspi_local_ip)
        +
        str(time2)
    )

    npy_data = os.path.join(
        folder_path,
        name
    )


    ################################################################################################
    # START TIMESTAMP
    ################################################################################################

    try:

        initial_audio_timestamp = (
            audio_processor.ts_queue.get()
        )

    except Exception:

        initial_audio_timestamp = (
            datetime.datetime.timestamp(
                datetime.datetime.now(
                    datetime.timezone.utc
                )
            )
        )


    timestamp_queue.put(
        [
            initial_audio_timestamp,
            0,
            0,
            "recording_start"
        ]
    )


    ################################################################################################
    # MAIN SINGLE-SEQUENCE PROCESSING LOOP
    ################################################################################################

    counter = 0

    logger.info(
        "============================================================"
    )

    logger.info(
        "ENTERING SINGLE-SEQUENCE CONTROL LOOP"
    )

    logger.info(
        "Threshold: %.2f dB SPL",
        trigger_level
    )

    logger.info(
        "Movement duration: 0.50 s"
    )

    logger.info(
        "Stop wait duration: 0.20 s"
    )

    logger.info(
        "============================================================"
    )


    try:

        while True:

            cycle_start = time.time()

            logger.info(
                "BEGIN CYCLE"
            )


            ####################################################################
            # PROCESS EXACTLY ONE DOA
            ####################################################################

            counter = process_one_doa_cycle(
                audio_processor=audio_processor,
                robot_move=robot_move,
                ring=ring,
                args=args,
                counter=counter
            )


            cycle_duration = (
                time.time() -
                cycle_start
            )

            logger.info(
                "END CYCLE #%d | total cycle processing time = %.3f s",
                counter,
                cycle_duration
            )


            ####################################################################
            # IMPORTANT:
            #
            # There is deliberately NO:
            #
            #     angle_queue.get()
            #
            #     while not angle_queue.empty()
            #
            #     robot processing thread
            #
            #     second movement loop
            #
            # The next call to process_one_doa_cycle() requests the next
            # DOA only after the complete previous movement cycle has finished.
            ####################################################################


    except KeyboardInterrupt:

        logger.info(
            "KeyboardInterrupt received"
        )

        robot_move.running = False

        robot_move.stop()


    except Exception as e:

        logger.exception(
            "Fatal error in main control loop: %s",
            e
        )

        robot_move.running = False

        robot_move.stop()

        raise


    finally:

        ####################################################################
        # ALWAYS STOP ROBOT
        ####################################################################

        logger.info(
            "Executing final robot stop"
        )

        robot_move.running = False

        robot_move.stop()


        ####################################################################
        # SAVE REMAINING TIMESTAMPS
        ####################################################################

        remaining_data = []

        while not timestamp_queue.empty():

            remaining_data.append(
                timestamp_queue.get()
            )


        if remaining_data:

            npy_file = (
                npy_data +
                ".npy"
            )

            if os.path.exists(
                npy_file
            ):

                existing_data = np.load(
                    npy_file,
                    allow_pickle=True
                )

                if (
                    existing_data.ndim == 1
                    and
                    len(existing_data) > 0
                    and
                    isinstance(
                        existing_data[0],
                        (list, np.ndarray)
                    )
                ):

                    combined_data = np.concatenate(
                        [
                            existing_data,
                            remaining_data
                        ],
                        axis=0
                    )

                else:

                    combined_data = np.array(
                        [existing_data]
                        +
                        remaining_data,
                        dtype=object
                    )

            else:

                combined_data = np.array(
                    remaining_data,
                    dtype=object
                )

            np.save(
                npy_data,
                combined_data
            )

            logger.info(
                "Timestamp matrix saved: %s.npy",
                npy_data
            )


        ####################################################################
        # JOIN AUDIO THREAD
        ####################################################################

        if inputstream_thread is not None:

            logger.info(
                "Audio acquisition thread is daemonized; "
                "main processing has terminated."
            )


        logger.info(
            "============================================================"
        )

        logger.info(
            "ROBAT CONTROL SYSTEM TERMINATED"
        )

        logger.info(
            "Robot is stopped."
        )

        logger.info(
            "============================================================"
        )