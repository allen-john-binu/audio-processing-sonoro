from thymiodirect import Connection
from thymiodirect import Thymio
from shared_queues import angle_queue, level_queue    
from collections import Counter, deque

import time
import random
from random import seed as py_seed
import math
import utils
import numpy as np
import csv
import datetime
import os

seed = 300
np.random.seed(seed)
py_seed(seed)

import math

  # ── NEW METHOD ───────────────────────────────────────────────
def arc_move(self, phi_degrees, duration=0.5, k=0.8):
    """
    Move in a smooth arc for `duration` seconds.
    phi_degrees : heading from ring attractor (relative, already in [-180, 180])
    k           : curvature gain in [0, 1]. 0 = always straight, 1 = max curve.
    Both wheels stay >= 0 (no reverse).
    """
    phi_norm = max(-1.0, min(1.0, phi_degrees / 180.0))   # clip to [-1, 1]
    left_speed  = int(self.forward_speed * (1.0 - k * phi_norm))
    right_speed = int(self.forward_speed * (1.0 + k * phi_norm))
    # Clamp: never reverse, never exceed motor max
    left_speed  = max(0, min(500, left_speed))
    right_speed = max(0, min(500, right_speed))
    self.robot["leds.top"]          = [0, 255, 0]
    self.robot["leds.bottom.right"] = [0, 255, 0]
    self.robot["leds.bottom.left"]  = [0, 255, 0]
    start = time.time()
    while time.time() - start < duration:
        if self.check_stop_all_motion():
            self.stop_bool = True
            self.stop()
            return
        self.robot['motor.left.target']  = left_speed
        self.robot['motor.right.target'] = right_speed
  # ─────────────────────────────────────────────────────────────

def doa_to_row(
    doa_estimate,
    global_min=0.0000000005,
    global_max=0.0000068100,
    NORMALIZING_FACTOR=0.1546,
):
    """
    Replicates the normalization performed in the ring attractor script.

    Parameters
    ----------
    doa_estimate : sequence of length 120
        The 120-element full_array.
    global_min : float
        Dataset-wide minimum DOA value (min(all_values)).
    global_max : float
        Dataset-wide maximum DOA value (max(all_values)).
    NORMALIZING_FACTOR : float
        Scaling factor applied after normalization.

    Returns
    -------
    list[float]
        Equivalent to row[1:] in the original script.
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
                norm = (v - global_min) / denom

            norm_arr.append(norm * NORMALIZING_FACTOR)

    return norm_arr

def log_ring_state(
    time_step,
    angles,
    doa_estimate,
    phi_in_degrees,
    state,
    base_folder="ring_results",
    session_name="session",
    max_angles=2
):
    """
    Logs:
        timestamp
        time_step
        phi_in_degrees
        angles (variable length, padded to max_angles)
        doa_estimate (fixed length)

    Notes:
        - angles may contain 1 or 2 detected DOAs.
        - To keep a fixed CSV structure, missing angles are filled with -1000.
        - Example:
              angles = [35]
          becomes:
              angle_0 = 35
              angle_1 = -1000
        - doa_estimate is assumed to have a fixed length across all calls.

    state: dict used to persist file_path between calls
    """

    # ---------- INIT ----------
    if "file_path" not in state:
        os.makedirs(base_folder, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{session_name}_{timestamp}.csv"
        file_path = os.path.join(base_folder, file_name)

        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)

            header = [
                "timestamp",
                "time_step",
                "phi_in_degrees"
            ]

            # Fixed angle columns.
            # Missing angles will be stored as -1000.
            header += [f"angle_{i}" for i in range(max_angles)]

            # One column per DOA estimate value.
            header += [f"doa_{i}" for i in range(len(doa_estimate))]

            writer.writerow(header)

        state["file_path"] = file_path

        print(f"[INIT] Ring-state logging started: {file_path}")

    # ---------- APPEND ----------

    # Keep a fixed number of angle columns.
    # If fewer angles are available, pad with -1000.
    padded_angles = list(angles[:max_angles])

    if len(padded_angles) < max_angles:
        padded_angles.extend([-1000] * (max_angles - len(padded_angles)))

    row = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        time_step,
        phi_in_degrees,
    ]

    row += padded_angles
    row += list(doa_estimate)

    with open(state["file_path"], mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)



class RobotMove():
    def __init__(self, forward_speed, turn_speed, left_sensor_threshold, right_sensor_threshold, critical_level, trigger_level,ring, ground_sensors_bool = True):
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed    
        self.left_sensor_threshold = left_sensor_threshold
        self.right_sensor_threshold = right_sensor_threshold
        self.ground_sensors_bool = ground_sensors_bool
        self.critical_level = critical_level
        self.trigger_level = trigger_level
        self.running = True
        self.ring = ring
        self.heading = 0
        self.max_count = 0
        self.time_step = 0
        self.log_ring_state = {}

        print("Initializing Thymio Robot")
        port = Connection.serial_default_port()
        th = Thymio(serial_port=port,
                    on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
        # Connect to Robot
        th.connect()
        self.robot = th[th.first_node()]

        print("Robot connected")

        if self.ground_sensors_bool:
            print('\nGround sensors values...\n')
            time.sleep(2)  # wait for sensors to initialize
            print('ground.delta  L R = ', self.robot['prox.ground.delta'])
            print('ground.reflected  L R = ', self.robot['prox.ground.reflected'])
            print('\n')

        self.stop_bool = False

    def angle_to_time(self, angle, speed):
        #calculate the time needed to turn the robot by a certain angle
        A = 612.33
        B = -0.94
        t = A*speed**B    
        return t * abs(angle) / 360 # time to turn by angle in seconds

    def move_for_target(self):
        while self.running:
            try:
                # Check for a global stop signal (e.g., if the robot is lifted)
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    self.stop()
                    continue

                while not angle_queue.empty():
                    angles, doa_estimate, counter, db_spl = angle_queue.get()

                    if counter >= self.max_count:
                        self.max_count = counter
                        if angles is not None:
                            if db_spl > 83 :
                                self.ring.h_ext = None
                                
                                row_from_doa = doa_to_row(
                                    doa_estimate=doa_estimate
                                )

                                self.ring.h_ext = row_from_doa
                                    
                                for _ in range (self.ring.updates_per_step):
                                    i = np.random.randint(0,self.ring.Ns)
                                    delta_H = utils.compute_delta_H(self.ring,i)

                                    if delta_H < 0:
                                        # Energy difference negative — always accept
                                        self.ring.spins[i] = 1 - self.ring.spins[i]
                                    else:
                                        # Energy difference positive — accept with probability exp(-beta * delta_H)
                                        p = np.exp(-self.ring.beta * delta_H)
                                        if np.random.rand() < p:
                                            self.ring.spins[i] = 1 - self.ring.spins[i]
                                            
                                active_indices = np.where(self.ring.spins == 1)[0]
                                n_active = len(active_indices)
                                if n_active > 0:
                                    phi = np.angle(np.sum(np.exp(1j * self.ring.thetas[self.ring.spins == 1])))
                                else:
                                    phi = 0.0 
                                # ring attractor processing end 
                                phi_in_degrees = math.degrees(phi)
                                self.heading = phi_in_degrees
                                self.time_step +=1
                                row =  list(self.ring.spins)+ [phi_in_degrees] + list(angles)
                                
                                log_ring_state(
                                    time_step=self.time_step,
                                    angles=angles,
                                    doa_estimate=row_from_doa,
                                    phi_in_degrees=phi_in_degrees,
                                    state=self.log_ring_state,
                                    session_name="target_tracking"
                                )
                                with open("output.csv", "a", newline="") as file:
                                    writer = csv.writer(file)
                                    writer.writerow(row)
                                    
                                # if self.heading < 0:
                                #     print('3: Negative angle received, rotating left')
                                #     self.smooth_rotate_left(-(self.heading))
                                #     self.heading = 0
                                # elif self.heading > 0:
                                #     print('4: Positive or zero angle received, rotating right')
                                #     self.smooth_rotate_right(self.heading)
                                #     self.heading = 0

                                # self.move_forward()
                                
                                # time.sleep(0.5) 
                                print(f'Arc move: phi={self.heading:.1f}°')
                                self.arc_move(self.heading, duration=0.5)
                                self.heading = 0

                            else:
                                print("im here without good sound")
                                self.time_step +=1
                                row_from_doa = doa_to_row(
                                        doa_estimate=doa_estimate
                                    )
                                log_ring_state(
                                        time_step=self.time_step,
                                        angles=angles,
                                        doa_estimate=row_from_doa,
                                        phi_in_degrees=-500,
                                        state=self.log_ring_state,
                                        session_name="target_tracking"
                                    )
                                # self.move_forward()
                                # time.sleep(0.5) 
                                self.arc_move(0, duration=0.5) 
                            
            except KeyboardInterrupt:
                self.running = False
                self.stop()
                raise
            finally:
                self.stop_bool = True
                self.stop()
        else:
            self.stop()


    def move_forward(self):
        self.robot["leds.top"] = [0, 255, 0]
        self.robot["leds.bottom.right"] = [0, 255, 0]
        self.robot["leds.bottom.left"] = [0, 255, 0]        
        if self.robot is not None:
            if self.check_stop_all_motion():
                self.stop_bool = True
                self.stop()
                return
            self.robot['motor.left.target'] = self.forward_speed
            self.robot['motor.right.target'] = self.forward_speed

        else:
            print("self.robot is None")
            self.stop_bool = True
            self.stop()

    def rotate_right(self, angle):
        if self.robot is not None:
            counter = self.angle_to_time(angle, self.turn_speed)
            start_time = time.time()
            while time.time() - start_time < counter:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                self.robot['motor.left.target']= self.turn_speed + self.turn_speed//2
                self.robot['motor.right.target']= -self.turn_speed 
        else:
            self.stop()

    def rotate_left(self, angle):
        if self.robot is not None:
            counter = self.angle_to_time(angle, self.turn_speed)
            start_time = time.time()
            while time.time() - start_time < counter:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                self.robot['motor.left.target']= -self.turn_speed 
                self.robot['motor.right.target']= self.turn_speed + self.turn_speed//2

        else:
            self.stop()

    def smooth_rotate_right(self, angle):
        if self.robot is not None:
            counter = self.angle_to_time(angle, self.turn_speed)
            start_time = time.time()
            while time.time() - start_time < counter:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                self.robot['motor.left.target']= self.turn_speed + self.turn_speed//2
                self.robot['motor.right.target']= -self.turn_speed + self.turn_speed//2
        else:
            self.stop()

    def smooth_rotate_left(self, angle):
        if self.robot is not None:
            counter = self.angle_to_time(angle, self.turn_speed)
            start_time = time.time()
            while time.time() - start_time < counter:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                self.robot['motor.left.target']= -self.turn_speed + self.turn_speed//2
                self.robot['motor.right.target']= self.turn_speed + self.turn_speed//2

        else:
            self.stop()

    def stop(self):
        self.robot['motor.left.target'] = 0
        self.robot['motor.right.target'] = 0
        self.robot["leds.top"] = [0, 0, 0]
        self.robot["leds.bottom.right"] = [0, 0, 0]
        self.robot["leds.bottom.left"] = [0, 0, 0]
        self.robot['leds.circle'] = [0, 0, 0, 0, 0, 0, 0 ,0]

