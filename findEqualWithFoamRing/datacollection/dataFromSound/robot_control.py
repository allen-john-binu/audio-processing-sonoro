from thymiodirect import Connection
from thymiodirect import Thymio 
from collections import Counter, deque

import time
import random
from random import seed as py_seed
import numpy as np

seed = 300
np.random.seed(seed)
py_seed(seed)

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
        self.turning_angle = 5  # degrees
        self.ring = ring
        self.heading = 0
        self.max_count = 0
        self.time_step = 0

        print("Initializing Thymio Robot")
        port = Connection.serial_default_port()
        th = Thymio(serial_port=port,
                    on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
        # Connect to Robot
        th.connect()
        self.robot = th[th.first_node()]

        # Delay to allow robot initialization of all variables

        # Print all variables
        # print(th.variables(th.first_node()))
        # time.sleep(1)

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

    def move_forward(self):
        self.robot["leds.top"] = [0, 255, 0]
        self.robot["leds.bottom.right"] = [0, 255, 0]
        self.robot["leds.bottom.left"] = [0, 255, 0]        
        if self.robot is not None:
            self.robot['motor.left.target'] = self.forward_speed
            self.robot['motor.right.target'] = self.forward_speed

        else:
            print("self.robot is None")
            self.stop_bool = True
            self.stop()

    def rotate_right(self, angle):
        if self.robot is not None:
            # print( "rotate right with angle:", angle)
            counter = self.angle_to_time(angle, self.turn_speed)
            start_time = time.time()
            while time.time() - start_time < counter:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                #print("rotate right with angle:", angle)
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
                #print("rotate left with angle:", angle)
                self.robot['motor.left.target']= -self.turn_speed 
                self.robot['motor.right.target']= self.turn_speed + self.turn_speed//2

        else:
            self.stop()

    def smooth_rotate_right(self, angle):
        if self.robot is not None:
            # print( "rotate right with angle:", angle)
            counter = self.angle_to_time(angle, self.turn_speed)
            start_time = time.time()
            while time.time() - start_time < counter:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                #print("rotate right with angle:", angle)
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
                #print("rotate left with angle:", angle)
                self.robot['motor.left.target']= -self.turn_speed + self.turn_speed//2
                self.robot['motor.right.target']= self.turn_speed + self.turn_speed//2

        else:
            self.stop()

    def move_back(self): # move back when robot sees a obstacle in front
        if self.robot is not None:
            counter = 0.2 #sec
            start_time = time.time()
            while time.time() - start_time < counter:
                self.robot['motor.left.target'] = -300
                self.robot['motor.right.target'] = -300
        else:
            self.stop()

    def check_stop_all_motion(self):
        if self.robot is not None:
            # print(f"Prox 0: {self.robot['prox.ground.delta'][0]}, Prox 1: {self.robot['prox.ground.delta'][1]}")
            if self.robot['prox.ground.delta'][0] < 10 or self.robot['prox.ground.delta'][1] < 10:
                # print("Robot lifted")
                return True
        else:
            self.stop()
            return False

    def avoid_white_line(self):
        if self.robot is not None:
            # check if the white line is detected
            if self.robot['prox.ground.reflected'][0] > self.left_sensor_threshold  and self.robot['prox.ground.reflected'][1] > self.right_sensor_threshold:
                # Both sensors detect the line
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    self.stop()
                    return
                self.random_turn(90)

            elif self.robot['prox.ground.reflected'][0] > self.left_sensor_threshold:
                # Left sensor detects the line
                #print('line detected L')
                counter = self.angle_to_time(50, self.turn_speed)
                start_time = time.time()
                while time.time() - start_time < counter:
                    if self.check_stop_all_motion():
                        #print("stop all motion: rotate right")
                        self.stop_bool = True
                        break
                    self.robot['motor.left.target'] = self.turn_speed
                    self.robot['motor.right.target'] = -self.turn_speed

            elif self.robot['prox.ground.reflected'][1] > self.right_sensor_threshold:
                # Right sensor detects the line   
                #print('line detected R')
                counter = self.angle_to_time(50, self.turn_speed)
                start_time = time.time()
                while time.time() - start_time < counter:
                    if self.check_stop_all_motion():
                        self.stop_bool = True
                        break
                    self.robot['motor.left.target'] = -self.turn_speed
                    self.robot['motor.right.target'] = self.turn_speed

        else:
            self.stop()

    def stop(self):
        self.robot['motor.left.target'] = 0
        self.robot['motor.right.target'] = 0
        self.robot["leds.top"] = [0, 0, 0]
        self.robot["leds.bottom.right"] = [0, 0, 0]
        self.robot["leds.bottom.left"] = [0, 0, 0]
        self.robot['leds.circle'] = [0, 0, 0, 0, 0, 0, 0 ,0]


    def random_move(self, angle):
        if self.robot is not None:
            if self.stop_bool:
                print("Stopping Robot")
                self.stop()
                return
            self.move_forward()
            if bool(random.getrandbits(1)):  
                self.rotate_right(angle)
            else:
                self.rotate_left(angle)

    def random_turn(self, angle):
        if self.robot is not None:
            if self.check_stop_all_motion():
                #print("stop all motion: move forward")
                self.stop_bool = True
                self.stop()
                return
            if bool(random.getrandbits(1)):  
                # print('random turn right')
                self.rotate_right(angle)
                return
            else:
                # print('random turn left')
                self.rotate_left(angle)
                return

