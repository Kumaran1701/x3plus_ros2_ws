#!/usr/bin/env python3

import time
import numpy as np

import rclpy
from rclpy.node import Node

from x3plus_msgs.msg import ArmJoint
from x3plus_msgs.srv import RobotArmArray


def drake_rad_to_servo_deg(q_rad):
    return np.degrees(q_rad) + 90.0


# ------------------------------- Arm Trajectory Controller Node -----------------------------------

class ArmTrajectoryController(Node):

    def __init__(self):
        """
        Pubishes time-sampled Drake GCS trajectory to the robot's arm controller at a fixed rate.
        """
        
        super().__init__('arm_trajectory_controller')

        # Parameters
        self.declare_parameter('traj_file', 'trajectory_export_minjerk.npz')
        self.declare_parameter('control_hz', 50.0)

        # Get parameters
        self.traj_file = (self.get_parameter('traj_file').get_parameter_value().string_value)
        self.control_hz = (self.get_parameter('control_hz').get_parameter_value().double_value)

        # Publisher for trajectory commands
        self.pub = self.create_publisher(ArmJoint, 'TrajectoryAngle', 10)

        # Service client to get current angles
        self.current_client = self.create_client(RobotArmArray, 'CurrentAngle')

        self.get_logger().info('Waiting for CurrentAngle service')
        self.current_client.wait_for_service()

        # Request current angles
        future = self.current_client.call_async(RobotArmArray.Request())
        rclpy.spin_until_future_complete(self, future)

        self.current_angles = list(future.result().angles)
        self.get_logger().info(f'Current angles: {self.current_angles}')

        # Load trajectory data
        data = np.load(self.traj_file)

        self.times = np.asarray(data['times'], dtype=float)
        self.q = np.asarray(data['q'], dtype=float)

        if self.q.shape[1] != 5:
            raise ValueError(f'Expected 5 joints, got {self.q.shape[1]}')

        if len(self.times) != len(self.q):
            raise ValueError('times and q have different lengths')
        
        self.duration = self.times[-1]

        self.get_logger().info(f'Trajectory points: {len(self.times)}')
        self.get_logger().info(f'Trajectory duration: {self.duration:.3f} s')

        # Trajectory state variables.
        self.start_time = None
        self.finished = False

        # Fixed-rate controller
        self.timer = self.create_timer(1.0 / self.control_hz, self.control_loop)
        self.get_logger().info(f'Trajectory controller running at {self.control_hz:.1f} Hz')


    # -------------------- Evaluate trajectory at time t (linear interpolation) --------------------
    
    def evaluate_trajectory(self, t):

        t = np.clip(t, self.times[0], self.times[-1])

        q_des = np.empty(5)

        for j in range(5):
            q_des[j] = np.interp(t, self.times, self.q[:, j])

        return q_des


    # ----------------------------- Control loop called at fixed rate ------------------------------
    
    def control_loop(self):

        if self.finished:
            return

        if self.start_time is None:
            self.start_time = time.monotonic()

            self.get_logger().info('Starting trajectory execution.')

        # Real elapsed trajectory time
        t = time.monotonic() - self.start_time
        
        # End of trajectory
        if t >= self.duration:

            self.send_position(self.q[-1])

            self.finished = True
            self.timer.cancel()

            self.get_logger().info('Trajectory complete.')

            # Allow final command to be transmitted
            self.create_timer(0.1, self.finish_shutdown)

            return
  
        # Continuous q(t) 
        q_des = self.evaluate_trajectory(t)

        # Send position
        self.send_position(q_des)


    # --------------------------------- Send direct servo position -------------------------------

    def send_position(self, q_rad):

        servo_deg = drake_rad_to_servo_deg(q_rad)
        msg = ArmJoint()

        # 5 arm joints + current gripper position
        msg.joints = (list(servo_deg) + [self.current_angles[5]])

        # Driver ignores this for TrajectoryAngle
        msg.run_time = 0

        self.pub.publish(msg)


    # ------------------------------------- Shutdown safely --------------------------------------

    def finish_shutdown(self):
        self.get_logger().info('Shutting down.')
        rclpy.shutdown()


# ---------------------------------------------- MAIN --------------------------------------------

def main():

    rclpy.init()
    node = ArmTrajectoryController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()