#!/usr/bin/env python3

import time
import numpy as np

import rclpy
from rclpy.node import Node

from x3plus_msgs.msg import ArmJoint
from x3plus_msgs.srv import RobotArmArray


def drake_rad_to_servo_deg(q_rad):
    """
    Drake joint coordinates -> X3Plus servo angles.

    q_rad: 5 arm joints in radians
    """
    return np.degrees(q_rad) + 90.0


class ArmTrajectoryController(Node):

    def __init__(self):
        super().__init__('arm_trajectory_controller')

        # ========================================================
        # Parameters
        # ========================================================

        self.declare_parameter(
            'traj_file',
            'trajectory_export_minjerk.npz'
        )

        self.declare_parameter(
            'control_hz',
            50.0
        )

        self.traj_file = (
            self.get_parameter('traj_file')
            .get_parameter_value()
            .string_value
        )

        self.control_hz = (
            self.get_parameter('control_hz')
            .get_parameter_value()
            .double_value
        )

        # ========================================================
        # Publisher
        # ========================================================

        self.pub = self.create_publisher(
            ArmJoint,
            'TrajectoryAngle',
            10
        )

        # ========================================================
        # Current arm angle
        # ========================================================

        self.current_client = self.create_client(
            RobotArmArray,
            'CurrentAngle'
        )

        self.get_logger().info(
            'Waiting for CurrentAngle service...'
        )

        if not self.current_client.wait_for_service(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                'CurrentAngle service not available.'
            )
            raise RuntimeError(
                'CurrentAngle service unavailable'
            )

        future = self.current_client.call_async(
            RobotArmArray.Request()
        )

        rclpy.spin_until_future_complete(
            self,
            future
        )

        if future.result() is None:
            raise RuntimeError(
                'Failed to read CurrentAngle'
            )

        self.current_angles = list(
            future.result().angles
        )

        self.get_logger().info(
            f'Current servo angles: '
            f'{self.current_angles}'
        )

        # ========================================================
        # Load trajectory
        # ========================================================

        data = np.load(self.traj_file)

        self.times = np.asarray(
            data['times'],
            dtype=float
        )

        self.q = np.asarray(
            data['q'],
            dtype=float
        )

        if self.q.ndim != 2:
            raise ValueError(
                f'q must be 2D, got {self.q.shape}'
            )

        if self.q.shape[1] != 5:
            raise ValueError(
                f'Expected 5 arm joints, '
                f'got {self.q.shape[1]}'
            )

        if len(self.times) != len(self.q):
            raise ValueError(
                'times and q must have same length'
            )

        if len(self.times) < 2:
            raise ValueError(
                'Trajectory must contain at least 2 points'
            )

        self.duration = self.times[-1]

        self.get_logger().info(
            f'Loaded trajectory: '
            f'{len(self.times)} points'
        )

        self.get_logger().info(
            f'Duration: {self.duration:.3f} s'
        )

        self.get_logger().info(
            f'Controller: {self.control_hz:.1f} Hz'
        )

        # ========================================================
        # Controller state
        # ========================================================

        self.start_time = None
        self.finished = False

        self.timer = self.create_timer(
            1.0 / self.control_hz,
            self.control_loop
        )

        self.get_logger().info(
            'Trajectory controller ready.'
        )

    # ============================================================
    # Continuous trajectory evaluation
    # ============================================================

    def evaluate_trajectory(self, t):

        # Clamp to trajectory limits
        t = np.clip(
            t,
            self.times[0],
            self.times[-1]
        )

        q_des = np.empty(5)

        for j in range(5):

            q_des[j] = np.interp(
                t,
                self.times,
                self.q[:, j]
            )

        return q_des

    # ============================================================
    # Fixed-rate controller
    # ============================================================

    def control_loop(self):

        if self.finished:
            return

        if self.start_time is None:

            self.start_time = time.monotonic()

            self.get_logger().info(
                'Starting trajectory.'
            )

        # Absolute elapsed time.
        #
        # IMPORTANT:
        # We do not increment t by 1/control_hz.
        # We measure real elapsed time so that small ROS timer
        # delays do not accumulate.
        t = time.monotonic() - self.start_time

        # ========================================================
        # End of trajectory
        # ========================================================

        if t >= self.duration:

            q_des = self.q[-1]

            self.send_position(q_des)

            self.get_logger().info(
                'Trajectory complete.'
            )

            self.finished = True
            self.timer.cancel()

            # Give the final command a little time to transmit.
            time.sleep(0.1)

            rclpy.shutdown()

            return

        # ========================================================
        # Evaluate q(t)
        # ========================================================

        q_des = self.evaluate_trajectory(t)

        # ========================================================
        # Send instantaneous position
        # ========================================================

        self.send_position(q_des)

    # ============================================================
    # Send arm position
    # ============================================================

    def send_position(self, q_rad):

        servo_deg = drake_rad_to_servo_deg(q_rad)

        # Keep gripper at its current value.
        gripper_deg = self.current_angles[5]

        msg = ArmJoint()

        msg.joints = (
            list(servo_deg)
            + [gripper_deg]
        )

        # Not used by TrajectoryAngle callback,
        # because the driver explicitly sends run_time=0.
        msg.run_time = 0

        self.pub.publish(msg)

    # ============================================================
    # Shutdown
    # ============================================================

    def destroy_node(self):

        self.get_logger().info(
            'Trajectory controller stopped.'
        )

        super().destroy_node()


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