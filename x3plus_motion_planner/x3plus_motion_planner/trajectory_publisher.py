#!/usr/bin/env python3
import csv
import os
import time

import numpy as np
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node

from x3plus_msgs.msg import ArmJoint
from x3plus_msgs.srv import RobotArmArray


def drake_rad_to_servo_deg(q_rad):
    return np.degrees(q_rad) + 90.0


SERVO_MAX_ANGLES_DEG = (180, 180, 180, 180, 270, 180)


def valid_servo_angles(angles, joint_count):
    return len(angles) >= joint_count and all(
        np.isfinite(angles[joint]) and 0 <= angles[joint] <= SERVO_MAX_ANGLES_DEG[joint]
        for joint in range(joint_count)
    )


# ------------------------------- Arm Trajectory Controller Node -----------------------------------

class ArmTrajectoryController(Node):

    def __init__(self):
        """
        Pubishes time-sampled Drake GCS trajectory to the robot's arm controller at a fixed rate.
        """
        
        super().__init__('arm_trajectory_controller')

        pkg_path_motion_planner = get_package_share_directory("x3plus_motion_planner")
        traj_file = os.path.join(pkg_path_motion_planner, "trajectory", "trajectory_export.npz")
        # Parameters
        self.declare_parameter("traj_file", str(traj_file))
        self.declare_parameter('control_hz', 50.0)
        self.declare_parameter("error_file", os.path.abspath("trajectory_error.csv"))

        # Get parameters
        self.traj_file = (self.get_parameter('traj_file').get_parameter_value().string_value)
        self.control_hz = (self.get_parameter('control_hz').get_parameter_value().double_value)
        self.error_file = self.get_parameter("error_file").get_parameter_value().string_value

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
        if not valid_servo_angles(self.current_angles, 6):
            raise RuntimeError(
                "CurrentAngle returned invalid servo angles; cannot safely start trajectory."
            )
        self.get_logger().info(f'Current angles: {self.current_angles}')

        # Load trajectory data
        data = np.load(self.traj_file)

        self.times = np.asarray(data['times'], dtype=float)
        self.q = np.asarray(data['q'], dtype=float)

        if self.q.shape[1] != 5:
            raise ValueError(f'Expected 5 joints, got {self.q.shape[1]}')

        if len(self.times) != len(self.q):
            raise ValueError('times and q have different lengths')
        if len(self.times) == 0:
            raise ValueError('Trajectory must contain at least one sample')
        
        self.duration = self.times[-1]

        self.get_logger().info(f'Trajectory points: {len(self.times)}')
        self.get_logger().info(f'Trajectory duration: {self.duration:.3f} s')

        # Trajectory state variables.
        self.start_time = None
        self.finished = False
        self.pending_feedback_future = None
        self.queued_feedback = None
        self.feedback_samples = []
        self.feedback_failures = 0
        self.feedback_saved = False
        self.shutdown_timer = None

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
            self.request_feedback(self.duration, self.q[-1])

            self.finished = True
            self.timer.cancel()

            self.get_logger().info('Trajectory complete.')

            # Allow final command to be transmitted
            self.shutdown_timer = self.create_timer(0.1, self.finish_shutdown)

            return
  
        # Continuous q(t) 
        q_des = self.evaluate_trajectory(t)

        # Send position
        self.send_position(q_des)
        self.request_feedback(t, q_des)


    # --------------------------------- Send direct servo position -------------------------------

    def send_position(self, q_rad):

        servo_deg = drake_rad_to_servo_deg(q_rad)
        msg = ArmJoint()

        # 5 arm joints + current gripper position
        msg.joints = (list(servo_deg) + [self.current_angles[5]])

        # Driver ignores this for TrajectoryAngle
        msg.run_time = 0

        self.pub.publish(msg)

    # ----------------------- Compare commanded and measured trajectory positions -------------------

    def request_feedback(self, trajectory_time, desired_q):
        if self.pending_feedback_future is not None:
            self.queued_feedback = (trajectory_time, desired_q.copy())
            return

        future = self.current_client.call_async(RobotArmArray.Request())
        self.pending_feedback_future = future
        future.add_done_callback(
            lambda completed: self.record_feedback(completed, trajectory_time, desired_q)
        )

    def record_feedback(self, future, trajectory_time, desired_q):
        try:
            response = future.result()
        except Exception as exc:
            self.feedback_failures += 1
            self.get_logger().error(f"CurrentAngle feedback request failed: {exc}")
        else:
            measured_deg = np.asarray(response.angles, dtype=float)
            if not valid_servo_angles(measured_deg, 5):
                self.feedback_failures += 1
                self.get_logger().error(
                    "CurrentAngle returned invalid arm positions; expected at least 5 "
                    "finite angles within servo limits"
                )
            else:
                desired_deg = drake_rad_to_servo_deg(desired_q)
                errors_deg = measured_deg[:5] - desired_deg
                self.feedback_samples.append(
                    (trajectory_time, desired_q, np.radians(measured_deg[:5] - 90.0), errors_deg)
                )
        finally:
            self.pending_feedback_future = None
            if self.queued_feedback is not None:
                queued_feedback = self.queued_feedback
                self.queued_feedback = None
                self.request_feedback(*queued_feedback)

    def save_feedback_report(self):
        if self.feedback_saved:
            return
        self.feedback_saved = True

        fieldnames = ["trajectory_time_s"]
        for joint in range(1, 6):
            fieldnames.extend(
                (
                    f"joint{joint}_desired_rad",
                    f"joint{joint}_measured_rad",
                    f"joint{joint}_error_deg",
                )
            )

        output_dir = os.path.dirname(os.path.abspath(self.error_file))
        os.makedirs(output_dir, exist_ok=True)
        with open(self.error_file, "w", newline="") as report:
            writer = csv.DictWriter(report, fieldnames=fieldnames)
            writer.writeheader()
            for trajectory_time, desired_q, measured_q, errors_deg in self.feedback_samples:
                row = {"trajectory_time_s": trajectory_time}
                for joint in range(5):
                    row[f"joint{joint + 1}_desired_rad"] = desired_q[joint]
                    row[f"joint{joint + 1}_measured_rad"] = measured_q[joint]
                    row[f"joint{joint + 1}_error_deg"] = errors_deg[joint]
                writer.writerow(row)

        self.get_logger().info(
            f"Saved {len(self.feedback_samples)} trajectory feedback comparisons to "
            f"{self.error_file} ({self.feedback_failures} failed readings)."
        )
        if not self.feedback_samples:
            self.get_logger().error("No valid CurrentAngle readings were collected.")
            return

        errors = np.asarray([sample[3] for sample in self.feedback_samples])
        for joint in range(5):
            joint_errors = errors[:, joint]
            mae = np.mean(np.abs(joint_errors))
            rmse = np.sqrt(np.mean(np.square(joint_errors)))
            max_error = np.max(np.abs(joint_errors))
            self.get_logger().info(
                f"Joint {joint + 1} tracking error (deg): "
                f"MAE={mae:.2f}, RMSE={rmse:.2f}, max_abs={max_error:.2f}"
            )

    # ------------------------------------- Shutdown safely --------------------------------------

    def finish_shutdown(self):
        if self.pending_feedback_future is not None:
            return
        self.shutdown_timer.cancel()
        self.save_feedback_report()
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