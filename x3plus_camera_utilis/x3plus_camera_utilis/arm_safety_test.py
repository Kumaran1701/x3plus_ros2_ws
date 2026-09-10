#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from x3plus_msgs.msg import ArmJoint
from x3plus_msgs.srv import RobotArmArray

def drake_rad_to_servo_deg(q_rad_5):
    return np.degrees(q_rad_5) + 90.0


class ArmSafetyTest(Node):
    def __init__(self):
        super().__init__('arm_safety_test')

        # Parameters
        self.declare_parameter("mode", "static")
        self.declare_parameter("test_joint", 1)
        self.declare_parameter("traj_file", "")

        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        self.test_joint = self.get_parameter("test_joint").get_parameter_value().integer_value
        self.traj_file = self.get_parameter("traj_file").get_parameter_value().string_value

        self.pub = self.create_publisher(ArmJoint, 'TargetAngle', 10)

        # Query current angles
        self.grip_client = self.create_client(RobotArmArray, 'CurrentAngle')
        self.get_logger().info("Waiting for CurrentAngle service...")
        self.grip_client.wait_for_service()
        future = self.grip_client.call_async(RobotArmArray.Request())
        rclpy.spin_until_future_complete(self, future)
        self.current_angles = list(future.result().angles)

        self.get_logger().info(f"Current angles: {self.current_angles}")

        # Run test ONCE
        self.create_timer(0.01, self.run_once)
        self.test_started = False

    def run_once(self):
        if self.test_started:
            return
        self.test_started = True

        if self.mode == "static":
            self.static_test()

        elif self.mode == "micro":
            self.micro_motion_test()

        elif self.mode == "single":
            self.single_waypoint_test()

        elif self.mode == "short":
            self.short_trajectory_test()

        else:
            self.get_logger().error(f"Unknown mode: {self.mode}")
            self.shutdown_safe()

    # ============================================================
    # Test 1: Static sanity check
    # ============================================================
    def static_test(self):
        self.get_logger().info("STATIC TEST — no motion")
        self.get_logger().info("Verify angles manually.")
        self.shutdown_safe()

    # ============================================================
    # Test 2: Micro‑motion (+2° and back)
    # ============================================================
    def micro_motion_test(self):
        j = self.test_joint - 1
        if j < 0 or j > 4:
            self.get_logger().error("test_joint must be 1–5")
            self.shutdown_safe()
            return

        self.get_logger().info(f"MICRO MOTION TEST — joint {self.test_joint}")

        msg = ArmJoint()
        msg.joints = self.current_angles.copy()
        msg.joints[j] += 2.0
        msg.run_time = 300

        self.pub.publish(msg)
        self.get_logger().info(f"Moving joint {self.test_joint} +2°")

        def move_back():
            msg2 = ArmJoint()
            msg2.joints = self.current_angles.copy()
            msg2.run_time = 300
            self.get_logger().info(f"Moving joint {self.test_joint} back")
            self.pub.publish(msg2)
            self.shutdown_safe()

        self.create_timer(0.5, move_back)

    # ============================================================
    # Test 3: Single waypoint test
    # ============================================================
    def single_waypoint_test(self):
        if self.traj_file == "":
            self.get_logger().error("traj_file required")
            self.shutdown_safe()
            return

        data = np.load(self.traj_file)
        q0 = data["q"][0]
        servo_deg = drake_rad_to_servo_deg(q0)

        self.get_logger().info("SINGLE WAYPOINT TEST — sending first waypoint")
        self.get_logger().info(f"servo_deg={servo_deg}")

        msg = ArmJoint()
        msg.joints = list(servo_deg) + [self.current_angles[5]]
        msg.run_time = 400

        self.pub.publish(msg)
        self.get_logger().info("Sent first waypoint.")

        self.create_timer(1.0, self.shutdown_safe)

    # ============================================================
    # Test 4: Short trajectory test (SAFE VERSION)
    # ============================================================
    def short_trajectory_test(self):
        if self.traj_file == "":
            self.get_logger().error("traj_file required")
            self.shutdown_safe()
            return

        data = np.load(self.traj_file)
        self.times = data["times"]
        self.q = data["q"]
        self.index = 0

        # SAFE timing
        #self.dt = max(self.times[1] - self.times[0], 0.05)  # at least 50 ms
        #self.run_time_ms = int(self.dt * 1000) - 10         # finish before next command
        self.dt = max(self.times[1] - self.times[0], 0.05)  # at least 50 ms
        self.run_time_ms = int(self.dt * 1000) + 20

        self.get_logger().info(f"SHORT TRAJECTORY TEST — {len(self.times)} waypoints")
        self.get_logger().info(f"dt={self.dt:.3f}s, run_time={self.run_time_ms}ms")

        self.timer = self.create_timer(self.dt, self.publish_next)

    def publish_next(self):
        if self.index >= len(self.times):
            self.get_logger().info("Short trajectory complete.")
            self.timer.cancel()
            self.shutdown_safe()
            return

        q_rad = self.q[self.index]
        servo_deg = drake_rad_to_servo_deg(q_rad)

        msg = ArmJoint()
        msg.joints = list(servo_deg) + [self.current_angles[5]]
        msg.run_time = self.run_time_ms

        self.pub.publish(msg)
        self.get_logger().info(
            f"Waypoint {self.index}/{len(self.times)}  "
            f"t={self.times[self.index]:.2f}s  "
            f"servo_deg={np.round(servo_deg, 1)}"
        )

        self.index += 1

    # ============================================================
    # Safe shutdown
    # ============================================================
    def shutdown_safe(self):
        self.get_logger().info("Shutting down safely...")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = ArmSafetyTest()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
