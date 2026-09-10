#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from x3plus_msgs.msg import ArmJoint
from x3plus_msgs.srv import RobotArmArray

# ============================================================
# Conversion: Drake radians → servo degrees
# deg = rad * 180/pi + 90
# ============================================================
def drake_rad_to_servo_deg(q_rad_5):
    return np.degrees(q_rad_5) + 90.0


class ArmSafetyTest(Node):
    def __init__(self):
        super().__init__('arm_safety_test')

        # Parameters to choose the test mode
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

        # Run selected test
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

    # ============================================================
    # Test 1: Static sanity check (no motion)
    # ============================================================
    def static_test(self):
        self.get_logger().info("STATIC TEST — no motion")
        self.get_logger().info("Verify angles manually. Node will exit.")
        rclpy.shutdown()

    # ============================================================
    # Test 2: Micro‑motion test (+2° and back)
    # ============================================================
    def micro_motion_test(self):
        j = self.test_joint - 1
        if j < 0 or j > 4:
            self.get_logger().error("test_joint must be 1–5")
            rclpy.shutdown()
            return

        self.get_logger().info(f"MICRO MOTION TEST — joint {self.test_joint}")

        msg = ArmJoint()
        msg.joints = self.current_angles.copy()
        msg.joints[j] += 2.0
        msg.run_time = 200

        self.get_logger().info(f"Moving joint {self.test_joint} +2°")
        self.pub.publish(msg)

        # Move back after 1 second
        def move_back():
            msg2 = ArmJoint()
            msg2.joints = self.current_angles.copy()
            msg2.run_time = 200
            self.get_logger().info(f"Moving joint {self.test_joint} back")
            self.pub.publish(msg2)
            rclpy.shutdown()

        self.create_timer(1.0, move_back)

    # ============================================================
    # Test 3: Single waypoint test
    # ============================================================
    def single_waypoint_test(self):
        if self.traj_file == "":
            self.get_logger().error("traj_file parameter required for single waypoint test")
            rclpy.shutdown()
            return

        data = np.load(self.traj_file)
        q0 = data["q"][0]  # first waypoint
        servo_deg = drake_rad_to_servo_deg(q0)

        self.get_logger().info(f"SINGLE WAYPOINT TEST — sending first waypoint")
        self.get_logger().info(f"servo_deg={servo_deg}")

        msg = ArmJoint()
        msg.joints = list(servo_deg) + [self.current_angles[5]]
        msg.run_time = 300

        self.pub.publish(msg)
        self.get_logger().info("Sent first waypoint. Node will exit in 2s.")

        self.create_timer(2.0, lambda: rclpy.shutdown())

    # ============================================================
    # Test 4: Short trajectory test (first 0.5s)
    # ============================================================
    def short_trajectory_test(self):
        if self.traj_file == "":
            self.get_logger().error("traj_file parameter required for short trajectory test")
            rclpy.shutdown()
            return

        data = np.load(self.traj_file)
        times = data["times"]
        q = data["q"]

        self.get_logger().info(f"SHORT TRAJECTORY TEST — {len(times)} waypoints")

        self.index = 0
        self.times = times
        self.q = q
        self.dt = times[1] - times[0]

        self.timer = self.create_timer(self.dt, self.publish_next)

    def publish_next(self):
        if self.index >= len(self.times):
            self.get_logger().info("Short trajectory complete.")
            self.timer.cancel()
            rclpy.shutdown()
            return

        q_rad = self.q[self.index]
        servo_deg = drake_rad_to_servo_deg(q_rad)

        msg = ArmJoint()
        msg.joints = list(servo_deg) + [self.current_angles[5]]
        msg.run_time = int(self.dt * 1000) + 20

        self.pub.publish(msg)
        self.get_logger().info(
            f"Waypoint {self.index}/{len(self.times)}  "
            f"t={self.times[self.index]:.2f}s  "
            f"servo_deg={np.round(servo_deg, 1)}"
        )

        self.index += 1


def main():
    rclpy.init()
    node = ArmSafetyTest()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
