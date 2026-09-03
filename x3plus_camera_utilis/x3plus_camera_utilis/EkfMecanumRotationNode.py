#!/usr/bin/env python3
"""
Continuous rotation sequencer for TSDF data collection, with closed-loop
translational drift correction.

Reverted to continuous motion (no stop/settle/wait-for-ack cycles) -
mecanum wheels slip more during start/stop transients than during
smooth motion, so a design built around ~150+ stop/start cycles was
fighting the platform. Capture density is now controlled entirely on
the recorder side (angular spacing + content freshness checks), so
this node just needs to sweep smoothly through the same targets as
before.

Drift correction: the original version only ever commanded angular.z,
so any translational drift from wheel slip during rotation was fully
open-loop - nothing ever pulled the base back toward its start
position. A mecanum base is holonomic, so this version continuously
servos linear.x/linear.y (in the body frame) to hold the base's XY
position at its starting point, using the same EKF odometry already
being read for yaw. This runs simultaneously with the yaw sweep, not
as a separate phase.

Caveat: this corrects against the EKF's OWN position estimate. If the
EKF's estimate has itself drifted from true physical position (e.g.
unmodeled slip with no external reference like a fiducial or lidar),
holding against that estimate stops further accumulation but can't
recover the true original position. Still strictly better than leaving
translation fully uncorrected.

Sequence (relative yaw from the starting heading), matching the
original physical motion plan:
    0 -> -30 (right)  -> 0 (back to centre)
    0 -> +30 (left)   -> 0 (back to centre)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import math


class EkfMecanumRotationNode(Node):

    def __init__(self):
        super().__init__('ekf_mecanum_rotation_node')

        # --- Configuration ---
        self.declare_parameter('target_offset_deg', 30.0)
        self.declare_parameter('rotation_speed', 0.25)         # rad/s
        self.declare_parameter('yaw_tolerance_deg', 1.5)
        self.declare_parameter('kp_linear', 0.8)                # proportional gain, position error (m) -> linear speed (m/s)
        self.declare_parameter('max_linear_speed', 0.15)        # m/s clamp on drift-correction strafing

        target_offset_deg = self.get_parameter('target_offset_deg').value
        self.rotation_speed = self.get_parameter('rotation_speed').value
        self.tolerance = math.radians(self.get_parameter('yaw_tolerance_deg').value)
        self.kp_linear = self.get_parameter('kp_linear').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value

        target_offset = math.radians(target_offset_deg)
        self.sequence = [
            (-target_offset, "Rotating RIGHT"),
            (0.0,             "Returning to CENTER (from right)"),
            (target_offset,   "Rotating LEFT"),
            (0.0,             "Returning to CENTER (from left)"),
        ]
        self.current_step = 0

        # Publishers / subscribers
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.pub_recording_enabled = self.create_publisher(Bool, '/recording_enabled', 10)

        self.current_yaw = 0.0
        self.start_yaw = None
        self.absolute_yaw = 0.0
        self.start_position = None   # (x, y) in odom frame
        self.current_position = None
        self.initialized = False

        self.timer = self.create_timer(0.02, self.control_loop)  # 50 Hz
        self.get_logger().info("EKF Mecanum node started. Waiting for filtered odometry...")

    # ------------------------------------------------------------------

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        pos = msg.pose.pose.position
        self.absolute_yaw = self.quaternion_to_yaw(q)
        self.current_position = (pos.x, pos.y)

        if not self.initialized:
            self.start_yaw = self.absolute_yaw
            self.start_position = (pos.x, pos.y)
            self.initialized = True
            self.pub_recording_enabled.publish(Bool(data=True))

        raw_diff = self.absolute_yaw - self.start_yaw
        self.current_yaw = math.atan2(math.sin(raw_diff), math.cos(raw_diff))

    def linear_correction_body_frame(self):
        """Proportional position hold: returns (vx, vy) in the body frame
        that drives the base back toward its start XY position."""
        dx_world = self.start_position[0] - self.current_position[0]
        dy_world = self.start_position[1] - self.current_position[1]

        # Rotate the world-frame error into the body frame using the
        # ACTUAL current heading (absolute_yaw), not the start-relative one.
        cos_yaw = math.cos(self.absolute_yaw)
        sin_yaw = math.sin(self.absolute_yaw)
        vx_body = cos_yaw * dx_world + sin_yaw * dy_world
        vy_body = -sin_yaw * dx_world + cos_yaw * dy_world

        vx = max(-self.max_linear_speed, min(self.max_linear_speed, self.kp_linear * vx_body))
        vy = max(-self.max_linear_speed, min(self.max_linear_speed, self.kp_linear * vy_body))
        return vx, vy

    def control_loop(self):
        if not self.initialized:
            return

        if self.current_step >= len(self.sequence):
            self.pub_recording_enabled.publish(Bool(data=False))
            self.stop_robot()
            self.get_logger().info("Sequence complete.")
            self.destroy_timer(self.timer)
            return

        target_yaw, description = self.sequence[self.current_step]
        error = target_yaw - self.current_yaw
        error = math.atan2(math.sin(error), math.cos(error))

        vx, vy = self.linear_correction_body_frame()

        if abs(error) > self.tolerance:
            direction = 1.0 if error > 0 else -1.0
            self.publish_twist(direction * self.rotation_speed, vx, vy)
        else:
            self.publish_twist(0.0, vx, vy)
            self.get_logger().info(
                f"Finished: {description} (yaw={math.degrees(self.current_yaw):.2f} deg)")
            self.current_step += 1

    def publish_twist(self, angular_z, vx=0.0, vy=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.angular.z = angular_z
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        self.publish_twist(0.0, 0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = EkfMecanumRotationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop_robot()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()