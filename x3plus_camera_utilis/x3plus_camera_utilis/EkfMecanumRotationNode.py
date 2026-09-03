#!/usr/bin/env python3
"""
Stop-and-go rotation sequencer for TSDF data collection.

Redesigned from a continuous-sweep-with-opportunistic-capture pattern to
an explicit stop -> settle -> request capture -> wait for ack -> proceed
pattern. This eliminates motion blur entirely and, combined with the
recorder's content-level freshness check, removes the failure mode
where a stale depth frame got paired with a fresh (moving) pose.

It also directly controls capture density: waypoints are spaced every
capture_step_deg degrees (default 1.5), which is a far more direct
handle on "how many captures do I need" than letting frame count fall
out of however fast the robot happened to move relative to however
fast the camera happened to publish.

Sequence (relative yaw from the starting heading), matching the
original physical motion plan:
    0 -> -30 (right)  -> 0 (back to centre)
    0 -> +30 (left)   -> 0 (back to centre)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int32
import math


def build_waypoints(target_offset_rad, step_rad):
    """Discrete waypoints (relative yaw, radians) covering
    0 -> -target -> 0 -> +target -> 0, spaced by step_rad."""
    legs = [(0.0, -target_offset_rad), (-target_offset_rad, 0.0),
            (0.0, target_offset_rad), (target_offset_rad, 0.0)]
    waypoints = []
    for (a, b) in legs:
        n = max(1, round(abs(b - a) / step_rad))
        for k in range(1, n + 1):
            waypoints.append(a + (b - a) * k / n)
    return waypoints


class EkfMecanumRotationNode(Node):

    def __init__(self):
        super().__init__('ekf_mecanum_rotation_node')

        # --- Configuration ---
        self.declare_parameter('target_offset_deg', 30.0)
        self.declare_parameter('capture_step_deg', 1.5)          # spacing between capture waypoints
        self.declare_parameter('rotation_speed', 0.25)            # rad/s while driving between waypoints
        self.declare_parameter('yaw_tolerance_deg', 0.3)          # how precisely to hit each waypoint
        self.declare_parameter('settle_time_sec', 0.3)            # dwell after stopping, before requesting capture
        self.declare_parameter('capture_ack_timeout_sec', 3.0)    # give up waiting for this waypoint's ack
        # NOTE: keep capture_ack_timeout_sec > the recorder's stale_retry_timeout_sec (default 2.0s)
        # so the recorder's own give-up fires first when depth is genuinely stuck.

        target_offset_deg = self.get_parameter('target_offset_deg').value
        capture_step_deg = self.get_parameter('capture_step_deg').value
        self.rotation_speed = self.get_parameter('rotation_speed').value
        self.tolerance = math.radians(self.get_parameter('yaw_tolerance_deg').value)
        self.settle_time_sec = self.get_parameter('settle_time_sec').value
        self.capture_ack_timeout_sec = self.get_parameter('capture_ack_timeout_sec').value

        self.waypoints = build_waypoints(
            math.radians(target_offset_deg), math.radians(capture_step_deg))
        self.get_logger().info(
            f"Built {len(self.waypoints)} waypoints "
            f"(step={capture_step_deg} deg, target=+/-{target_offset_deg} deg)."
        )

        # Publishers / subscribers
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.pub_ready_for_capture = self.create_publisher(Bool, '/ready_for_capture', 10)
        self.sub_capture_ack = self.create_subscription(
            Int32, '/capture_ack', self.capture_ack_callback, 10)

        # State machine phases: init -> settling (initial yaw=0 capture) -> awaiting_capture
        # -> moving -> settling -> awaiting_capture -> ... -> done
        self.phase = 'init'
        self.waypoint_idx = -1   # -1 = the initial (yaw=0) capture, before any motion
        self.current_yaw = 0.0
        self.start_yaw = None
        self.initialized = False

        self.settle_start_time = None
        self.capture_request_time = None
        self.last_seen_ack_count = 0
        self.ack_count_at_request = 0

        self.timer = self.create_timer(0.02, self.control_loop)  # 50 Hz control resolution
        self.get_logger().info("Waiting for filtered odometry...")

    # ------------------------------------------------------------------

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        absolute_yaw = self.quaternion_to_yaw(q)
        if not self.initialized:
            self.start_yaw = absolute_yaw
            self.initialized = True
            self.phase = 'settling'  # go straight to requesting the initial (yaw=0) capture
            self.settle_start_time = self.get_clock().now()
        raw_diff = absolute_yaw - self.start_yaw
        self.current_yaw = math.atan2(math.sin(raw_diff), math.cos(raw_diff))

    def capture_ack_callback(self, msg):
        self.last_seen_ack_count = msg.data

    def current_target(self):
        if self.waypoint_idx < 0:
            return 0.0
        return self.waypoints[self.waypoint_idx]

    # ------------------------------------------------------------------

    def control_loop(self):
        if not self.initialized or self.phase == 'done':
            return

        if self.phase == 'moving':
            target_yaw = self.current_target()
            error = target_yaw - self.current_yaw
            error = math.atan2(math.sin(error), math.cos(error))

            if abs(error) > self.tolerance:
                direction = 1.0 if error > 0 else -1.0
                self.publish_twist(direction * self.rotation_speed)
            else:
                self.stop_robot()
                self.phase = 'settling'
                self.settle_start_time = self.get_clock().now()
            return

        if self.phase == 'settling':
            self.stop_robot()
            elapsed = (self.get_clock().now() - self.settle_start_time).nanoseconds / 1e9
            if elapsed >= self.settle_time_sec:
                self.ack_count_at_request = self.last_seen_ack_count
                self.capture_request_time = self.get_clock().now()
                self.pub_ready_for_capture.publish(Bool(data=True))
                self.phase = 'awaiting_capture'
            return

        if self.phase == 'awaiting_capture':
            if self.last_seen_ack_count > self.ack_count_at_request:
                # Recorder confirmed a valid, non-stale frame was saved here.
                self.pub_ready_for_capture.publish(Bool(data=False))
                self.advance_to_next_waypoint()
                return

            elapsed = (self.get_clock().now() - self.capture_request_time).nanoseconds / 1e9
            if elapsed > self.capture_ack_timeout_sec:
                self.get_logger().warn(
                    f"No capture ack for waypoint {self.waypoint_idx} "
                    f"(yaw={math.degrees(self.current_target()):.2f} deg) after "
                    f"{elapsed:.1f}s - moving on. This waypoint will be a gap "
                    f"in the dataset rather than risking a stale/corrupted frame."
                )
                self.pub_ready_for_capture.publish(Bool(data=False))
                self.advance_to_next_waypoint()
            return

    def advance_to_next_waypoint(self):
        self.waypoint_idx += 1
        if self.waypoint_idx >= len(self.waypoints):
            self.phase = 'done'
            self.stop_robot()
            self.get_logger().info("Sequence complete.")
            return
        self.phase = 'moving'
        self.get_logger().info(
            f"Moving to waypoint {self.waypoint_idx}/{len(self.waypoints)} "
            f"(target yaw={math.degrees(self.current_target()):.2f} deg)"
        )

    def publish_twist(self, angular_z):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.angular.z = angular_z
        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        self.publish_twist(0.0)


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