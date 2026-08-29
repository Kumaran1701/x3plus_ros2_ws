#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy


class EkfMecanumRotationNode(Node):

    def __init__(self):
        super().__init__('ekf_mecanum_rotation_node')
        
        # QoS for latched messages
        qos_transient = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.rotation_finished_pub = self.create_publisher(Bool, '/tsdf_rotation_finished', qos_transient)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.capture_started_sub = self.create_subscription(
            Bool, '/tsdf_capture_started', self.capture_started_cb, qos_transient
        )
        
        # Configuration
        self.target_offset = math.radians(30.0)
        self.rotation_speed = 0.2
        self.tolerance = math.radians(1.5)
        
        self.sequence = [
            (-self.target_offset, "Rotating 30° RIGHT"),
            (0.0,                 "Returning to CENTER (from right)"),
            (self.target_offset,  "Rotating 30° LEFT"),
            (0.0,                 "Returning to CENTER (from left)")
        ]
        
        self.current_step = 0
        self.current_yaw = 0.0
        self.start_yaw = None
        self.initialized = False

        self.capture_started = False

        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("Rotation node started — waiting for capture_started...")

    def capture_started_cb(self, msg: Bool):
        if msg.data:
            self.capture_started = True
            self.get_logger().info("Capture started — beginning rotation sequence.")

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        absolute_yaw = self.quaternion_to_yaw(q)
        
        if not self.initialized:
            self.start_yaw = absolute_yaw
            self.initialized = True
            
        raw_diff = absolute_yaw - self.start_yaw
        self.current_yaw = math.atan2(math.sin(raw_diff), math.cos(raw_diff))

    def control_loop(self):
        if not self.initialized or not self.capture_started:
            return

        if self.current_step >= len(self.sequence):
            self.stop_robot()
            self.rotation_finished_pub.publish(Bool(data=True))
            self.get_logger().info("Rotation sequence completed — rotation_finished published.")
            self.destroy_timer(self.timer)
            return

        target_yaw, description = self.sequence[self.current_step]
        
        error = target_yaw - self.current_yaw
        error = math.atan2(math.sin(error), math.cos(error))
        
        if abs(error) > self.tolerance:
            direction = 1.0 if error > 0 else -1.0
            
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_footprint'
            msg.twist.angular.z = direction * self.rotation_speed
            self.cmd_vel_pub.publish(msg)
        else:
            self.stop_robot()
            self.get_logger().info(f"Finished: {description}")
            self.current_step += 1

    def stop_robot(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EkfMecanumRotationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
