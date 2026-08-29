#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import math

class EkfMecanumRotationNode(Node):

    def __init__(self):
        super().__init__('ekf_mecanum_rotation_node')
        
        # Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        
        # NOTE: If your EKF outputs to a different topic name, change '/odom' to match it
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Configuration (Angles in Radians)
        self.target_offset = math.radians(30.0)  # ~0.5236 rad
        self.rotation_speed = 0.2                # rad/s
        self.tolerance = math.radians(1.5)       # ~1.5 degree error threshold
        
        # Sequence Definition: [Target relative yaw, Step Name]
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

        # Control loop timer (50Hz)
        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("EKF Mecanum node started. Waiting for filtered odometry...")

    def quaternion_to_yaw(self, q):
        """Converts quaternion (x, y, z, w) to yaw (rotation around Z-axis)."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        # Extract orientation quaternion from the EKF filtered message
        q = msg.pose.pose.orientation
        absolute_yaw = self.quaternion_to_yaw(q)
        
        if not self.initialized:
            # Capture the robot's initial baseline heading as absolute zero
            self.start_yaw = absolute_yaw
            self.initialized = True
            
        # Calculate the current yaw relative to our starting orientation
        # Normalize the angle variation to keep it within [-pi, pi]
        raw_diff = absolute_yaw - self.start_yaw
        self.current_yaw = math.atan2(math.sin(raw_diff), math.cos(raw_diff))

    def control_loop(self):
        if not self.initialized:
            return

        if self.current_step >= len(self.sequence):
            self.stop_robot()
            self.get_logger().info("EKF-guided sequence successfully completed! Shutting down.")
            self.destroy_timer(self.timer)
            rclpy.shutdown()
            return

        target_yaw, description = self.sequence[self.current_step]
        
        # Compute shortest angular error
        error = target_yaw - self.current_yaw
        error = math.atan2(math.sin(error), math.cos(error))
        
        if abs(error) > self.tolerance:
            # Multiplier sets direction: positive error = turn CCW (left), negative = turn CW (right)
            direction = 1.0 if error > 0 else -1.0
            
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_footprint'
            msg.twist.angular.z = direction * self.rotation_speed
            self.cmd_vel_pub.publish(msg)
        else:
            self.stop_robot()
            self.get_logger().info(f"Finished: {description} (Current Relative Yaw: {math.degrees(self.current_yaw):.2f}°)")
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
        if rclpy.ok():
            node.stop_robot()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
