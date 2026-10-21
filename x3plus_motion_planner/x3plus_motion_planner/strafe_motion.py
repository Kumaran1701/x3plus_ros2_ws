#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import math


class EkfMecanumStrafeNode(Node):

    def __init__(self):
        """
        Strafe 10 cm right from the initial pose using EKF-filtered odometry.

        Position is controlled using the robot's initial reference frame,
        while heading is actively maintained using yaw feedback.
        """

        super().__init__('ekf_mecanum_strafe_node')

        # Closed loop controller gains and limits.
        self.declare_parameter('kp_linear', 1.5)           # Speed scale based on position error.
        self.declare_parameter('kp_angular', 2.0)          # Correction scale to keep heading locked.
        self.declare_parameter('linear_tolerance_m', 0.01) # 1 cm accuracy threshold.
        self.declare_parameter('max_linear_speed', 0.15)   # Limit speed (m/s) to prevent aggressive transients.
        self.declare_parameter('max_angular_speed', 0.25)  # Limit rotation speed (rad/s) for holding heading.

        self.kp_linear = self.get_parameter('kp_linear').value
        self.kp_angular = self.get_parameter('kp_angular').value
        self.tolerance = self.get_parameter('linear_tolerance_m').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value

        # Target lateral offset relative to the initial robot pose.
        # Positive Y = left, Negative Y = right (ROS convention).
        self.sequence = [
            (-0.10, "Strafing 10cm RIGHT from center"),
        ]
        self.current_step = 0

        # Publishers and subscriber
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.pub_recording_enabled = self.create_publisher(Bool, '/recording_enabled', 10)

        self.odom_sub = self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        
        # State tracking variables
        self.start_yaw = None
        self.absolute_yaw = 0.0
        self.start_position = None   # (x, y) in odom frame
        self.current_position = None # (x, y) in odom frame
        self.initialized = False

        self.timer = self.create_timer(0.02, self.control_loop)  # 50 Hz Loop
        self.get_logger().info("EKF Mecanum Strafe node started. Waiting for filtered odometry...")


    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    

    def odom_callback(self, msg):
        """
        Callback to update robot's pose from EKF-filterd odometry.
        The first received odometry message will:
        1. Establish initial reference frame for lateral strafe motion.
        2. Start the RGBDPoseRecorder node to save the data throughout the strafe sequence. 
        """

        q = msg.pose.pose.orientation
        pos = msg.pose.pose.position
        self.absolute_yaw = self.quaternion_to_yaw(q)
        self.current_position = (pos.x, pos.y)

        if not self.initialized:
            self.start_yaw = self.absolute_yaw
            self.start_position = (pos.x, pos.y)
            self.initialized = True
            # Enable recorder node to start capturing RGBD-Pose data during the strafe sequence.
            self.pub_recording_enabled.publish(Bool(data=True))

    def control_loop(self):

        if not self.initialized:
            return

        # Check sequence termination
        if self.current_step >= len(self.sequence):
            self.pub_recording_enabled.publish(Bool(data=False))
            self.stop_robot()
            self.get_logger().info("Strafe Sequence complete.")
            self.destroy_timer(self.timer)
            return

        # Target relative Y parameter
        target_relative_y, description = self.sequence[self.current_step]

        # Current displacement from the starting pose in odom coordinates
        dx_world = self.current_position[0] - self.start_position[0]
        dy_world = self.current_position[1] - self.start_position[1]

        # Transform displacement in the robot's initial frame.
        # Allows trajectory tracking realtive to starting pose regardless of robots current heading.
        cos_start = math.cos(self.start_yaw)
        sin_start = math.sin(self.start_yaw)
        current_relative_x = cos_start * dx_world + sin_start * dy_world
        current_relative_y = -sin_start * dx_world + cos_start * dy_world

        # Position errors.
        error_x = 0.0 - current_relative_x 
        error_y = target_relative_y - current_relative_y
        
        # Heading error: maintain initial heading throughout the motion.
        error_yaw = self.start_yaw - self.absolute_yaw
        error_yaw = math.atan2(math.sin(error_yaw), math.cos(error_yaw))

        # 4. Proportional position controller.
        vx_start = self.kp_linear * error_x
        vy_start = self.kp_linear * error_y
        omega_z = self.kp_angular * error_yaw

        # Convert velocities from the initial frame to the current robot frame before publishing to /cmd_vel.
        heading_delta = self.absolute_yaw - self.start_yaw
        cos_delta = math.cos(heading_delta)
        sin_delta = math.sin(heading_delta)
        
        vx_body = cos_delta * vx_start + sin_delta * vy_start
        vy_body = -sin_delta * vx_start + cos_delta * vy_start

        # Apply velocitiy limits.
        vx = max(-self.max_linear_speed, min(self.max_linear_speed, vx_body))
        vy = max(-self.max_linear_speed, min(self.max_linear_speed, vy_body))
        v_omega = max(-self.max_angular_speed, min(self.max_angular_speed, omega_z))

        # Check if step target has been reached
        if abs(error_y) <= self.tolerance:
            self.publish_twist(0.0, 0.0, v_omega) # Breifly pause translation and maintain heading hold
            self.get_logger().info(f"Finished Step {self.current_step + 1}: {description} (Relative Y={current_relative_y:.3f}m)")
            self.current_step += 1
        else:
            # Continue active driving execution
            self.publish_twist(v_omega, vx, vy)


    def publish_twist(self, angular_z, vx=0.0, vy=0.0):
        """
        Publish velocity in body-frame to /cmd_vel.
        """
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
    node = EkfMecanumStrafeNode()
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
