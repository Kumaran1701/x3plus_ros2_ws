#!/usr/bin/env python3

import rclpy
from rclpy.node import Node 
from geometry_msgs.msg import Twist
from Rosmaster_Lib import Rosmaster
from sensor_msgs.msg import JointState  # Used for Unified Output State Telemetry
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np
from tf_transformations import quaternion_from_euler

# Imports your custom ROS 2 generated interfaces
from yahboomcar_msgs.msg import ArmJoint      # Used for Incoming Joint Commands
from yahboomcar_msgs.srv import RobotArmArray  # Used for On-Demand Service Queries

class x3plusDriver(Node):

    def __init__(self):
        super().__init__('driver_node')

        self.car = Rosmaster()
        self.car.set_car_type(2)

        # Base Subscriptions and Publishers
        self.cmd_vel_sub_ = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.joint_pub_ = self.create_publisher(JointState, '/joint_states', 10) # Unified Publisher
        self.odom_pub_ = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster_ = TransformBroadcaster(self)

        # --- ROBOTIC ARM ROS 2 INTEGRATION ---
        # 1. Custom Subscriber listens for incoming ArmJoint targets (In Degrees)
        self.arm_sub_ = self.create_subscription(ArmJoint, 'TargetAngle', self.arm_callback, 10)
        
        # 2. Custom Service provides the absolute physical angles on-demand
        self.srv_arm_angle_ = self.create_service(RobotArmArray, 'CurrentAngle', self.srv_arm_callback)
        
        # Optional Namespace Prefix Parameter
        self.declare_parameter('prefix', '')
        self.prefix = self.get_parameter('prefix').get_parameter_value().string_value
        if self.prefix and not self.prefix.endswith('/'):
            self.prefix += '/'

        # Base Odometry Variables
        self.vx_pub = 0.0
        self.vy_pub = 0.0
        self.angular_pub = 0.0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Wheel Positions
        self.fl_pos = 0.0
        self.fr_pos = 0.0
        self.rl_pos = 0.0
        self.rr_pos = 0.0

        self.last_time = self.get_clock().now()

        self.car.set_car_motion(0, 0, 0)
        self.car.create_receive_threading()
        self.get_logger().info("Driver node started with Custom Interfaces and Unified JointState Telemetry")

        # Track internal state (Array of 6 joints in Degrees)
        self.joints = [90.0, 145.0, 0.0, 45.0, 90.0, 30.0]
        self.car.set_uart_servo_angle_array(self.joints, 1000)

        # Master Loop running at 50Hz (0.02s)
        self.timer_ = self.create_timer(0.02, self.timer_callback)

    def cmd_vel_callback(self, msg):
        self.vx_pub = msg.linear.x
        self.vy_pub = msg.linear.y
        self.angular_pub = msg.angular.z
        self.car.set_car_motion(self.vx_pub, self.vy_pub, self.angular_pub)

    def arm_callback(self, msg):
        """ Listens to the Custom Input Message (In Degrees) and updates hardware """
        if len(msg.joints) != 0:
            # Full Array Command
            target_angles = list(msg.joints)
            for _ in range(2):
                self.car.set_uart_servo_angle_array(target_angles, msg.run_time)
                self.joints = target_angles
        else:
            # Single Joint Command
            for _ in range(2):
                self.car.set_uart_servo_angle(msg.id, msg.angle, msg.run_time)
                self.joints[msg.id - 1] = msg.angle

    def srv_arm_callback(self, request, response):
        """ Handles synchronous system checks directly over the hardware bus """
        joints = self.car.get_uart_servo_angle_array()
        response.angles = [float(val) for val in joints]
        return response

    def timer_callback(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            return
        self.last_time = current_time

        # --- BASE ODOMETRY CALCULATIONS ---
        measured_vx, measured_vy, measured_angular = self.car.get_motion_data()
        self.x += (measured_vx * math.cos(self.theta) - measured_vy * math.sin(self.theta)) * dt
        self.y += (measured_vx * math.sin(self.theta) + measured_vy * math.cos(self.theta)) * dt
        self.theta += measured_angular * dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Broadcast Transforms (TF)
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, self.theta)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster_.sendTransform(t)

        # Publish Odometry Msg
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = measured_vx
        odom.twist.twist.linear.y = measured_vy
        odom.twist.twist.angular.z = measured_angular
        
        odom.pose.covariance = [0.0] * 36
        odom.pose.covariance[0] = 0.001
        odom.pose.covariance[7] = 0.001
        odom.pose.covariance[35] = 0.001
        odom.twist.covariance = [0.0] * 36
        odom.twist.covariance[0] = 0.0001
        odom.twist.covariance[7] = 0.0001
        odom.twist.covariance[35] = 0.0001
        self.odom_pub_.publish(odom)

        # --- WHEEL POSITION CALCULATIONS ---
        r, lx, ly = 0.04, 0.12, 0.10
        fl_vel = (measured_vx - measured_vy - (lx + ly) * measured_angular) / r
        fr_vel = (measured_vx + measured_vy + (lx + ly) * measured_angular) / r
        rl_vel = (measured_vx + measured_vy - (lx + ly) * measured_angular) / r
        rr_vel = (measured_vx - measured_vy + (lx + ly) * measured_angular) / r

        self.fl_pos += fl_vel * dt
        self.fr_pos += fr_vel * dt
        self.rl_pos += rl_vel * dt
        self.rr_pos += rr_vel * dt

        # --- UNIFIED JOINT_STATE TELEMETRY OUTPUT (WHEELS + ARM) ---
        joint_state = JointState()
        joint_state.header.stamp = current_time.to_msg()

        # Step 1: Inject Wheel Names & Data
        joint_state.name = [
            'front_right_joint', 'front_left_joint',
            'back_right_joint', 'back_left_joint'
        ]
        joint_state.position = [
            self.fr_pos, self.fl_pos, self.rr_pos, self.rl_pos
        ]

        # Step 2: Inject Dynamic Arm Joint Names (Applies Namespace Prefix if used)
        arm_names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "grip_joint"]
        joint_state.name.extend([self.prefix + name for name in arm_names])

        # Step 3: Math Pipeline for Arm Data (Converts raw degrees to ROS 2 standard Radians)
        arm_joints_deg = list(self.joints)
        
        # Linearly map the physical gripper range [30, 180] to a simulated workspace range [0, 90]
        arm_joints_deg[5] = float(np.interp(arm_joints_deg[5], [30.0, 180.0], [0.0, 90.0]))
        
        # Zero-center the angles around a neutral reference frame (Subtract 90 degrees)
        mid_offset = np.array([90.0] * 6)
        normalized_deg = np.array(arm_joints_deg) - mid_offset
        
        # Scale to standard ROS SI Units (Radians) and feed to output array
        rad_positions = list(normalized_deg * (math.pi / 180.0))
        joint_state.position.extend(rad_positions)

        # Publish the final combined state packet to the ROS ecosystem [1]
        self.joint_pub_.publish(joint_state)


def main():
    rclpy.init()
    node = x3plusDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
