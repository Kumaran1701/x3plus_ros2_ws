#!/usr/bin/env python3

import rclpy
from rclpy.node import Node 
from geometry_msgs.msg import TwistStamped
from Rosmaster_Lib import Rosmaster
from sensor_msgs.msg import Imu, MagneticField, JointState
import math
import numpy as np
from std_msgs.msg import Float32
from x3plus_msgs.msg import ArmJoint
from x3plus_msgs.srv import RobotArmArray

class x3plusDriver(Node):

    def __init__(self):
        super().__init__('driver_node')

        self.car = Rosmaster()
        self.car.set_car_type(2)

        self.vx_pub = 0.0
        self.vy_pub = 0.0
        self.angular_pub = 0.0

        self.fl_pos = 0.0
        self.fr_pos = 0.0
        self.rl_pos = 0.0
        self.rr_pos = 0.0

        self.cmd_vel_sub_ = self.create_subscription(TwistStamped, 'cmd_vel', self.cmd_vel_callback, 10)
        self.arm_sub_ = self.create_subscription(ArmJoint, 'TargetAngle', self.arm_callback, 10)
        
        self.voltage_pub_ = self.create_publisher(Float32, 'voltage', 10)
        self.imu_pub_ = self.create_publisher(Imu, '/imu/raw', 10)
        self.mag_pub_ = self.create_publisher(MagneticField, '/mag/raw', 10)
        self.vel_raw_pub_ = self.create_publisher(TwistStamped, '/vel_raw', 10)
        self.joint_pub_ = self.create_publisher(JointState, '/joint_states', 10)

        self.srv_arm_angle_ = self.create_service(RobotArmArray, 'CurrentAngle', self.srv_arm_callback)

        self.declare_parameter('prefix', '')
        self.prefix = self.get_parameter('prefix').get_parameter_value().string_value
        if self.prefix and not self.prefix.endswith('/'):
            self.prefix += '/'

        self.car.set_car_motion(0, 0, 0)
        self.car.create_receive_threading()
        self.get_logger().info("Driver node started")

        self.joints = [90, 90, 90, 90, 90, 90]
        self.car.set_uart_servo_angle_array(self.joints, 1000)

        self.last_time = self.get_clock().now()
        self.timer_ = self.create_timer(0.02 , self.timer_callback)

    def cmd_vel_callback(self, msg):
        self.vx_pub = msg.twist.linear.x
        self.vy_pub = msg.twist.linear.y
        self.angular_pub = msg.twist.angular.z

        self.car.set_car_motion(self.vx_pub, self.vy_pub, self.angular_pub)

        #self.get_logger().info(f"cmd_vel: {vx}, {vy}, {angular}")

    def arm_callback(self, msg):
        if len(msg.joints) != 0:
            target_angles = list(msg.joints)
            for i in range(2):
                self.car.set_uart_servo_angle_array(target_angles, msg.run_time)
                self.joints = target_angles

        else:
            for i in range(2):
                self.car.set_uart_servo_angle(msg.id, msg.angle, msg.run_time)
                self.joints[msg.id - 1] = msg.angle

    def srv_arm_callback(self, request, response):
        joints = self.car.get_uart_servo_angle_array()
        response.angles = [float(val) for val in joints]
        return response

    def timer_callback(self):

        twist = TwistStamped()
        imu = Imu()
        battery = Float32()
        mag = MagneticField()
        
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            return
        self.last_time = current_time

        battery.data = self.car.get_battery_voltage()
        acc_x, acc_y, acc_z = self.car.get_accelerometer_data()
        gyro_x, gyro_y, gyro_z = self.car.get_gyroscope_data()
        mag_x, mag_y, mag_z = self.car.get_magnetometer_data()
        vel_x, vel_y, vel_angular = self.car.get_motion_data()

        imu.header.stamp = current_time.to_msg()
        imu.header.frame_id = 'imu_link'
        imu.linear_acceleration.x = acc_x
        imu.linear_acceleration.y = acc_y
        imu.linear_acceleration.z = acc_z
        imu.angular_velocity.x = gyro_x
        imu.angular_velocity.y = gyro_y
        imu.angular_velocity.z = gyro_z

        mag.header.stamp = current_time.to_msg()
        mag.header.frame_id = 'imu_link'
        mag.magnetic_field.x = mag_x
        mag.magnetic_field.y = mag_y
        mag.magnetic_field.z = mag_z

        twist.header.stamp = current_time.to_msg()
        twist.header.frame_id = 'base_footprint'
        twist.twist.linear.x = vel_x
        twist.twist.linear.y = vel_y
        twist.twist.angular.z = vel_angular

        self.imu_pub_.publish(imu)
        self.mag_pub_.publish(mag)
        self.vel_raw_pub_.publish(twist)
        self.voltage_pub_.publish(battery)

        r = 0.04
        lx = 0.12
        ly = 0.10

        fl_vel = (vel_x - vel_y - (lx + ly) * vel_angular) / r
        fr_vel = (vel_x + vel_y + (lx + ly) * vel_angular) / r
        rl_vel = (vel_x + vel_y - (lx + ly) * vel_angular) / r
        rr_vel = (vel_x - vel_y + (lx + ly) * vel_angular) / r

        self.fl_pos += fl_vel * dt
        self.fr_pos += fr_vel * dt
        self.rl_pos += rl_vel * dt
        self.rr_pos += rr_vel * dt

        joint_state = JointState()
        joint_state.header.stamp = current_time.to_msg()

        joint_state.name = [
            'front_right_joint', 'front_left_joint',
            'back_right_joint', 'back_left_joint'
        ]

        joint_state.position = [self.fr_pos, self.fl_pos, self.rr_pos, self.rl_pos]

        arm_names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "grip_joint",]
        joint_state.name.extend([self.prefix + name for name in arm_names])

        arm_joints_deg = list(self.joints)

        arm_joints_deg[5] = float(np.interp(arm_joints_deg[5], [30.0, 180.0], [0.0, 90.0]))
        mid_offset = np.array([90.0] * 6)
        normalized_deg = np.array(arm_joints_deg) - mid_offset

        rad_positions = list(normalized_deg * (math.pi / 180.0))
        joint_state.position.extend(rad_positions)

        self.joint_pub_.publish(joint_state)

def main():
    rclpy.init()
    node = x3plusDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

