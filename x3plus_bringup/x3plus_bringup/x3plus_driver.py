#!/usr/bin/env python3

import rclpy
from rclpy.node import Node 
from geometry_msgs.msg import Twist
from Rosmaster_Lib import Rosmaster
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class x3plusDriver(Node):

    def __init__(self):
        super().__init__('driver_node')

        self.car = Rosmaster()
        self.car.set_car_type(2)

        self.cmd_vel_sub_ = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.joint_pub_ = self.create_publisher(JointState, '/joint_states', 10)
        self.odom_pub_ = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster_ = TransformBroadcaster(self)

        self.vx = 0.0
        self.vy = 0.0
        self.angular = 0.0

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.fl_pos = 0.0
        self.fr_pos = 0.0
        self.rl_pos = 0.0
        self.rr_pos = 0.0

        self.last_time = self.get_clock().now()

        self.car.set_car_motion(0, 0, 0)
        self.car.create_receive_threading()
        self.get_logger().info("Driver node started")

        self.timer_ = self.create_timer(0.02 , self.timer_callback)

    def cmd_vel_callback(self, msg):
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.angular = msg.angular.z

        self.car.set_car_motion(self.vx, self.vy, self.angular)

        #self.get_logger().info(f"cmd_vel: {vx}, {vy}, {angular}")

    def timer_callback(self):

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        self.x += (self.vx * math.cos(self.theta) - self.vy * math.sin(self.theta)) * dt
        self.y += (self.vx * math.sin(self.theta) + self.vy * math.cos(self.theta)) * dt
        self.theta += self.wz * dt

        t = TransformStamped()

        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)

        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster_.sendTransform(t)

        odom = Odometry()

        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'

        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = self.vy
        odom.twist.twist.angular.z = self.wz

        self.odom_pub_.publish(odom)

        r = 0.05
        lx = 0.12
        ly = 0.10

        fl_vel = (self.vx - self.vy - (lx + ly) * self.wz) / r

        fr_vel = (self.vx + self.vy + (lx + ly) * self.wz) / r

        rl_vel = (self.vx + self.vy - (lx + ly) * self.wz) / r

        rr_vel = (self.vx - self.vy + (lx + ly) * self.wz) / r

        self.fl_pos += fl_vel * dt
        self.fr_pos += fr_vel * dt
        self.rl_pos += rl_vel * dt
        self.rr_pos += rr_vel * dt

        joint_state = JointState()

        joint_state.header.stamp = current_time.to_msg()

        joint_state.name = [
            'front_right_joint',
            'front_left_joint',
            'back_right_joint',
            'back_left_joint'
        ]

        joint_state.position = [
            self.fl_pos,
            self.fr_pos,
            self.rl_pos,
            self.rr_pos
        ]

        self.joint_pub_.publish(joint_state)



def main():
    rclpy.init()
    node = x3plusDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

