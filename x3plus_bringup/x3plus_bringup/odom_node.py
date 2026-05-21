#!/usr/bin/env python3

import rclpy
from rclpy.node import Node 

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler

import math


class OdometryPublisher(Node):
    

    def __init__(self):
        super().__init__('odometry_publisher_node')

        self.declare_parameter("linear_scale_x", 1.0)
        self.declare_parameter("linear_scale_y", 1.0)

        self.linear_scale_x = self.get_parameter("linear_scale_x").value
        self.linear_scale_y = self.get_parameter("linear_scale_y").value

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_time = None

        self.vel_sub_ = self.create_subscription(TwistStamped, '/vel_raw', self.vel_callback, 10)
        self.odom_pub_ = self.create_publisher(Odometry, '/odom_raw', 10)


    def vel_callback(self, msg):

        current_time = rclpy.time.Time.from_msg(msg.header.stamp)

        if self.last_time is None:
            self.last_time = current_time
            return

        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            return
        
        self.last_time = current_time

        vx = msg.twist.linear.x * self.linear_scale_x
        vy = msg.twist.linear.y * self.linear_scale_y
        w  = msg.twist.angular.z

        self.x += (vx * math.cos(self.theta) - vy * math.sin(self.theta)) * dt
        self.y += (vx * math.sin(self.theta) + vy * math.cos(self.theta)) * dt
        self.theta += w * dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        q = quaternion_from_euler(0, 0, self.theta)

        
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.pose.covariance = [0.0] * 36
        odom.pose.covariance[0] = 0.001
        odom.pose.covariance[7] = 0.001
        odom.pose.covariance[35] = 0.001

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = 0.0

        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = w

        odom.twist.covariance = [0.0] * 36
        odom.twist.covariance[0] = 0.0001
        odom.twist.covariance[7] = 0.0001
        odom.twist.covariance[35] = 0.0001

        self.odom_pub_.publish(odom)


def main():
    rclpy.init()
    node = OdometryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
