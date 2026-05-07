#!/usr/bin/env python

import rclpy
from rclpy.node import Node 
from geometry_msgs.msg import Twist
from Rosmaster_Lib import Rosmaster

class x3plusDriver(Node):

    def __init__(self):
        super().__init__('driver_node')

        self.car = Rosmaster()
        self.car.set_car_type(2)

        self.cmd_vel_sub_ = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)

        self.car.set_car_motion(0, 0, 0)
        self.car.create_receive_threading()
        self.get_logger().info("Driver node started")

        self.timer_ = self.create_timer(0.5 , self.timer_callback)

    def cmd_vel_callback(self, msg):
        vx = msg.linear.x
        vy = msg.linear.y
        angular = msg.angular.z

        self.car.set_car_motion(vx, vy, angular)

        #self.get_logger().info(f"cmd_vel: {vx}, {vy}, {angular}")

    def timer_callback(self):

        self.get_logger().info(str(self.car.get_accelerometer_data()))



def main():
    rclpy.init()
    node = x3plusDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

