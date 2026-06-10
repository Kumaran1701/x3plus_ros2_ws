#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

class x3plusCameraArm(Node):

    def __init__(self):
        super().__init__('camera_arm_node')

        self.camera_arm_pub_ = self.create_publisher(Image, '/camera_arm/image_raw', 10)
        
        self.bridge = CvBridge()

        self.capture = cv2.VideoCapture(0)

        self.timer = self.create_timer(0.3, self.timer_callback)

        self.get_logger().info("Camera_Arm_Node started")

    def timer_callback(self):
        ret, frame = self.capture.read()

        if not ret:
            self.get_logger().info("Failed to capture frame")
            return
        
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "mono_link"

        self.camera_arm_pub_.publish(msg)

def main():
    rclpy.init()
    node = x3plusCameraArm()
    rclpy.spin(node)
    node.capture.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

