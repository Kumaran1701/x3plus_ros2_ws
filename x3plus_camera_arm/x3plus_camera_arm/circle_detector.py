#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

import numpy as np

class x3plusCircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector_node')

        self.bridge = CvBridge()
        self.camera_arm_pub_ = self.create_publisher(Image, '/camera_arm/image_circle', 10)
        self.camera_arm_sub_ = self.create_subscription(Image, '/camera_arm/image_raw', self.image_callback, 10)


    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, encoding='bgr8')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_blurred = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            image=gray_blurred,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )

        if circles is not None:
            circles = np.unit16(np.around(circles))

            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

        output = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        output.header.stamp = msg.header.stamp
        output.header.frame_id = msg.header.frame_id

        self.camera_arm_pub_.publish(output)

def main():
    rclpy.init()
    node = x3plusCircleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        