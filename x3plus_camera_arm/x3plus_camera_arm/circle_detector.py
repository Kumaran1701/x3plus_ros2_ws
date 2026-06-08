#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener

import cv2
from cv_bridge import CvBridge

import numpy as np

class x3plusCircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector_node')

        self.depth_frame_flag = False

        self.fx = 543.2409
        self.fy = 543.2409
        self.cx0 = 323.0906
        self.cy0 = 241.3283

        self.bridge = CvBridge()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.camera_arm_pub_ = self.create_publisher(Image, '/camera_arm/image_circle', 10)
        self.depth_camera_pub_ = self.create_publisher(Image, '/camera_arm/depth_image', 10)
        #self.camera_arm_sub_ = self.create_subscription(Image, '/camera_arm/image_raw', self.image_callback, 10)
        self.camera_arm_sub_ = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.depth_camera_sub_ = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_frame_flag = True

    def image_callback(self, msg):

        if not self.depth_frame_flag:
            return
        point = PointStamped()

        frame = self.bridge.imgmsg_to_cv2(msg)

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
            maxRadius=20
        )

        depth_vis = cv2.normalize(
            self.depth_frame,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        depth_vis = depth_vis.astype(np.uint8)

        if circles is not None:
            circles = np.uint16(np.around(circles))

            circle = circles[0][0]
            cx = int(circle[0])
            cy = int(circle[1])
            r = int(circle[2])

            cv2.circle(frame, (cx, cy), r, (0, 0, 255), 5)
            cv2.circle(depth_vis, (cx, cy), r, 255, 5)
    

        output = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        output_depth = self.bridge.cv2_to_imgmsg(depth_vis)

        output.header.stamp = msg.header.stamp
        output.header.frame_id = msg.header.frame_id

        output_depth.header.stamp = msg.header.stamp
        output_depth.header.frame_id = msg.header.frame_id

        circle_distance = self.depth_frame[cy, cx]

        u = cx
        v = cy
        z = circle_distance

        x = (u - self.cx0) * z / self.fx
        y = (v - self.cy0) * z / self.fy

        self.get_logger().info(
            f"Camera: ({x:.3f}, {y:.3f}, {z:.3f})"
        )

        point.header.stamp = msg.header.stamp
        point.header.frame_id = "camera_color_optical_frame"

        point.point.x = x
        point.point.y = y
        point.point.z = z

        point_base = self.tf_buffer.transform(point, "base_link")

        

        self.get_logger().info(
            f"Base: ({point_base.point.x:.3f}, "
            f"{point_base.point.y:.3f}, "
            f"{point_base.point.z:.3f})"
        )

        

        self.camera_arm_pub_.publish(output)
        self.depth_camera_pub_.publish(output_depth)
        

def main():
    rclpy.init()
    node = x3plusCircleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        