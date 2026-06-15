#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from visualization_msgs.msg import Marker

import cv2
from cv_bridge import CvBridge

import numpy as np
import time

class x3plusCircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector_node')

        self.depth_frame_flag = False

        self.fx = 543.2409
        self.fy = 543.2409
        self.cx0 = 323.0906
        self.cy0 = 241.3283

        self.bridge = CvBridge()

        self.marker_pub = self.create_publisher(
            Marker,
            '/circle_marker',
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        time.sleep(5)
        
        self.camera_arm_pub_ = self.create_publisher(Image, '/camera_circle/image_circle', 10)
        self.depth_camera_pub_ = self.create_publisher(Image, '/camera_circle/depth_image', 10)
        self.depth_camera_sub_ = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.camera_arm_sub_ = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        

    def depth_callback(self, msg):
        # print("Depth Image received")
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_frame_flag = True

    def image_callback(self, msg):

        # print("Image_callback started")

        if not self.depth_frame_flag:
            print("depth_frame_flag: ", self.depth_frame_flag)
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
            minRadius=30,
            maxRadius=40
        )

        #print(circles)

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
            self.get_logger().info(f"cx: {cx}, radius: {r}")
        else:
            # print("No valid circle")
            return
    
        
        output = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        output_depth = self.bridge.cv2_to_imgmsg(depth_vis)

        output.header.stamp = msg.header.stamp
        output.header.frame_id = msg.header.frame_id

        output_depth.header.stamp = msg.header.stamp
        output_depth.header.frame_id = msg.header.frame_id

        

        u = cx
        v = cy
        roi = self.depth_frame[cy-2:cy+3, cx-2:cx+3]
        z = (np.median(roi))
        z = z /1000.0
        # print("max roi:", z)

        # print(self.depth_frame.dtype)
        # print(self.depth_frame[cy, cx])
        # print(self.depth_frame[cy-2:cy+3, cx-2:cx+3])
        #print("max depth:", np.max(self.depth_frame))
        x = (u - self.cx0) * z / self.fx
        y = (v - self.cy0) * z / self.fy
        self.get_logger().info(f"distance: {z}")
        self.get_logger().info(
            f"Camera: ({x:.3f}, {y:.3f}, {z:.3f})"
        )
        #'''

        point.header.stamp = msg.header.stamp
        point.header.frame_id = "camera_color_optical_frame"

        point.point.x = x
        point.point.y = y
        point.point.z = z

        point_base = self.tf_buffer.transform(point, "base_link")

        marker = Marker()

        marker.header.frame_id = "base_link"
        marker.header.stamp = msg.header.stamp

        marker.ns = "circle_wrt_base_link"
        marker.id = 0

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = point_base.point.x
        marker.pose.position.y = point_base.point.y
        marker.pose.position.z = point_base.point.z

        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.get_logger().info(f"Image stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        
        tf_arm_link1 = self.tf_buffer.lookup_transform(
            "arm_link1",
            point.header.frame_id,
            Time()
        )

        self.get_logger().info(f"tf_arm_link1 stamp: {tf_arm_link1.header.stamp.sec}.{tf_arm_link1.header.stamp.nanosec}")

        point_arm_link1 = tf2_geometry_msgs.do_transform_point(point, tf_arm_link1)

        marker_arm_link1 = Marker()

        marker_arm_link1.header.frame_id = "arm_link1"
        marker_arm_link1.header.stamp = msg.header.stamp

        marker_arm_link1.ns = "circle_wrt_arm_link1"
        marker_arm_link1.id = 1

        marker_arm_link1.type = Marker.SPHERE
        marker_arm_link1.action = Marker.ADD

        marker_arm_link1.pose.position.x = point_arm_link1.point.x
        marker_arm_link1.pose.position.y = point_arm_link1.point.y
        marker_arm_link1.pose.position.z = point_arm_link1.point.z

        marker_arm_link1.pose.orientation.w = 1.0

        marker_arm_link1.scale.x = 0.06
        marker_arm_link1.scale.y = 0.06
        marker_arm_link1.scale.z = 0.06

        marker_arm_link1.color.a = 1.0
        marker_arm_link1.color.r = 0.0
        marker_arm_link1.color.g = 1.0
        marker_arm_link1.color.b = 0.0

        self.marker_pub.publish(marker)
        self.marker_pub.publish(marker_arm_link1)

    
        self.get_logger().info(
            f"Base: ({point_base.point.x:.3f}, "
            f"{point_base.point.y:.3f}, "
            f"{point_base.point.z:.3f})"
        )

        self.get_logger().info(
            f"Base: ({point_arm_link1.point.x:.3f}, "
            f"{point_arm_link1.point.y:.3f}, "
            f"{point_arm_link1.point.z:.3f})"
        )
        #'''
        

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

"""
[INFO] [1781188283.609719093] [circle_detector_node]: cx: 338, radius: 36
max roi: 0.525
uint16
525
[[525 525 525 525 525]
 [525 525 525 525 525]
 [525 525 525 525 525]
 [525 525 525 525 525]
 [525 525 525 525 525]]
[INFO] [1781188283.612478313] [circle_detector_node]: distance: 0.525
[INFO] [1781188283.613345837] [circle_detector_node]: Camera: (0.014, 0.109, 0.525)
[INFO] [1781188283.615616332] [circle_detector_node]: Base: (0.482, -0.037, 0.305)

"""