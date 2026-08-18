#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import open3d as o3d
import numpy as np
import struct

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, Header
import sensor_msgs_py.point_cloud2 as pc2

class PlaneSegmentationNode(Node):
    def __init__(self):
        super().__init__('plane_segmentation_node')

        self.declare_parameter('distance_threshold', 0.02)
        self.declare_parameter('ransac_n', 3)
        self.declare_parameter('num_iterations', 200)

        print("Success")

        self.pc2_sub_ = self.create_subscription(PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10)
        self.plane_coeff_pub_ = self.create_publisher(Float32MultiArray, '/dominant_plane_coeffiecients', 10)
        self.plane_vis_pub_ = self.create_publisher(PointCloud2, '/dominant_plane_vis', 10)

        self.get_logger().info("Plane Segmentation with Visualizer initialized")

    def plane_visualizer(self, points_downsampled, inliers, frame_id):
        num_points = len(points_downsampled)

        red_packed = struct.unpack('I', struct.pack('BBBB', 0, 0, 255, 255))[0]
        blue_packed = struct.unpack('I', struct.pack('BBBB', 255, 0, 0, 255))[0]

        packed_colors = np.full((num_points, 1), blue_packed, dtype=np.uint32)

        packed_colors[inliers] = red_packed

        packed_points = np.hstack([points_downsampled, packed_colors.view(np.float32)])

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]

        vis_header = Header()
        vis_header.stamp = self.get_clock().now().to_msg()
        vis_header.frame_id = frame_id

        vis_cloud_msg = pc2.create_cloud(vis_header, fields, packed_points)
        self.plane_vis_pub_.publish(vis_cloud_msg)


    def pointcloud_callback(self, msg: PointCloud2):
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(gen))
        self.get_logger().info(
            f"Received {len(points)} points, "
            f"shape={points.shape}, dtype={points.dtype}"
        )
        if len(points) < 3:
            print("Not enough points")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        self.get_logger().info(f"Total pcd points: {len(pcd.points)}")

        pcd = pcd.voxel_downsample(voxel_size=0.01)
        points_downsampled = np.asarray(pcd.points)

        dist_thresh = self.get_parameter('distance_threshold').get_parameter_value().double_value
        ransac_n = self.get_parameter('ransac_n').get_parameter_value().integer_value
        num_iter = self.get_parameter('num_iterations').get_parameter_value().integer_value

        plane_model, inliers = pcd.segment_plane(
                distance_threshold=dist_thresh,
                ransac_n=ransac_n,
                num_iterations=num_iter
            )

        [A, B, C, D] = plane_model
        self.get_logger().info(f"Plane Model: A:{A}, B:{B}, C:{C}, D:{D}")
        if D > 0:
            A, B, C, D = -A, -B, -C, -D
        
        plane_coeff_msg = Float32MultiArray()
        plane_coeff_msg.data = [float(A), float(B), float(C), float(D)]
        self.plane_coeff_pub_.publish(plane_coeff_msg)
        frame_id = msg.header.frame_id
        print("Publishing Plane Coefficients")

        self.plane_visualizer(points_downsampled, inliers, frame_id)

def main():
    rclpy.init()
    node = PlaneSegmentationNode()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()