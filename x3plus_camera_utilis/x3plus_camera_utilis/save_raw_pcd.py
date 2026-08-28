#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import open3d as o3d
import numpy as np
import os

from sensor_msgs.msg import PointCloud2
# FIX: Use read_points_numpy for significantly faster, bug-free array conversion
from sensor_msgs_py.point_cloud2 import read_points_numpy

class PlaneSegmentationNode(Node):
    def __init__(self):
        super().__init__('plane_segmentation_node')

        self.declare_parameter('total_saves_count', 10)
        self.current_saves_count = 0
        self.output_dir = "raw_pcd_sequence"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Subscription to the topic
        self.pc2_sub_ = self.create_subscription(
            PointCloud2, 
            '/points2', 
            self.pointcloud_callback, 
            10
        )

        self.get_logger().info("Raw Pointcloud saving node started")

    def pointcloud_callback(self, msg: PointCloud2):
        total_saves_count = self.get_parameter('total_saves_count').get_parameter_value().integer_value
        
        # Stop processing once we reach the total saves count
        if self.current_saves_count >= total_saves_count:
            return

        # 1. FIX: Read points directly into a flat (N, 3) float32 numpy array
        # This completely replaces the slow list(gen) and manual field mapping
        points = read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        if points.shape[0] < 3:
            self.get_logger().warn("Not enough points in the incoming message")
            return

        # 2. Filter by max distance (80 cm)
        max_distance = 0.80
        distances = np.linalg.norm(points, axis=1)
        mask = distances <= max_distance
        points = points[mask]

        self.get_logger().info(
            f"Points within {max_distance:.2f} m: {len(points)}"
        )

        # Check if enough points remain after slicing
        if len(points) < 3:
            # FIX: Updated log string to match the 80 cm max_distance variable
            self.get_logger().info(f"Not enough points within {int(max_distance*100)} cm.")
            return

        # 3. Create Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        # Open3D expects float64 vectors
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        
        # Generate sequential filename
        filename = os.path.join(self.output_dir, f"frame_{str(self.current_saves_count).zfill(5)}.pcd")
        
        # Write as a compressed binary .pcd for speed and storage efficiency
        o3d.io.write_point_cloud(filename, pcd, write_ascii=False, compressed=True)
        
        self.current_saves_count += 1
        
        if self.current_saves_count == total_saves_count:
            self.get_logger().info(f"Successfully saved all {total_saves_count} frames. Ready to shutdown.")

def main(args=None):
    rclpy.init(args=args)
    node = PlaneSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
