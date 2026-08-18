#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import open3d as o3d
import numpy as np
import pyoctomap


class OctoMapBuilder(Node):

    def __init__(self):
        super().__init__('octomap_builder')

        # -----------------------------
        # Path to Open3D example point cloud
        # -----------------------------
        ply_file = o3d.data.PLYPointCloud()

        self.get_logger().info(f"Loading point cloud:\n{ply_file}")

        # -----------------------------
        # Load point cloud
        # -----------------------------
        pcd = o3d.io.read_point_cloud(ply_file.path)

        # -----------------------------
        # Optional preprocessing
        # -----------------------------
        pcd = pcd.voxel_down_sample(voxel_size=0.03)

        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )

        points = np.asarray(pcd.points)

        self.get_logger().info(
            f"Loaded {len(points)} filtered points."
        )

        # -----------------------------
        # Create OctoMap
        # -----------------------------
        resolution = 0.05  # 5 cm voxels

        tree = pyoctomap.OcTree(resolution)

        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        octo_pc = pyoctomap.Pointcloud()
        octo_pc.push_back(points)
        
        # Update the map
        tree.insertPointCloud(octo_pc, sensor_origin)
        output_file = "living_room.bt"
        # 5. Save the resulting map to disk
        print(f"Saving OctoMap to: {output_file}")
        tree.writeBinary(output_file)
        print("Done!")



def main(args=None):
    rclpy.init(args=args)

    node = OctoMapBuilder()

    rclpy.spin(node)


if __name__ == '__main__':
    main()