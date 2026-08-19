#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import open3d as o3d
import numpy as np
import struct

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, Header
import sensor_msgs_py.point_cloud2 as pc2

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

class PlaneSegmentationNode(Node):
    def __init__(self):
        super().__init__('plane_segmentation_node')

        self.declare_parameter('distance_threshold', 0.02)
        self.declare_parameter('ransac_n', 3)
        self.declare_parameter('num_iterations', 200)

        self.declare_parameter('num_planes', 3)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pc2_sub_ = self.create_subscription(PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10)
        self.plane_coeff_pub_ = self.create_publisher(Float32MultiArray, '/dominant_plane_coeffiecients', 10)
        self.plane_vis_pub_ = self.create_publisher(PointCloud2, '/dominant_plane_vis', 10)

        self.get_logger().info("Multi-plane Segmentation with Visualizer initialized")

    def plane_visualizer(self, points_downsampled, inliers, frame_id):
        num_points = len(points_downsampled)

        # 1. Define colors using standard RGB integers
        # Red for the dominant plane, Blue for everything else
        red_packed = struct.unpack('I', struct.pack('BBBB', 0, 0, 255, 255))[0]
        blue_packed = struct.unpack('I', struct.pack('BBBB', 255, 0, 0, 255))[0]

        # 2. CREATE STRUCTURED ARRAY: Maps distinct types to a clean byte layout
        data_type = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)]
        packed_points = np.zeros(num_points, dtype=data_type)

        # 3. Populate coordinates
        packed_points['x'] = points_downsampled[:, 0]
        packed_points['y'] = points_downsampled[:, 1]
        packed_points['z'] = points_downsampled[:, 2]

        # 4. Colorize points based on RANSAC inliers
        packed_points['rgb'] = blue_packed
        if len(inliers) > 0:
            packed_points['rgb'][inliers] = red_packed

        # 5. Build individual field maps explicitly
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]

        vis_header = Header()
        vis_header.stamp = self.get_clock().now().to_msg()
        vis_header.frame_id = frame_id

        # 6. Publish the compliant cloud message
        vis_cloud_msg = pc2.create_cloud(vis_header, fields, packed_points)
        self.plane_vis_pub_.publish(vis_cloud_msg)



    def pointcloud_callback(self, msg: PointCloud2):
        target_frame = 'base_link'

        try: 
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                msg.header.frame_id,
                rclpy.time.Time()
            )
            msg = do_transform_cloud(msg, transform)
            frame_id = target_frame
        except TransformException as ex:
            self.get_logger().info(f"Waiting for TF frame transform: {ex}")
            return
        
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        structured_points = np.array(list(gen))
        
        if len(structured_points) < 3:
            print("Not enough points")
            return

        # 2. FIX: Convert structured tuple array to standard flat 2D float array (N, 3)
        points = np.zeros((len(structured_points), 3), dtype=np.float32)
        points[:, 0] = structured_points['x']
        points[:, 1] = structured_points['y']
        points[:, 2] = structured_points['z']

        max_distance = 0.60

        distances = np.linalg.norm(points, axis=1)

        mask = distances <= max_distance

        points = points[mask]

        self.get_logger().info(
            f"Points within {max_distance:.2f} m: {len(points)}"
        )

        if len(points) < 3:
            self.get_logger().info("Not enough points within 60 cm.")
            return

        # 3. Create Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        self.get_logger().info(f"Total pcd points: {len(pcd.points)}")

        # 4. FIX: Use underscore format required by Open3D 0.17.0
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        points_downsampled = np.asarray(pcd.points)

        dist_thresh = self.get_parameter('distance_threshold').get_parameter_value().double_value
        ransac_n = self.get_parameter('ransac_n').get_parameter_value().integer_value
        num_iter = self.get_parameter('num_iterations').get_parameter_value().integer_value

        num_planes = (self.get_parameter('num_planes').get_parameter_value().integer_value)

        remaining_pcd = pcd

        detected_planes = []

        for plane_index in range(num_planes):

            if len(remaining_pcd.points) < ransac_n:

                self.get_logger().warning(
                    "Not enough remaining points "
                    "for another plane."
                )

                break

            plane_model, inliers = (
                remaining_pcd.segment_plane(
                    distance_threshold=dist_thresh,
                    ransac_n=ransac_n,
                    num_iterations=num_iter
                )
            )

            if len(inliers) < ransac_n:

                self.get_logger().warning(
                    f"Plane {plane_index + 1}: "
                    "not enough inliers."
                )

                break

            plane_model = np.asarray(
                plane_model,
                dtype=np.float64
            )

            plane_points = np.asarray(
                remaining_pcd.points
            )[inliers]

            # -------------------------------------------------
            # Store plane
            # -------------------------------------------------

            detected_planes.append({
                "coefficients": plane_model,
                "points": plane_points
            })

            A, B, C, D = plane_model

            self.get_logger().info(
                f"Plane {plane_index + 1}: "
                f"{len(plane_points)} inliers"
            )

            self.get_logger().info(
                f"  coefficients: "
                f"[{A:.6f}, "
                f"{B:.6f}, "
                f"{C:.6f}, "
                f"{D:.6f}]"
            )

            # -------------------------------------------------
            # Remove plane from remaining cloud
            # -------------------------------------------------

            remaining_pcd = (
                remaining_pcd.select_by_index(
                    inliers,
                    invert=True
                )
            )
        if len(detected_planes) == 0:

            self.get_logger().warning(
                "No planes detected."
            )

            return

        save_path = (
            "/home/jetson/segmented_scene.npz"
        )

        save_data = {
            "points_downsampled": points_downsampled
        }

        for i, plane in enumerate(detected_planes):

            save_data[
                f"plane_{i}_coefficients"
            ] = plane["coefficients"]

            save_data[
                f"plane_{i}_points"
            ] = plane["points"]

        np.savez(
            save_path,
            **save_data
        )

        self.get_logger().info(
            f"Saved segmented scene to: "
            f"{save_path}"
        )

        all_plane_points = np.vstack([
            plane["points"]
            for plane in detected_planes
        ])

        all_plane_inliers = []

        for plane_point in all_plane_points:

            distances = np.linalg.norm(
                points_downsampled - plane_point,
                axis=1
            )

            index = np.argmin(distances)

            all_plane_inliers.append(index)

        all_plane_inliers = np.unique(
            all_plane_inliers
        )

        self.plane_visualizer(
            points_downsampled,
            all_plane_inliers,
            frame_id
        )

        first_plane = detected_planes[0]["coefficients"]

        plane_coeff_msg = Float32MultiArray()

        plane_coeff_msg.data = [
            float(x)
            for x in first_plane
        ]

        self.plane_coeff_pub_.publish(
            plane_coeff_msg
        )

        self.get_logger().info(
            f"Detected {len(detected_planes)} planes."
        )


def main():
    rclpy.init()
    node = PlaneSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()