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
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf2_ros.transform_listener import TransformListener


class PlaneSegmentationNode(Node):

    def __init__(self):
        super().__init__('plane_segmentation_node')

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------

        self.declare_parameter('distance_threshold', 0.01)
        self.declare_parameter('ransac_n', 3)
        self.declare_parameter('num_iterations', 3000)

        # Minimum number of points required for a valid plane
        self.declare_parameter('min_points', 50)

        # Keep only points within this distance from the sensor
        # / point-cloud origin
        self.declare_parameter('max_distance', 0.60)

        self.declare_parameter('voxel', 0.005)

        # We want the three surfaces in your current scene
        self.declare_parameter('num_planes', 8)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------------------------------------------------
        # Subscribers / Publishers
        # ---------------------------------------------------------

        self.pc2_sub_ = self.create_subscription(
            PointCloud2,
            '/points2',
            self.pointcloud_callback,
            10
        )

        self.plane_coeff_pub_ = self.create_publisher(
            Float32MultiArray,
            '/dominant_plane_coeffiecients',
            10
        )

        self.plane_vis_pub_ = self.create_publisher(
            PointCloud2,
            '/dominant_plane_vis',
            10
        )

        self.get_logger().info(
            "Multi-plane Segmentation with Visualizer initialized"
        )

    # =============================================================
    # Iterative plane segmentation
    # =============================================================

    def segment_planes(
        self,
        point_cloud,
        distance_threshold,
        ransac_n,
        num_iterations,
        min_points,
        num_planes
    ):

        significant_planes = []

        remaining_cloud = point_cloud

        for plane_index in range(num_planes):

            if len(remaining_cloud.points) < ransac_n:

                self.get_logger().info(
                    "Not enough points remaining."
                )

                break

            # -----------------------------------------------------
            # RANSAC
            # -----------------------------------------------------

            plane_model, inliers = (
                remaining_cloud.segment_plane(
                    distance_threshold=distance_threshold,
                    ransac_n=ransac_n,
                    num_iterations=num_iterations
                )
            )

            # -----------------------------------------------------
            # Check whether this is a significant plane
            # -----------------------------------------------------

            if len(inliers) < min_points:

                self.get_logger().info(
                    f"Stopping: plane {plane_index + 1} "
                    f"has only {len(inliers)} points."
                )

                break

            # -----------------------------------------------------
            # Extract plane points
            # -----------------------------------------------------

            inlier_point_cloud = (
                remaining_cloud.select_by_index(inliers)
            )

            # Give each plane a random color
            inlier_point_cloud.paint_uniform_color(
                np.random.rand(3)
            )

            # Store everything we need later
            significant_planes.append({
                "coefficients": np.asarray(
                    plane_model,
                    dtype=np.float64
                ),

                "points": np.asarray(
                    inlier_point_cloud.points
                ),

                "point_cloud": inlier_point_cloud
            })

            A, B, C, D = plane_model

            self.get_logger().info(
                f"Plane {plane_index + 1}: "
                f"{len(inliers)} points"
            )

            self.get_logger().info(
                f"  [{A:.6f}, "
                f"{B:.6f}, "
                f"{C:.6f}, "
                f"{D:.6f}]"
            )

            # -----------------------------------------------------
            # Remove this plane before finding the next one
            # -----------------------------------------------------

            remaining_cloud = (
                remaining_cloud.select_by_index(
                    inliers,
                    invert=True
                )
            )

        return significant_planes, remaining_cloud

    # =============================================================
    # Visualization
    # =============================================================

    def plane_visualizer(
        self,
        points_downsampled,
        planes_list,
        input_header
    ):

        num_points = len(points_downsampled)

        # ---------------------------------------------------------
        # Default colour: grey
        # ---------------------------------------------------------

        grey_packed = struct.unpack(
            'I',
            struct.pack(
                'BBBB',
                150,
                150,
                155,
                255
            )
        )[0]

        data_type = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ]

        packed_points = np.zeros(
            num_points,
            dtype=data_type
        )

        packed_points['x'] = points_downsampled[:, 0]
        packed_points['y'] = points_downsampled[:, 1]
        packed_points['z'] = points_downsampled[:, 2]

        packed_points['rgb'] = grey_packed

        # ---------------------------------------------------------
        # Colour each segmented plane
        # ---------------------------------------------------------

        void_dt = np.dtype(
            (
                np.void,
                points_downsampled.dtype.itemsize *
                points_downsampled.shape[1]
            )
        )

        downsampled_view = (
            points_downsampled
            .view(void_dt)
            .ravel()
        )

        for plane in planes_list:

            plane_pts = plane["points"]

            if len(plane_pts) == 0:
                continue

            o3d_color = np.asarray(
                plane["point_cloud"].colors
            )[0]

            r = int(o3d_color[0] * 255)
            g = int(o3d_color[1] * 255)
            b = int(o3d_color[2] * 255)

            packed_color = struct.unpack(
                'I',
                struct.pack(
                    'BBBB',
                    b,
                    g,
                    r,
                    255
                )
            )[0]

            plane_view = (
                plane_pts
                .view(void_dt)
                .ravel()
            )

            inliers_mask = np.isin(
                downsampled_view,
                plane_view
            )

            packed_points['rgb'][inliers_mask] = (
                packed_color
            )

        # ---------------------------------------------------------
        # ROS PointCloud2 fields
        # ---------------------------------------------------------

        fields = [
            PointField(
                name='x',
                offset=0,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='y',
                offset=4,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='z',
                offset=8,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='rgb',
                offset=12,
                datatype=PointField.UINT32,
                count=1
            )
        ]

        vis_header = Header()

        # Keep original timestamp
        vis_header.stamp = input_header.stamp

        vis_header.frame_id = input_header.frame_id

        vis_cloud_msg = pc2.create_cloud(
            vis_header,
            fields,
            packed_points
        )

        self.plane_vis_pub_.publish(
            vis_cloud_msg
        )

    # =============================================================
    # Point cloud callback
    # =============================================================

    def pointcloud_callback(self, msg):
        '''
        target_frame = "base_link"

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                msg.header.frame_id,
                rclpy.time.Time()
            )
            msg = do_transform_cloud(msg, transform)

            self.get_logger().info(
                f"Transformed pointcloud from "
                f"{transform.child_frame_id} to {target_frame}"
            )

        except TransformException as ex:
            self.get_logger().warning(f"Waiting for Transform: {ex}")
            return
        '''
        # ---------------------------------------------------------
        # Read point cloud
        # ---------------------------------------------------------

        gen = pc2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True
        )

        structured_points = np.array(
            list(gen)
        )

        if len(structured_points) < 3:

            self.get_logger().warning(
                "Not enough points."
            )

            return

        points = np.zeros(
            (len(structured_points), 3),
            dtype=np.float32
        )

        points[:, 0] = structured_points['x']
        points[:, 1] = structured_points['y']
        points[:, 2] = structured_points['z']

        # =========================================================
        # 60 cm distance filter
        # =========================================================

        max_distance = (
            self.get_parameter(
                'max_distance'
            )
            .get_parameter_value()
            .double_value
        )

        distances = np.linalg.norm(
            points,
            axis=1
        )

        mask = distances <= max_distance

        points = points[mask]

        self.get_logger().info(
            f"Points within "
            f"{max_distance:.2f} m: "
            f"{len(points)}"
        )

        if len(points) < 3:
            return

        # =========================================================
        # Open3D cloud
        # =========================================================

        pcd = o3d.geometry.PointCloud()

        pcd.points = (
            o3d.utility.Vector3dVector(
                points.astype(np.float64)
            )
        )

        self.get_logger().info(
            f"Filtered point cloud: "
            f"{len(pcd.points)} points"
        )

        # =========================================================
        # Downsample
        # =========================================================

        voxel = (
            self.get_parameter(
                'voxel'
            )
            .get_parameter_value()
            .double_value
        )

        if voxel > 0.0:

            pcd = pcd.voxel_down_sample(
                voxel_size=voxel
            )

        points_downsampled = np.asarray(
            pcd.points
        )

        self.get_logger().info(
            f"After voxel downsampling: "
            f"{len(points_downsampled)} points"
        )

        # =========================================================
        # Parameters
        # =========================================================

        dist_thresh = (
            self.get_parameter(
                'distance_threshold'
            )
            .get_parameter_value()
            .double_value
        )

        ransac_n = (
            self.get_parameter(
                'ransac_n'
            )
            .get_parameter_value()
            .integer_value
        )

        num_iter = (
            self.get_parameter(
                'num_iterations'
            )
            .get_parameter_value()
            .integer_value
        )

        min_points = (
            self.get_parameter(
                'min_points'
            )
            .get_parameter_value()
            .integer_value
        )

        num_planes = (
            self.get_parameter(
                'num_planes'
            )
            .get_parameter_value()
            .integer_value
        )

        # =========================================================
        # Segment the three planes
        # =========================================================

        planes, outliers = self.segment_planes(
            point_cloud=pcd,
            distance_threshold=dist_thresh,
            ransac_n=ransac_n,
            num_iterations=num_iter,
            min_points=min_points,
            num_planes=num_planes
        )

        self.get_logger().info(
            f"Total planes detected: {len(planes)}"
        )

        # =========================================================
        # SAVE DATA FOR DRAKE
        # =========================================================

        save_path = (
            "/home/jetson/segmented_scene.npz"
        )

        save_data = {
            "points_downsampled":
                points_downsampled
        }

        for i, plane in enumerate(planes):

            save_data[
                f"plane_{i}_coefficients"
            ] = plane["coefficients"]

            save_data[
                f"plane_{i}_points"
            ] = plane["points"]

        # Also save the remaining non-plane points
        save_data[
            "remaining_points"
        ] = np.asarray(
            outliers.points
        )

        np.savez(
            save_path,
            **save_data
        )

        self.get_logger().info(
            f"Saved scene data to:"
            f"\n  {save_path}"
        )

        # =========================================================
        # Visualize
        # =========================================================

        self.plane_visualizer(
            points_downsampled,
            planes,
            msg.header
        )

        # =========================================================
        # Publish first plane for compatibility
        # =========================================================

        if len(planes) > 0:

            plane_coeff_msg = (
                Float32MultiArray()
            )

            plane_coeff_msg.data = [
                float(x)
                for x in planes[0]["coefficients"]
            ]

            self.plane_coeff_pub_.publish(
                plane_coeff_msg
            )


def main():

    rclpy.init()

    node = PlaneSegmentationNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()