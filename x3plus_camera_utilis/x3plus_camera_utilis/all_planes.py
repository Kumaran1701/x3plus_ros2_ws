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

        self.declare_parameter('distance_threshold', 0.01)
        self.declare_parameter('ransac_n', 3)
        self.declare_parameter('num_iterations', 1000)
        self.declare_parameter('min_points', 400)
        self.declare_parameter('voxel', 0.1)
        self.declare_parameter('num_planes', 3)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pc2_sub_ = self.create_subscription(PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10)
        self.plane_coeff_pub_ = self.create_publisher(Float32MultiArray, '/dominant_plane_coeffiecients', 10)
        self.plane_vis_pub_ = self.create_publisher(PointCloud2, '/dominant_plane_vis', 10)

        self.get_logger().info("Multi-plane Segmentation with Visualizer initialized")

    def segment_planes(self, point_cloud, distance_threshold, ransac_n, num_iterations, min_points):
        significant_planes = []  # List to store significant planes
        remaining_cloud = point_cloud
        plane_index = 1

        while True:
            # Segment the largest plane
            plane_model, inliers = remaining_cloud.segment_plane(distance_threshold=distance_threshold,
                                                                ransac_n=ransac_n,
                                                                num_iterations=num_iterations)
            # Count number of inliers to determine the significance of the plane
            if len(inliers) < min_points:
                break

            print(f"Plane {plane_index} equation {plane_model}")
            plane_index += 1

            # Extract Inliers from the plane
            inlier_point_cloud = remaining_cloud.select_by_index(inliers)
            inlier_point_cloud.paint_uniform_color(np.random.rand(3))
            significant_planes.append(inlier_point_cloud)  # add the significant plane to the list of significant planes

            # Extract remaining point cloud after inlier extraction
            remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)

        return significant_planes, remaining_cloud

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
            '''
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
            '''
            
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
    
            max_distance = 0.80
    
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

            voxel = self.get_parameter('voxel').get_parameter_value().double_value
            # 4. FIX: Use underscore format required by Open3D 0.17.0
            pcd = pcd.voxel_down_sample(voxel_size=voxel)
            points_downsampled = np.asarray(pcd.points)
    
            dist_thresh = self.get_parameter('distance_threshold').get_parameter_value().double_value
            ransac_n = self.get_parameter('ransac_n').get_parameter_value().integer_value
            num_iter = self.get_parameter('num_iterations').get_parameter_value().integer_value
            min_points = self.get_parameter('min_points').get_parameter_value().integer_value
            voxel = self.get_parameter('voxel').get_parameter_value().double_value

            planes, outliers = self.segment_planes(point_cloud=pcd, 
                                                   distance_threshold=dist_thresh,
                                                   ransac_n=ransac_n,
                                                   num_iterations=num_iter,
                                                   min_points=min_points)
            self.get_logger().info(f"Total Planes: {len(planes)}")
            plane_points = []
            for i in range(len(planes)):
                new_points = np.asarray(planes[i].points)
                plane_points.append(new_points)

            self.plane_visualizer(
                        points_downsampled,
                        plane_points,
                        msg.header.frame_id
                    )
            self.get_logger().info("Visualizing Planes")

def main():
    rclpy.init()
    node = PlaneSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()