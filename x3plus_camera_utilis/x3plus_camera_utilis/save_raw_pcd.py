#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import json
import os

# ROS 2 Core Message Types
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# ROS 2 Time Synchronization and TF Tracking
from message_filters import Subscriber, TimeSynchronizer
from tf2_ros import TransformException, Buffer, TransformListener
import tf2_geometry_msgs

class TSDFDataRecorder(Node):
    def __init__(self):
        super().__init__('tsdf_data_recorder')

        # 1. Configurable Parameters
        self.declare_parameter('total_frames_to_save', 50)
        self.declare_parameter('world_frame', 'odom')           # Change to 'base_link' if odom isn't used
        self.declare_parameter('camera_frame', 'camera_link')   # Adjust to match your specific camera tf frame
        self.declare_parameter('output_directory', 'tsdf_dataset')

        self.total_frames = self.get_parameter('total_frames_to_save').value
        self.world_frame = self.get_parameter('world_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.output_dir = self.get_parameter('output_directory').value
        
        self.frame_count = 0
        self.bridge = CvBridge()

        # Create clean structured storage folders
        self.depth_dir = os.path.join(self.output_dir, 'depth')
        self.rgb_dir = os.path.join(self.output_dir, 'rgb')
        self.pose_dir = os.path.join(self.output_dir, 'pose')
        self.intrinsics_dir = os.path.join(self.output_dir, 'intrinsics')
        for folder in [self.depth_dir, self.rgb_dir, self.pose_dir, self.intrinsics_dir]:
            os.makedirs(folder, exist_ok=True)

        # 2. Setup TF Listener to capture robot/camera extrinsic tracking
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 3. Synchronize Subscriptions (RGB-to-Depth matched tracking)
        # We use exact time synchronization across the three primary streams
        self.sub_depth = Subscriber(self, Image, '/depth/image_raw')
        self.sub_rgb = Subscriber(self, Image, '/rgb_to_depth/image_raw')
        self.sub_info = Subscriber(self, CameraInfo, '/rgb_to_depth/camera_info')

        # Buffer queue size of 10 to manage Jetson I/O latency spikes
        self.ts = TimeSynchronizer([self.sub_depth, self.sub_rgb, self.sub_info], queue_size=10)
        self.ts.registerCallback(self.synchronized_callback)

        self.get_logger().info(f"TSDF Dataset Recorder started. Ready to capture {self.total_frames} matched frames.")

    def synchronized_callback(self, depth_msg, rgb_msg, info_msg):
        if self.frame_count >= self.total_frames:
            return

        # Use the exact timestamp of the depth sensor capture for all operations
        timestamp = depth_msg.header.stamp
        timestamp_str = f"{timestamp.sec}_{timestamp.nanosec}"
        frame_idx = str(self.frame_count).zfill(5)

        # ----------------------------------------------------
        # STEP 1: Capture 4x4 Extrinsic Camera Pose Matrix from TF
        # ----------------------------------------------------
        try:
            # Lookup where the camera was in the world EXACTLY at the time of frame capture
            tf_transform = self.tf_buffer.lookup_transform(
                self.world_frame, 
                self.camera_frame, 
                timestamp,
                rclpy.duration.Duration(seconds=0.1) # 100ms tolerance window
            )
            
            # Extract Translation vectors
            t = tf_transform.transform.translation
            # Extract Rotation Quaternions
            q = tf_transform.transform.rotation

            # Build standard 4x4 homogenous transformation matrix (Open3D TSDF standard input)
            pose_matrix = np.eye(4, dtype=np.float64)
            
            # Populate rotation from quaternion equations
            r = self.quaternion_to_matrix(q.x, q.y, q.z, q.w)
            pose_matrix[0:3, 0:3] = r
            pose_matrix[0:3, 3] = [t.x, t.y, t.z]

            # Save Pose matrix directly as a standard txt file
            pose_path = os.path.join(self.pose_dir, f"frame_{frame_idx}.txt")
            np.savetxt(pose_path, pose_matrix, fmt='%.8f')

        except TransformException as ex:
            self.get_logger().warn(f"Could not resolve TF pose alignment at {timestamp_str}: {ex}")
            return  # Skip frame if mapping transforms are broken or lagging

        # ----------------------------------------------------
        # STEP 2: Process & Save Raw 16-bit Depth Map Image
        # ----------------------------------------------------
        try:
            # Convert ROS Image to raw OpenCV Mat (16UC1 type - Unsigned 16-bit short millimeters)
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            depth_path = os.path.join(self.depth_dir, f"frame_{frame_idx}.png")
            # Save raw pixel data without losing 16-bit compression scales
            cv2.imwrite(depth_path, cv_depth)
        except Exception as e:
            self.get_logger().error(f"Depth compression pipeline error: {e}")
            return

        # ----------------------------------------------------
        # STEP 3: Process & Save Synchronized RGB Frame
        # ----------------------------------------------------
        try:
            # Convert ROS Image to classic standard BGR OpenCV array
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            rgb_path = os.path.join(self.rgb_dir, f"frame_{frame_idx}.png")
            cv2.imwrite(rgb_path, cv_rgb)

        except Exception as e:
            self.get_logger().error(f"RGB has visual compilation error: {e}")
            return


        # ----------------------------------------------------
        # STEP 4: Capture & Save Digital Camera Intrinsics
        # ----------------------------------------------------
        # Read the internal optical projection matrix parameters directly from topic header fields
        intrinsics_data = {
            "timestamp": {"secs": timestamp.sec, "nsecs": timestamp.nanosec},
            "width": info_msg.width,
            "height": info_msg.height,
            "fx": info_msg.k[0],   # Focal Length X
            "fy": info_msg.k[4],   # Focal Length Y
            "cx": info_msg.k[2],   # Principal Center Point X
            "cy": info_msg.k[5],   # Principal Center Point Y
            "depth_scale_factor": 1000.0 # Standard ROS depth maps represent 1 meter = 1000 pixels integer value
        }
        
        intrinsics_path = os.path.join(self.intrinsics_dir, f"frame_{frame_idx}.json")
        with open(intrinsics_path, 'w') as f:
            json.dump(intrinsics_data, f, indent=4)

        # Increment frame count logging updates
        self.get_logger().info(f"Stored Synchronized Triple Setup Sequence: [Index ID: {frame_idx}]")
        self.frame_count += 1

        if self.frame_count == self.total_frames:
            self.get_logger().info("\n=== Target Dataset Completed Successfully! Node idle ===")

    def quaternion_to_matrix(self, x, y, z, w):
        """Helper to convert standard spatial orientation quaternion strings into 3x3 rotational frameworks."""
        sqw = w*w; sqx = x*x; sqy = y*y; sqz = z*z
        invs = 1 / (sqx + sqy + sqz + sqw)
        m00 = ( sqx - sqy - sqz + sqw)*invs
        m11 = (-sqx + sqy - sqz + sqw)*invs
        m22 = (-sqx - sqy + sqz + sqw)*invs
        
        tmp1 = x*y; tmp2 = z*w
        m10 = 2.0 * (tmp1 + tmp2)*invs
        m01 = 2.0 * (tmp1 - tmp2)*invs
        
        tmp1 = x*z; tmp2 = y*w
        m20 = 2.0 * (tmp1 - tmp2)*invs
        m02 = 2.0 * (tmp1 + tmp2)*invs


        
        tmp1 = y*z; tmp2 = x*w
        m21 = 2.0 * (tmp1 + tmp2)*invs
        m12 = 2.0 * (tmp1 - tmp2)*invs
        
        return np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])


def main(args=None):
    rclpy.init(args=args)
    node = TSDFDataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
