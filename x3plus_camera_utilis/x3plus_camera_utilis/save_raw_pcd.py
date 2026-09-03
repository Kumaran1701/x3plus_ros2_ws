#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os
import multiprocessing as mp

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration
from std_msgs.msg import Bool


def disk_writer_worker(queue):
    """Runs in a separate process. Drains the queue until it receives
    the sentinel value None, then exits cleanly."""
    while True:
        item = queue.get()
        if item is None:
            break
        (frame_idx, cv_depth, cv_rgb, pose_matrix,
         depth_intrinsics, depth_distortion,
         rgb_intrinsics, rgb_distortion, output_dir) = item
        np.savez_compressed(
            os.path.join(output_dir, f"{frame_idx}.npz"),
            depth=cv_depth,
            rgb=cv_rgb,
            pose=pose_matrix,
            depth_intrinsics=depth_intrinsics,
            depth_distortion=depth_distortion,
            rgb_intrinsics=rgb_intrinsics,
            rgb_distortion=rgb_distortion
        )


class TSDFHighSpeedRecorder(Node):
    def __init__(self):
        super().__init__('tsdf_high_speed_recorder')

        # --- Parameters ---
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('camera_frame', 'depth_camera_link')
        self.declare_parameter('output_directory', 'tsdf_dataset')
        self.declare_parameter('max_captures', 0)               # 0 = unlimited
        self.declare_parameter('min_translation_m_between_captures', 0.01) # 1cm of linear movement
        self.declare_parameter('recording_enabled', True)       # motion node sets this False when done

        self.world_frame = self.get_parameter('world_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.output_dir = self.get_parameter('output_directory').value
        self.max_captures = self.get_parameter('max_captures').value
        self.min_translation_m = self.get_parameter('min_translation_m_between_captures').value
        self.recording_enabled = self.get_parameter('recording_enabled').value

        self.frame_count = 0
        self.bridge = CvBridge()

        os.makedirs(self.output_dir, exist_ok=True)

        # Multiprocessing Setup
        self.save_queue = mp.Queue(maxsize=200)
        self.worker = mp.Process(target=disk_writer_worker, args=(self.save_queue,), daemon=True)
        self.worker.start()

        # Displacement and Tracking references
        self.last_saved_position = None  # Tracks [x, y, z] spatial matrix

        # Motion node status interface
        self.sub_recording_enabled = self.create_subscription(
            Bool, '/recording_enabled', self.recording_enabled_callback, 10)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Camera Cache State
        self.depth_info_received = False
        self.depth_intrinsics = None
        self.depth_distortion = None

        # Direct Subscriptions (No Synchronizer Roadblocks)
        self.sub_depth_info = self.create_subscription(
            CameraInfo, '/depth/camera_info', self.depth_info_callback, 10)
        
        self.sub_depth = self.create_subscription(
            Image, '/depth/image_raw', self.depth_callback, 10)

        self.get_logger().info("Recorder running in optimized Depth-Only Mode for linear strafing.")

    def recording_enabled_callback(self, msg):
        new_state = msg.data
        # Safety gate: Only process a shutdown request (False) if we have successfully saved data
        if not new_state and self.frame_count > 0 and self.recording_enabled:
            self.get_logger().info("Recording finished by motion node request. Sending shutdown token...")
            self.recording_enabled = False
            self.save_queue.put(None)
        elif new_state:
            self.recording_enabled = True

    def depth_info_callback(self, msg):
        """Caches camera parameters once to save precious CPU cycles on the Jetson."""
        if not self.depth_info_received:
            # Layout matching your parser expectation: [fx, fy, cx, cy, width, height]
            self.depth_intrinsics = np.array([
                msg.k[0], msg.k[4], msg.k[2], msg.k[5],
                msg.width, msg.height
            ], dtype=np.float32)
            self.depth_distortion = np.array(msg.d, dtype=np.float32)
            self.depth_info_received = True
            self.get_logger().info("Successfully cached Depth Camera Intrinsics.")

    def depth_callback(self, depth_msg):
        """Processes depth images instantly based on discrete physical displacement metrics."""
        if not self.recording_enabled or not self.depth_info_received:
            return
        if self.max_captures and self.frame_count >= self.max_captures:
            return

        timestamp = depth_msg.header.stamp
        try:
            tf_transform = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame, timestamp,
                timeout=Duration(seconds=0.05)
            )
        except TransformException:
            return

        t = tf_transform.transform.translation
        q = tf_transform.transform.rotation
        current_position = np.array([t.x, t.y, t.z], dtype=np.float32)

        # Displacement Gate: Has the robot translated far enough sideways?
        if self.last_saved_position is not None:
            distance_moved = np.linalg.norm(current_position - self.last_saved_position)
            if distance_moved < self.min_translation_m:
                return

        # Transform Orientation Math
        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = self.quaternion_to_matrix(q.x, q.y, q.z, q.w)
        pose_matrix[:3, 3] = current_position

        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            # Synthesize a clean black dummy image matrix to satisfy downstream .npz dataset loaders
            cv_rgb = np.zeros((depth_msg.height, depth_msg.width, 3), dtype=np.uint8)
        except Exception:
            return

        frame_idx = str(self.frame_count).zfill(5)

        try:
            self.save_queue.put_nowait((
                frame_idx, cv_depth, cv_rgb, pose_matrix,
                self.depth_intrinsics, self.depth_distortion,
                self.depth_intrinsics, self.depth_distortion, self.output_dir
            ))
        except Exception:
            self.get_logger().warn("Multiprocessing save queue full - dropping frame.")
            return

        self.last_saved_position = current_position.copy()
        self.frame_count += 1
        self.get_logger().info(f"Saved capture {frame_idx} at translation offset Y={t.y:.3f}m")

    def quaternion_to_matrix(self, x, y, z, w):
        sqw = w*w; sqx = x*x; sqy = y*y; sqz = z*z
        invs = 1.0 / (sqx + sqy + sqz + sqw)
        m00 = (sqx - sqy - sqz + sqw) * invs
        m11 = (-sqx + sqy - sqz + sqw) * invs
        m22 = (-sqx - sqy + sqz + sqw) * invs
        tmp1 = x*y; tmp2 = z*w
        m10 = 2.0 * (tmp1 + tmp2) * invs
        m01 = 2.0 * (tmp1 - tmp2) * invs
        tmp1 = x*z; tmp2 = y*w
        m20 = 2.0 * (tmp1 - tmp2) * invs
        m02 = 2.0 * (tmp1 + tmp2) * invs
        tmp1 = y*z; tmp2 = x*w
        m21 = 2.0 * (tmp1 + tmp2) * invs
        m12 = 2.0 * (tmp1 - tmp2) * invs
        return np.array([[m00, m01, m02],
                         [m10, m11, m12],
                         [m20, m21, m22]], dtype=np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = TSDFHighSpeedRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
