#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os
import multiprocessing as mp
import time
from collections import deque
from queue import Full

from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from std_msgs.msg import Bool


def disk_writer_worker(queue):
    """Runs in a separate process to write data to disk."""
    while True:
        item = queue.get()
        if item is None:
            break
        (frame_idx, cv_depth, cv_rgb, pose_matrix, timestamp_ns,
         depth_intrinsics, depth_distortion,
         rgb_intrinsics, rgb_distortion, output_dir) = item
        np.savez_compressed(
            os.path.join(output_dir, f"{frame_idx}.npz"),
            depth=cv_depth,
            rgb=cv_rgb,
            pose=pose_matrix,
            timestamp_ns=np.int64(timestamp_ns),
            depth_intrinsics=depth_intrinsics,
            depth_distortion=depth_distortion,
            rgb_intrinsics=rgb_intrinsics,
            rgb_distortion=rgb_distortion
        )


class RGBDPoseRecorder(Node):
    """
    Record time syncronized RGB-D, camera intrinsics, pose 
    for TSDF-based reconstruction or pointcloud fusion.

    Queues frames until TF for their image timestamps is available.
    Frames are saved only when a minimum translation threshold is met.
    """

    def __init__(self):
        super().__init__('rgbd_pose_recorder')

        pkg_path_perception = get_package_share_directory('x3plus_perception_utils')

        # --- Parameters ---
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('camera_frame', 'depth_camera_link')
        self.declare_parameter('output_directory', os.path.join(pkg_path_perception, 'rgbd_pose_dataset/latest'))
        self.declare_parameter('max_captures', 0)               
        self.declare_parameter('min_translation_m_between_captures', 0.01) # 1cm
        self.declare_parameter('recording_enabled', True)       
        self.declare_parameter('tf_wait_timeout', 3.0)
        self.declare_parameter('max_pending_images', 100)

        self.world_frame = self.get_parameter('world_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.output_dir = self.get_parameter('output_directory').value
        self.max_captures = self.get_parameter('max_captures').value
        self.min_translation_m = self.get_parameter('min_translation_m_between_captures').value
        self.recording_enabled = self.get_parameter('recording_enabled').value
        self.tf_wait_timeout = self.get_parameter('tf_wait_timeout').value
        self.max_pending_images = self.get_parameter('max_pending_images').value
        if self.tf_wait_timeout <= 0:
            raise ValueError('tf_wait_timeout must be positive')
        if self.max_pending_images <= 0:
            raise ValueError('max_pending_images must be positive')

        self.frame_count = 0
        self.bridge = CvBridge()
        self.pending_images = deque()
        self.stop_requested = False
        self.writer_stop_sent = False

        os.makedirs(self.output_dir, exist_ok=True)

        # Multiprocessing Setup
        self.save_queue = mp.Queue(maxsize=200)
        self.worker = mp.Process(target=disk_writer_worker, args=(self.save_queue,), daemon=True)
        self.worker.start()

        self.last_saved_position = None  

        # Stop recording once the motion controller publishes False.
        self.recording_status_sub = self.create_subscription(
            Bool, '/recording_enabled', self.recording_enabled_callback, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Camera Cache State
        self.depth_info_received = False
        self.depth_intrinsics = None
        self.depth_distortion = None

        # Direct Subscriptions
        self.camera_info_direct_sub = self.create_subscription(
            CameraInfo, '/depth/camera_info', self.depth_info_callback, 10)
        
        self.depth_image_direct_sub = self.create_subscription(
            Image, '/depth/image_raw', self.depth_callback, 10)
        self.pending_image_timer = self.create_timer(0.02, self.process_pending_images)

        self.get_logger().info("Recorder initialized. Waiting for /depth/camera_info topic...")


    def recording_enabled_callback(self, msg):
        """
        Starts accepting frames on True and drains pending frames on False.
        """

        new_state = msg.data
        if not new_state and self.recording_enabled:
            self.get_logger().info("Recording stop requested; draining pending timestamped frames.")
            self.recording_enabled = False
            self.stop_requested = True
        elif new_state and not self.writer_stop_sent:
            self.recording_enabled = True


    def depth_info_callback(self, msg):
        """
        Stores camera parameters once and stops logging.
        """

        if not self.depth_info_received:
            self.depth_intrinsics = np.array([
                msg.k[0], msg.k[4], msg.k[2], msg.k[5],
                msg.width, msg.height
            ], dtype=np.float32)
            self.depth_distortion = np.array(msg.d, dtype=np.float32)
            self.depth_info_received = True
            self.get_logger().info(f"Successfully connected to camera info! Dimensions: {msg.width}x{msg.height}")

    def depth_callback(self, depth_msg):
        """
        Queues depth images until TF for their image timestamps is available.
        """

        if not self.recording_enabled:
            return
            
        if not self.depth_info_received:
            self.get_logger().warn("Skipping frame: Still waiting for camera info metadata...", throttle_duration_sec=2.0)
            return
            
        if self.max_captures and self.frame_count >= self.max_captures:
            return

        timestamp = depth_msg.header.stamp
        timestamp_ns = timestamp.sec * 1_000_000_000 + timestamp.nanosec
        if timestamp_ns == 0:
            self.get_logger().warn(
                "Skipping frame: image timestamp is zero; cannot look up timestamped TF.",
                throttle_duration_sec=2.0,
            )
            return

        if len(self.pending_images) >= self.max_pending_images:
            self.get_logger().warn(
                "Pending image queue is full; dropping incoming frame.",
                throttle_duration_sec=2.0,
            )
            return

        self.pending_images.append((depth_msg, time.monotonic()))

    def process_pending_images(self):
        if not self.pending_images:
            if self.stop_requested:
                self.stop_writer()
            return

        if self.max_captures and self.frame_count >= self.max_captures:
            self.pending_images.clear()
            if self.stop_requested:
                self.stop_writer()
            return

        depth_msg, queued_at = self.pending_images[0]
        timestamp = depth_msg.header.stamp
        timestamp_ns = timestamp.sec * 1_000_000_000 + timestamp.nanosec
        source_frame = depth_msg.header.frame_id or self.camera_frame
        try:
            tf_transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                source_frame,
                Time.from_msg(timestamp),
            )
        except TransformException as exc:
            waited = time.monotonic() - queued_at
            if waited < self.tf_wait_timeout:
                self.get_logger().warn(
                    f"Waiting for timestamped TF {self.world_frame} <- {source_frame} "
                    f"at {timestamp.sec}.{timestamp.nanosec:09d}: {exc}",
                    throttle_duration_sec=2.0,
                )
                return

            self.pending_images.popleft()
            self.get_logger().warn(
                f"Dropping image after waiting {waited:.2f}s for TF "
                f"{self.world_frame} <- {source_frame} at "
                f"{timestamp.sec}.{timestamp.nanosec:09d}: {exc}",
            )
            return

        self.pending_images.popleft()
        self.save_image_with_transform(depth_msg, timestamp_ns, tf_transform)

    def save_image_with_transform(self, depth_msg, timestamp_ns, tf_transform):

        t = tf_transform.transform.translation
        q = tf_transform.transform.rotation
        current_position = np.array([t.x, t.y, t.z], dtype=np.float32)

        # Distance Check Gate
        if self.last_saved_position is not None:
            distance_moved = np.linalg.norm(current_position - self.last_saved_position)
            if distance_moved < self.min_translation_m:
                return

        # Build transform matrix
        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = self.quaternion_to_matrix(q.x, q.y, q.z, q.w)
        pose_matrix[:3, 3] = current_position

        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            cv_rgb = np.zeros((depth_msg.height, depth_msg.width, 3), dtype=np.uint8)
        except Exception as e:
            self.get_logger().error(f"CvBridge Conversion failed: {str(e)}")
            return

        frame_idx = str(self.frame_count).zfill(5)

        try:
            self.save_queue.put_nowait((
                frame_idx, cv_depth, cv_rgb, pose_matrix, timestamp_ns,
                self.depth_intrinsics, self.depth_distortion,
                self.depth_intrinsics, self.depth_distortion, self.output_dir
            ))
        except Exception:
            self.get_logger().warn("Save queue full - dropping frame.")
            return

        self.last_saved_position = current_position.copy()
        self.frame_count += 1
        self.get_logger().info(
            f"Saved capture {frame_idx} at image timestamp {timestamp_ns} ns "
            f"and translation offset Y={t.y:.3f}m"
        )

    def stop_writer(self):
        if self.writer_stop_sent:
            return
        try:
            self.save_queue.put_nowait(None)
        except Full:
            self.get_logger().warn(
                "Waiting for disk writer queue to drain before stopping.",
                throttle_duration_sec=2.0,
            )
            return
        self.writer_stop_sent = True
        self.get_logger().info("Disk writer queue closed after pending frames were processed.")


    def quaternion_to_matrix(self, x, y, z, w):
        """
        Convert a quaternion to a 3×3 rotation matrix.
        """

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
    node = RGBDPoseRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
