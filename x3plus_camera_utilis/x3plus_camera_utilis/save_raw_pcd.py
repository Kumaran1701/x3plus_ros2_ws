#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os
import multiprocessing as mp

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
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
        self.declare_parameter('min_depth_change_mm', 3.0)      # tune to your sensor's noise floor
        self.declare_parameter('recording_enabled', True)       # motion node sets this False when done

        self.world_frame = self.get_parameter('world_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.output_dir = self.get_parameter('output_directory').value
        self.max_captures = self.get_parameter('max_captures').value
        self.min_translation_m = self.get_parameter('min_translation_m_between_captures').value
        self.min_depth_change_mm = self.get_parameter('min_depth_change_mm').value
        self.recording_enabled = self.get_parameter('recording_enabled').value

        self.frame_count = 0
        self.bridge = CvBridge()

        os.makedirs(self.output_dir, exist_ok=True)

        self.save_queue = mp.Queue(maxsize=200)
        self.worker = mp.Process(target=disk_writer_worker, args=(self.save_queue,), daemon=True)
        self.worker.start()

        # Target historical references swapped for linear spatial tracking
        self.last_saved_depth = None
        self.last_saved_position = None  # Tracks [x, y, z] arrays

        # Motion node status interface
        self.sub_recording_enabled = self.create_subscription(
            Bool, '/recording_enabled', self.recording_enabled_callback, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub_depth = Subscriber(self, Image, '/depth/image_raw')
        self.sub_depth_info = Subscriber(self, CameraInfo, '/depth/camera_info')
        self.sub_rgb = Subscriber(self, Image, '/rgb/image_raw')
        self.sub_rgb_info = Subscriber(self, CameraInfo, '/rgb/camera_info')

        self.ts = ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_depth_info,
             self.sub_rgb, self.sub_rgb_info],
            queue_size=30,
            slop=0.03
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.get_logger().info("Recorder updated for STRAFE linear continuous tracking.")

    def recording_enabled_callback(self, msg):
        self.recording_enabled = msg.data
        # If motion node sets to False, trigger clean worker shutdown
        if not self.recording_enabled:
            self.get_logger().info("Recording finished. Sending shutdown token to worker...")
            self.save_queue.put(None)

    def synchronized_callback(self, depth_msg, depth_info_msg, rgb_msg, rgb_info_msg):
        if not self.recording_enabled:
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

        # FIX: Linear displacement space tracking gate
        if self.last_saved_position is not None:
            distance_moved = np.linalg.norm(current_position - self.last_saved_position)
            if distance_moved < self.min_translation_m:
                return  # Skip processing, robot hasn't strafed far enough yet

        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = self.quaternion_to_matrix(q.x, q.y, q.z, q.w)
        pose_matrix[:3, 3] = current_position

        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception:
            return

        # Content freshness gate (retained to filter sensor lag)
        if self.last_saved_depth is not None:
            depth_diff = float(np.mean(np.abs(
                cv_depth.astype(np.float32) - self.last_saved_depth.astype(np.float32)
            )))
            if depth_diff < self.min_depth_change_mm:
                return

        frame_idx = str(self.frame_count).zfill(5)

        depth_intrinsics = np.array([
            depth_info_msg.k[0], depth_info_msg.k[4],
            depth_info_msg.k[2], depth_info_msg.k[5],
            depth_info_msg.width, depth_info_msg.height
        ], dtype=np.float32)
        depth_distortion = np.array(depth_info_msg.d, dtype=np.float32)

        rgb_intrinsics = np.array([
            rgb_info_msg.k[0], rgb_info_msg.k[4],
            rgb_info_msg.k[2], rgb_info_msg.k[5],
            rgb_info_msg.width, rgb_info_msg.height
        ], dtype=np.float32)
        rgb_distortion = np.array(rgb_info_msg.d, dtype=np.float32)

        try:
            self.save_queue.put_nowait((
                frame_idx, cv_depth, cv_rgb, pose_matrix,
                depth_intrinsics, depth_distortion,
                rgb_intrinsics, rgb_distortion, self.output_dir
            ))
        except Exception:
            self.get_logger().warn("Save queue full - dropping frame.")
            return

        self.last_saved_depth = cv_depth.copy()
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
        # Cleanly shut down node structures
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
