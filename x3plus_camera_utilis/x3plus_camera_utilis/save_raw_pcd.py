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

class TSDFHighSpeedRecorder(Node):
    def __init__(self):
        super().__init__('tsdf_high_speed_recorder')

        # Parameters
        self.declare_parameter('total_frames_to_save', 50)
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('output_directory', 'tsdf_dataset')

        self.total_frames = self.get_parameter('total_frames_to_save').value
        self.world_frame = self.get_parameter('world_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.output_dir = self.get_parameter('output_directory').value

        self.frame_count = 0
        self.bridge = CvBridge()

        os.makedirs(self.output_dir, exist_ok=True)

        # Multiprocessing queue (fast, bypasses GIL)
        self.save_queue = mp.Queue(maxsize=200)

        # Start worker process
        self.worker = mp.Process(target=self._disk_writer_worker, daemon=True)
        self.worker.start()

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.sub_depth = Subscriber(self, Image, '/depth/image_raw')
        self.sub_rgb = Subscriber(self, Image, '/rgb_to_depth/image_raw')
        self.sub_info = Subscriber(self, CameraInfo, '/rgb_to_depth/camera_info')

        # Approximate sync (MUCH faster)
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_rgb, self.sub_info],
            queue_size=30,
            slop=0.03
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.get_logger().info("Optimized TSDF Recorder Started.")

    def synchronized_callback(self, depth_msg, rgb_msg, info_msg):
        if self.frame_count >= self.total_frames:
            return

        timestamp = depth_msg.header.stamp
        frame_idx = str(self.frame_count).zfill(5)

        # TF lookup
        try:
            tf_transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_frame,
                timestamp,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except TransformException:
            return

        t = tf_transform.transform.translation
        q = tf_transform.transform.rotation

        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = self.quaternion_to_matrix(q.x, q.y, q.z, q.w)
        pose_matrix[:3, 3] = [t.x, t.y, t.z]

        # Convert images (zero-copy)
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception:
            return

        intrinsics = np.array([
            info_msg.k[0], info_msg.k[4], info_msg.k[2], info_msg.k[5],
            info_msg.width, info_msg.height
        ], dtype=np.float32)

        # Push to multiprocessing queue
        self.save_queue.put_nowait((frame_idx, cv_depth, cv_rgb, pose_matrix, intrinsics))

        self.frame_count += 1

    def _disk_writer_worker(self):
        """Runs in a separate process (true parallelism)."""
        while True:
            frame_idx, cv_depth, cv_rgb, pose_matrix, intrinsics = self.save_queue.get()

            # Save everything in one NPZ (fast)
            np.savez_compressed(
                os.path.join(self.output_dir, f"{frame_idx}.npz"),
                depth=cv_depth,
                rgb=cv_rgb,
                pose=pose_matrix,
                intrinsics=intrinsics
            )

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

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
