#!/usr/bin/env python3
"""
TSDF capture recorder - continuous-motion version.

Reverted from a stop-and-go handshake design: that eliminated the
stale-frame bug but required ~150+ stop/start cycles for a mecanum
base, which is exactly the kind of motion mecanum wheels handle worst
(each start/stop is a slip opportunity). This version captures
continuously during a smooth sweep instead, and relies on two
independent checks per candidate frame - no handshake, no waypoints,
no stopping:

  1. Angular spacing: only consider saving once the camera has moved
     more than min_angle_deg_between_captures since the last SAVED frame.
     This alone throttles capture rate to match desired density
     regardless of how fast or slow the sweep happens to be.

  2. Content freshness: once that much motion has happened, the new
     frame's depth must actually differ from the last saved frame by
     more than min_depth_change_mm. If the pose moved but depth content
     didn't, the sensor pipeline has fallen behind (this is exactly the
     bug that produced a duplicated object in the fused reconstruction
     previously) - skip and wait for a genuinely new frame instead of
     silently saving a stale one.

No stopping is required for either check - both operate purely on
already-published frames while the base keeps sweeping smoothly.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import numpy as np
import os
import multiprocessing as mp

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import Buffer, TransformListener, TransformException
from std_msgs.msg import Bool


def disk_writer_worker(queue):
    """Runs in a separate process. Drains the queue until it receives
    the sentinel value None, then exits cleanly - does not depend on
    any live attribute of the parent Node (multiprocessing.Process
    forks a snapshot, so attributes on the parent keep changing after
    the fork but the child never sees those updates)."""
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
        self.declare_parameter('min_angle_deg_between_captures', 1.5)
        self.declare_parameter('min_depth_change_mm', 3.0)      # tune to your sensor's noise floor
        self.declare_parameter('recording_enabled', True)       # motion node can set this False when done

        self.world_frame = self.get_parameter('world_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.output_dir = self.get_parameter('output_directory').value
        self.max_captures = self.get_parameter('max_captures').value
        self.min_angle_deg = self.get_parameter('min_angle_deg_between_captures').value
        self.min_depth_change_mm = self.get_parameter('min_depth_change_mm').value
        self.recording_enabled = self.get_parameter('recording_enabled').value

        self.frame_count = 0
        self.bridge = CvBridge()

        os.makedirs(self.output_dir, exist_ok=True)

        self.save_queue = mp.Queue(maxsize=200)
        self.worker = mp.Process(target=disk_writer_worker, args=(self.save_queue,), daemon=True)
        self.worker.start()

        self.last_saved_depth = None
        self.last_saved_yaw_deg = None

        # Motion node publishes False once its sequence completes, to stop
        # the recorder cleanly. Optional - if never received, max_captures
        # (if set) or simply killing the node both work fine too.
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

        self.get_logger().info("Recorder started (continuous-motion mode).")

    # ------------------------------------------------------------------

    def recording_enabled_callback(self, msg):
        self.recording_enabled = msg.data

    @staticmethod
    def yaw_deg_from_matrix(R):
        return np.degrees(np.arctan2(R[1, 0], R[0, 0]))

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
        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = self.quaternion_to_matrix(q.x, q.y, q.z, q.w)
        pose_matrix[:3, 3] = [t.x, t.y, t.z]
        current_yaw_deg = self.yaw_deg_from_matrix(pose_matrix[:3, :3])

        # Angular spacing gate - skip entirely if we haven't moved enough
        # yet. This is what controls capture density (no handshake needed).
        angle_moved = None
        if self.last_saved_yaw_deg is not None:
            angle_moved = abs(current_yaw_deg - self.last_saved_yaw_deg)
            if angle_moved < self.min_angle_deg:
                return

        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception:
            return

        # Content freshness gate - the pose moved enough, but did the
        # depth actually change? If not, the sensor pipeline has fallen
        # behind the motion - skip rather than save a stale/duplicated frame.
        if self.last_saved_depth is not None:
            depth_diff = float(np.mean(np.abs(
                cv_depth.astype(np.float32) - self.last_saved_depth.astype(np.float32)
            )))
            if depth_diff < self.min_depth_change_mm:
                self.get_logger().warn(
                    f"Skipping frame at yaw={current_yaw_deg:.2f} deg - pose moved "
                    f"{angle_moved:.2f} deg but depth diff only {depth_diff:.2f}mm "
                    f"(sensor may be lagging the motion)."
                )
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
            self.get_logger().warn("Save queue full - dropping this frame.")
            return

        self.last_saved_depth = cv_depth.copy()
        self.last_saved_yaw_deg = current_yaw_deg
        self.frame_count += 1
        self.get_logger().info(f"Saved capture {frame_idx} at yaw={current_yaw_deg:.2f} deg")

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

    def shutdown(self):
        try:
            self.save_queue.put_nowait(None)
        except Exception:
            pass
        self.worker.join(timeout=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = TSDFHighSpeedRecorder()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()