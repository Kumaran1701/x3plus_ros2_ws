#!/usr/bin/env python3
"""
TSDF capture recorder - stop-and-go handshake version.

Redesigned to fix a data-corruption bug found empirically: during fast
continuous rotation, the depth sensor pipeline fell behind and
republished the last-rendered depth frame under a fresh timestamp while
TF (computed independently from joint/odometry state) kept reporting
the robot's true, moving pose. ApproximateTimeSynchronizer has no way
to detect this - it only checks that timestamps line up, not that
content actually changed - so stale depth got silently saved paired
with a genuinely-updated pose. In the fused reconstruction this showed
up as the same real object duplicated at multiple positions.

Fix, two layers:
  1. Protocol: capturing now only happens once the motion node reports
     it has stopped and settled at a waypoint (see
     ekf_mecanum_rotation_node.py), not continuously during motion.
     This eliminates motion blur and removes almost all of the odds of
     a stale-frame race in the first place.
  2. Verification: even while "settled", this node refuses to accept a
     frame unless its depth content actually differs from the
     previously SAVED frame by more than min_depth_change_mm, once the
     commanded pose has moved by more than min_angle_deg_between_captures.
     This is a hard content-level check, so it catches a frozen sensor
     even if the handshake protocol above is ever bypassed or misconfigured.

Handshake:
  motion node -> /ready_for_capture (Bool): True once stopped + settled
                 at a waypoint and requesting a capture there.
  this node   -> /capture_ack (Int32): monotonically increasing count,
                 published once a frame has been validated and saved.
                 The motion node waits for this count to increase before
                 advancing to the next waypoint (with a timeout fallback
                 so a single bad waypoint can't deadlock the whole run).
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
from std_msgs.msg import Bool, Int32


def disk_writer_worker(queue):
    """Runs in a separate process. Drains the queue until it receives
    the sentinel value None, then exits cleanly.

    Does NOT depend on any live attribute of the parent Node.
    multiprocessing.Process forks a SNAPSHOT of the parent's memory, so
    attributes like self.frame_count or self.state_motion in the parent
    keep changing after the fork but the child process never sees those
    updates. The previous version's loop condition relied on exactly
    that live state and was effectively dead logic - a separate,
    unrelated correctness bug worth fixing alongside the stale-frame issue.
    """
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
        self.declare_parameter('min_angle_deg_between_captures', 1.0)
        self.declare_parameter('min_depth_change_mm', 3.0)       # tune to your sensor's noise floor
        self.declare_parameter('stale_retry_timeout_sec', 2.0)   # give up on a waypoint after this long

        self.world_frame = self.get_parameter('world_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.output_dir = self.get_parameter('output_directory').value
        self.max_captures = self.get_parameter('max_captures').value
        self.min_angle_deg = self.get_parameter('min_angle_deg_between_captures').value
        self.min_depth_change_mm = self.get_parameter('min_depth_change_mm').value
        self.stale_retry_timeout_sec = self.get_parameter('stale_retry_timeout_sec').value

        self.frame_count = 0
        self.bridge = CvBridge()

        os.makedirs(self.output_dir, exist_ok=True)

        # Multiprocessing queue + writer process (see disk_writer_worker docstring)
        self.save_queue = mp.Queue(maxsize=200)
        self.worker = mp.Process(target=disk_writer_worker, args=(self.save_queue,), daemon=True)
        self.worker.start()

        # Handshake state
        self.ready_for_capture = False
        self.last_saved_depth = None
        self.last_saved_yaw_deg = None
        self.awaiting_since = None  # ROS time when the current waypoint's capture request started

        self.pub_capture_ack = self.create_publisher(Int32, '/capture_ack', 10)
        self.sub_ready_for_capture = self.create_subscription(
            Bool, '/ready_for_capture', self.ready_for_capture_callback, 10)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
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

        self.get_logger().info("Recorder started (stop-and-go handshake mode).")

    # ------------------------------------------------------------------

    def ready_for_capture_callback(self, msg):
        was_ready = self.ready_for_capture
        self.ready_for_capture = msg.data
        if self.ready_for_capture and not was_ready:
            self.awaiting_since = self.get_clock().now()

    @staticmethod
    def yaw_deg_from_matrix(R):
        return np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    def synchronized_callback(self, depth_msg, depth_info_msg, rgb_msg, rgb_info_msg):
        if not self.ready_for_capture:
            return  # only ever capture while the motion node reports stopped + settled

        if self.max_captures and self.frame_count >= self.max_captures:
            return  # safety cap reached

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

        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception:
            return

        # --- Content-level freshness check ---
        # Enforced once the commanded pose has moved by more than
        # min_angle_deg since the last SAVED frame. The very first
        # capture (nothing saved yet) has nothing to compare against
        # and is accepted immediately.
        if self.last_saved_yaw_deg is not None and self.last_saved_depth is not None:
            angle_moved = abs(current_yaw_deg - self.last_saved_yaw_deg)
            if angle_moved > self.min_angle_deg:
                depth_diff = float(np.mean(np.abs(
                    cv_depth.astype(np.float32) - self.last_saved_depth.astype(np.float32)
                )))
                if depth_diff < self.min_depth_change_mm:
                    # Pose moved but depth content didn't - sensor is stuck.
                    # Refuse to save. Keep waiting for a genuinely new
                    # frame, but give up after stale_retry_timeout_sec so
                    # one bad waypoint can't hang the run forever.
                    if self.awaiting_since is not None:
                        elapsed = (self.get_clock().now() - self.awaiting_since).nanoseconds / 1e9
                        if elapsed > self.stale_retry_timeout_sec:
                            self.get_logger().warn(
                                f"Depth frozen for {elapsed:.1f}s at yaw={current_yaw_deg:.2f} deg "
                                f"(diff={depth_diff:.2f}mm < {self.min_depth_change_mm}mm threshold). "
                                f"Giving up on this waypoint - it will be a gap in the dataset "
                                f"rather than a silently corrupted duplicate."
                            )
                            self.ready_for_capture = False
                        else:
                            self.get_logger().info(
                                f"Waiting for fresh depth at yaw={current_yaw_deg:.2f} deg "
                                f"(diff={depth_diff:.2f}mm, retrying)..."
                            )
                    return

        # --- Passed all checks: accept and queue this frame ---
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
        self.ready_for_capture = False  # consumed - wait for the next explicit request

        ack_msg = Int32()
        ack_msg.data = self.frame_count
        self.pub_capture_ack.publish(ack_msg)
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