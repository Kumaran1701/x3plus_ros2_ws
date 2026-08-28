#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import json
import os
import threading
from collections import deque

# ROS 2 Core Message Types
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# ROS 2 Time Synchronization and TF Tracking
from message_filters import Subscriber, TimeSynchronizer
from tf2_ros import TransformException, Buffer, TransformListener

class TSDFHighSpeedRecorder(Node):
    def __init__(self):
        super().__init__('tsdf_high_speed_recorder')

        # Configurable Parameters
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

        # Create structured storage folders
        self.depth_dir = os.path.join(self.output_dir, 'depth')
        self.rgb_dir = os.path.join(self.output_dir, 'rgb')
        self.pose_dir = os.path.join(self.output_dir, 'pose')
        self.intrinsics_dir = os.path.join(self.output_dir, 'intrinsics')
        for folder in [self.depth_dir, self.rgb_dir, self.pose_dir, self.intrinsics_dir]:
            os.makedirs(folder, exist_ok=True)

        # 1. Thread-safe Queue and Background Worker Setup
        self.save_queue = deque()
        self.queue_lock = threading.Lock()
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self._disk_writer_worker, daemon=True)
        self.worker_thread.start()

        # 2. Setup TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 3. Synchronize Subscriptions
        self.sub_depth = Subscriber(self, Image, '/depth/image_raw')
        self.sub_rgb = Subscriber(self, Image, '/rgb_to_depth/image_raw')
        self.sub_info = Subscriber(self, CameraInfo, '/rgb_to_depth/camera_info')

        self.ts = TimeSynchronizer([self.sub_depth, self.sub_rgb, self.sub_info], queue_size=30)
        self.ts.registerCallback(self.synchronized_callback)

        self.get_logger().info("High-Speed TSDF Recorder Started (Queue Multi-Threading Active).")

    def synchronized_callback(self, depth_msg, rgb_msg, info_msg):
        if self.frame_count >= self.total_frames:
            return

        timestamp = depth_msg.header.stamp
        frame_idx = str(self.frame_count).zfill(5)

        # 1. Grab TF Pose instantly while timestamps match
        try:
            tf_transform = self.tf_buffer.lookup_transform(
                self.world_frame, 
                self.camera_frame, 
                timestamp,
                rclpy.duration.Duration(seconds=0.05)
            )
            t = tf_transform.transform.translation
            q = tf_transform.transform.rotation

            pose_matrix = np.eye(4, dtype=np.float64)
            pose_matrix[0:3, 0:3] = self.quaternion_to_matrix(q.x, q.y, q.z, q.w)
            pose_matrix[0:3, 3] = [t.x, t.y, t.z]

        except TransformException as ex:
            self.get_logger().warn(f"TF lookup skipped for frame {frame_idx}: {ex}")
            return

        # 2. Fast memory conversion (No disk saving here)
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Image pointer parsing error: {e}")
            return

        intrinsics_data = {
            "timestamp": {"secs": timestamp.sec, "nsecs": timestamp.nanosec},
            "width": info_msg.width, "height": info_msg.height,
            "fx": info_msg.k[0], "fy": info_msg.k[4], "cx": info_msg.k[2], "cy": info_msg.k[5],
            "depth_scale_factor": 1000.0
        }

        # 3. Offload to Queue instantly
        with self.queue_lock:
            self.save_queue.append((frame_idx, cv_depth, cv_rgb, pose_matrix, intrinsics_data))
        
        self.get_logger().info(f"Queued frame {frame_idx} [Queue Size: {len(self.save_queue)}]")
        self.frame_count += 1

    def _disk_writer_worker(self):
        """Runs continuously in the background, consuming data from the queue and writing to disk."""
        while self.worker_running:
            data = None
            with self.queue_lock:
                if self.save_queue:
                    data = self.save_queue.popleft()
            
            if data is None:
                threading.Event().wait(0.01) # Sleep 10ms if queue is empty
                continue

            frame_idx, cv_depth, cv_rgb, pose_matrix, intrinsics_data = data

            # Execute the heavy disk I/O operations asynchronously
            cv2.imwrite(os.path.join(self.depth_dir, f"frame_{frame_idx}.png"), cv_depth)
            cv2.imwrite(os.path.join(self.rgb_dir, f"frame_{frame_idx}.png"), cv_rgb)
            np.savetxt(os.path.join(self.pose_dir, f"frame_{frame_idx}.txt"), pose_matrix, fmt='%.8f')
            
            with open(os.path.join(self.intrinsics_dir, f"frame_{frame_idx}.json"), 'w') as f:
                json.dump(intrinsics_data, f, indent=4)

    def quaternion_to_matrix(self, x, y, z, w):
        sqw = w*w; sqx = x*x; sqy = y*y; sqz = z*z
        invs = 1 / (sqx + sqy + sqz + sqw)
        m00 = (sqx - sqy - sqz + sqw)*invs
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

    def shutdown_node(self):
        self.get_logger().info("Flushing remaining frames in background queue before exit...")
        while len(self.save_queue) > 0:
            threading.Event().wait(0.1)
        self.worker_running = False
        self.worker_thread.join()
        self.get_logger().info("All data flushed safely to disk.")

def main(args=None):
    rclpy.init(args=args)
    node = TSDFHighSpeedRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_node()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
