# `x3plus_perception_utils`

Perception tools for the X3Plus: saving pose recorded depth frames while the base strafes, calibrating the Azure Kinect to the robot. The recorded frames are the input for the planner in `x3plus_drake_planner`. For the project overview, see the [repository README](../README.md).

## Nodes

| Node | Role | Main inputs and outputs |
| --- | --- | --- |
| `rgbd_pose_recorder` | Saves depth frames with the camera pose, for reconstruction | Reads `/depth/image_raw`, `/depth/camera_info`, TF `odom` to `depth_camera_link`, and `/recording_enabled`; writes `NNNNN.npz` files |
| `kinect_extrinsic_calibrator` | Calibrates the Kinect against the robot's Astra camera using a ChArUco board | Reads both cameras' RGB images and camera info and the robot TF; writes `kinect_extrinsic.yaml` |

## Capture format

Each `NNNNN.npz` is one frame:

| Key | Content |
| --- | --- |
| `depth` | Raw depth image, 16-bit, in millimetres |
| `rgb` | Zero-filled placeholder at the depth resolution; **no colour is recorded** |
| `pose` | 4×4 pose of the depth camera frame in the `odom` frame |
| `timestamp_ns` | Depth image header timestamp represented as integer nanoseconds |
| `depth_intrinsics` | `[fx, fy, cx, cy, width, height]` |
| `depth_distortion` | Distortion coefficients from the camera info |
| `rgb_intrinsics`, `rgb_distortion` | Copies of the depth values, kept so that the keys match |

## Design notes

**Local capture workflow.** Start the recorder before starting the strafe node.
`recording_enabled` defaults to `true`, so the recorder is already accepting frames
and can save one initial stationary view. After that, it saves a new frame only
when the camera has translated at least 1 cm from the last saved frame
(`min_translation_m_between_captures`). This threshold checks translation, not
rotation. The strafe node's `/recording_enabled` `true` message leaves recording
enabled, and its `false` message stops accepting new frames and drains pending
timestamped images before closing the writer queue. Disk compression and writing run
in a separate process through a queue (200 frames), so slow writes do not block
the camera callback; frames are dropped with a warning if the queue is full.

**Poses and timestamps.** Each frame's pose is looked up from TF as `odom` to
`depth_camera_link` at that depth image's header timestamp, not at the latest available TF.
The timestamp is stored in the NPZ as `timestamp_ns`. Frames with a zero timestamp or without
TF available at the image timestamp are not saved with a latest-transform fallback. The recorder
queues images and retries timestamped lookups for up to 3 seconds by default
(`tf_wait_timeout`); images still lacking TF after that are dropped. The pending-image queue is
bounded (`max_pending_images`, default 100) and drops new frames when full.
The timestamp aligns the transform lookup with the depth image, but the pose is
still an estimate: it combines EKF odometry with the Kinect extrinsic from
`x3plus_bringup`, not an independent ground-truth measurement. Timestamped TF
must be available in the same ROS time domain as the image header; otherwise the
frame is skipped.

**Kinect calibration.** The Kinect is mounted on the Astra, and a ChArUco board is used as reference to find the pose of Kinect relative to Astra. Once found, Kinect has a valid part in the TF tree. `camera_base` is the base link equivalent for the Kinect. It is done as follows:

1. The board's pose is estimated in each camera with `solvePnP`.
2. The Astra-to-Kinect transform follows from the two board poses.
3. Samples are averaged: translations directly, and rotations as quaternions after aligning their
   signs.
4. The mean is combined with the Astra's pose in the robot TF to give `base_link` to `camera_base`,
   which is saved as `kinect_extrinsic.yaml`.

The result is only as accurate as the Astra's mounting in the URDF plus the board-based calibration.

## Run

```bash
# Start bringup and the Kinect driver, then start the recorder first:
ros2 run x3plus_perception_utils capture_rgbd_pose.py 

# In another terminal, start the strafe after the recorder is running:
ros2 run x3plus_motion_planner strafe_motion.py

# Calibrate (Astra and Kinect drivers and bringup running, board visible to both cameras)
ros2 run x3plus_perception_utils kinect_astra_calibration.py
```

Calibration keys: **Space** saves a sample, **S** computes and saves the result (at least 5 samples),
**Q** quits. Move the board between samples, in distance and tilt. The keys only respond while the
board is detected in both cameras. The calibration overwrites `config/kinect_extrinsic.yaml` in the
installed `x3plus_bringup` package share directory, which is used by the bringup launch file.

## Things to be aware of

- **The recorder saves depth only.** The `rgb` entry is a placeholder of zeros, so nothing downstream
  can use colour.
- **One recording per run.** The strafe node's end message stops new image intake, lets pending
  timestamped images finish or expire, then closes the disk-writer queue. Restart the recorder for
  the next capture.
- **Use a new, empty output folder for each capture.** Files are numbered from `00000.npz`, and the
  recorder does not clear existing files. Stale files in a reused folder can be mistaken for part of
  the current capture. The default output directory is inside the installed package; set
  `output_directory` explicitly if the planner should read the capture from the source tree.
- **Sampling and pose limitations.** The 1 cm gate is translation-only, so rotation by itself does
  not trigger a capture. Saved poses are timestamp-aligned TF estimates, but can still contain
  odometry drift or camera-extrinsic error. Frames that exceed the TF wait or queue capacity, or
  encounter a full disk-writer queue, are dropped and logged.

## Requirements

ROS 2, OpenCV with the `aruco` module, Open3D, NumPy, `cv_bridge`, `message_filters` and
`tf_transformations`. The robot's TF (`x3plus_bringup`), the Azure Kinect driver, and for calibration
the Astra driver (`OrbbecSDK_ROS2`).
