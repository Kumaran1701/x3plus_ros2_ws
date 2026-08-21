#!/usr/bin/env python3

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer

import tf2_ros
from tf_transformations import quaternion_from_matrix


class KinectExtrinsicCalibrator(Node):

    def __init__(self):
        super().__init__('kinect_extrinsic_calibrator')

        # ------------------------------------------------------------
        # Topics
        # ------------------------------------------------------------

        self.declare_parameter(
            'astra_image_topic',
            '/camera/color/image_raw'
        )

        self.declare_parameter(
            'astra_info_topic',
            '/camera/color/camera_info'
        )

        self.declare_parameter(
            'kinect_image_topic',
            '/rgb/image_raw'
        )

        self.declare_parameter(
            'kinect_info_topic',
            '/rgb/camera_info'
        )

        self.declare_parameter(
            'astra_frame',
            'camera_color_optical_frame'
        )

        self.declare_parameter(
            'base_frame',
            'base_link'
        )

        self.declare_parameter(
            'kinect_frame',
            'camera_base'
        )

        self.declare_parameter(
            'output_file',
            'kinect_extrinsic.yaml'
        )

        self.astra_image_topic = self.get_parameter(
            'astra_image_topic').value

        self.astra_info_topic = self.get_parameter(
            'astra_info_topic').value

        self.kinect_image_topic = self.get_parameter(
            'kinect_image_topic').value

        self.kinect_info_topic = self.get_parameter(
            'kinect_info_topic').value

        self.astra_frame = self.get_parameter(
            'astra_frame').value

        self.base_frame = self.get_parameter(
            'base_frame').value

        self.kinect_frame = self.get_parameter(
            'kinect_frame').value

        self.output_file = self.get_parameter(
            'output_file').value

        # ------------------------------------------------------------
        # ChArUco board
        # ------------------------------------------------------------

        self.squares_x = 5
        self.squares_y = 5

        self.square_length = 0.035   # 35 mm
        self.marker_length = 0.026   # 26 mm

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        # OpenCV API differs between versions
        try:
            self.board = cv2.aruco.CharucoBoard(
                (self.squares_x, self.squares_y),
                self.square_length,
                self.marker_length,
                self.dictionary
            )
        except Exception:
            self.board = cv2.aruco.CharucoBoard_create(
                self.squares_x,
                self.squares_y,
                self.square_length,
                self.marker_length,
                self.dictionary
            )

        self.detector_params = cv2.aruco.DetectorParameters()

        # ------------------------------------------------------------
        # ROS
        # ------------------------------------------------------------

        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.astra_image_sub = Subscriber(
            self,
            Image,
            self.astra_image_topic
        )

        self.astra_info_sub = Subscriber(
            self,
            CameraInfo,
            self.astra_info_topic
        )

        self.kinect_image_sub = Subscriber(
            self,
            Image,
            self.kinect_image_topic
        )

        self.kinect_info_sub = Subscriber(
            self,
            CameraInfo,
            self.kinect_info_topic
        )

        # Synchronize everything approximately.
        self.sync = ApproximateTimeSynchronizer(
            [
                self.astra_image_sub,
                self.astra_info_sub,
                self.kinect_image_sub,
                self.kinect_info_sub
            ],
            queue_size=20,
            slop=0.1
        )

        self.sync.registerCallback(self.image_callback)

        self.samples = []

        self.get_logger().info(
            'Kinect extrinsic calibration started.'
        )

        self.get_logger().info(
            f'Astra image:  {self.astra_image_topic}'
        )

        self.get_logger().info(
            f'Kinect image: {self.kinect_image_topic}'
        )

        self.get_logger().info(
            'Show the ChArUco board to BOTH cameras.'
        )

        self.get_logger().info(
            'Press SPACE to save a valid calibration sample.'
        )

        self.get_logger().info(
            'Press S to calculate final transform.'
        )

        self.get_logger().info(
            'Press Q to quit.'
        )

    # ================================================================
    # ChArUco detection
    # ================================================================

    def detect_board(self, image, camera_matrix, dist_coeffs):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.detector_params
        )

        if ids is None or len(ids) < 4:
            return None

        retval, charuco_corners, charuco_ids = \
            cv2.aruco.interpolateCornersCharuco(
                corners,
                ids,
                gray,
                self.board,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs
            )

        if retval is None or retval < 6:
            return None

        # ------------------------------------------------------------
        # Board 3D points corresponding to detected ChArUco IDs
        # ------------------------------------------------------------

        board_corners = self.board.getChessboardCorners()

        object_points = np.asarray(
            [board_corners[int(i)] for i in charuco_ids.flatten()],
            dtype=np.float32
        )

        image_points = np.asarray(
            charuco_corners,
            dtype=np.float32
        ).reshape(-1, 2)

        # ------------------------------------------------------------
        # Board -> camera transform
        # ------------------------------------------------------------

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        R, _ = cv2.Rodrigues(rvec)

        T_camera_board = np.eye(4)

        T_camera_board[:3, :3] = R
        T_camera_board[:3, 3] = tvec.flatten()

        return (
            T_camera_board,
            corners,
            ids,
            charuco_corners,
            charuco_ids
        )

    # ================================================================
    # Image callback
    # ================================================================

    def image_callback(
        self,
        astra_msg,
        astra_info,
        kinect_msg,
        kinect_info
    ):

        try:
            astra_image = self.bridge.imgmsg_to_cv2(
                astra_msg,
                desired_encoding='bgr8'
            )

            kinect_image = self.bridge.imgmsg_to_cv2(
                kinect_msg,
                desired_encoding='bgr8'
            )

        except Exception as e:
            self.get_logger().error(
                f'Image conversion failed: {e}'
            )
            return

        K_astra = np.array(
            astra_info.k,
            dtype=np.float64
        ).reshape(3, 3)

        D_astra = np.array(
            astra_info.d,
            dtype=np.float64
        )

        K_kinect = np.array(
            kinect_info.k,
            dtype=np.float64
        ).reshape(3, 3)

        D_kinect = np.array(
            kinect_info.d,
            dtype=np.float64
        )

        astra_result = self.detect_board(
            astra_image,
            K_astra,
            D_astra
        )

        kinect_result = self.detect_board(
            kinect_image,
            K_kinect,
            D_kinect
        )

        if astra_result is None or kinect_result is None:

            astra_display = astra_image.copy()
            kinect_display = kinect_image.copy()

            astra_text = "CHARUCO DETECTED" if astra_result is not None else "CHARUCO NOT DETECTED"
            kinect_text = "CHARUCO DETECTED" if kinect_result is not None else "CHARUCO NOT DETECTED"

            cv2.putText(
                astra_display, astra_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if astra_result is not None else (0, 0, 255),
                2
            )

            cv2.putText(
                kinect_display, kinect_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if kinect_result is not None else (0, 0, 255),
                2
            )

            cv2.imshow('Astra', astra_display)
            cv2.imshow('Kinect', kinect_display)

            cv2.waitKey(1)
            return

        T_A_B = astra_result[0]
        T_K_B = kinect_result[0]

        # ------------------------------------------------------------
        # Same physical board:
        #
        # T_A_B = Astra <- Board
        # T_K_B = Kinect <- Board
        #
        # Therefore:
        #
        # T_A_K = T_A_B * inverse(T_K_B)
        # ------------------------------------------------------------

        T_A_K = T_A_B @ np.linalg.inv(T_K_B)

        # ------------------------------------------------------------
        # Draw detections
        # ------------------------------------------------------------

        astra_display = astra_image.copy()
        kinect_display = kinect_image.copy()

        cv2.aruco.drawDetectedMarkers(
            astra_display,
            astra_result[1],
            astra_result[2]
        )

        cv2.aruco.drawDetectedMarkers(
            kinect_display,
            kinect_result[1],
            kinect_result[2]
        )

        cv2.aruco.drawDetectedCornersCharuco(
            astra_display,
            astra_result[3],
            astra_result[4]
        )

        cv2.aruco.drawDetectedCornersCharuco(
            kinect_display,
            kinect_result[3],
            kinect_result[4]
        )

        cv2.putText(
            astra_display,
            'CHARUCO DETECTED',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            kinect_display,
            'CHARUCO DETECTED',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            'Astra',
            astra_display
        )

        cv2.imshow(
            'Kinect',
            kinect_display
        )

        key = cv2.waitKey(1) & 0xFF

        # ------------------------------------------------------------
        # SPACE -> save sample
        # ------------------------------------------------------------

        if key == ord(' '):

            self.samples.append(T_A_K.copy())

            self.get_logger().info(
                f'Saved calibration sample '
                f'{len(self.samples)}'
            )

            self.print_transform(
                'Astra <- Kinect',
                T_A_K
            )

        # ------------------------------------------------------------
        # S -> calculate final
        # ------------------------------------------------------------

        if key == ord('s'):

            if len(self.samples) < 5:

                self.get_logger().warn(
                    'Need at least 5 samples.'
                )

            else:

                self.calculate_final_transform()

        # ------------------------------------------------------------
        # Q -> quit
        # ------------------------------------------------------------

        if key == ord('q'):

            rclpy.shutdown()

    # ================================================================
    # Final calibration
    # ================================================================

    def calculate_final_transform(self):

        self.get_logger().info(
            f'Calculating transform from '
            f'{len(self.samples)} samples...'
        )

        # ------------------------------------------------------------
        # Average translation
        # ------------------------------------------------------------

        translations = np.array([
            T[:3, 3]
            for T in self.samples
        ])

        translation = np.mean(
            translations,
            axis=0
        )

        # ------------------------------------------------------------
        # Average rotations using quaternion averaging
        # ------------------------------------------------------------

        quaternions = []

        for T in self.samples:

            H = np.eye(4)
            H[:3, :3] = T[:3, :3]

            q = quaternion_from_matrix(H)

            # Keep quaternion signs consistent
            if len(quaternions) > 0:

                if np.dot(q, quaternions[0]) < 0:
                    q = -q

            quaternions.append(q)

        q_mean = np.mean(
            np.array(quaternions),
            axis=0
        )

        q_mean /= np.linalg.norm(q_mean)

        # ------------------------------------------------------------
        # Quaternion -> rotation matrix
        # ------------------------------------------------------------

        from tf_transformations import quaternion_matrix

        T_A_K = quaternion_matrix(q_mean)

        T_A_K[:3, 3] = translation

        # ------------------------------------------------------------
        # Existing Astra transform from robot TF
        # ------------------------------------------------------------

        try:

            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.astra_frame,
                rclpy.time.Time()
            )

        except Exception as e:

            self.get_logger().error(
                f'Could not find TF '
                f'{self.base_frame} -> {self.astra_frame}: {e}'
            )

            self.get_logger().error(
                'Make sure the Astra URDF/TF is running.'
            )

            return

        T_base_astra = self.transform_msg_to_matrix(
            tf_msg
        )

        # ------------------------------------------------------------
        # Final:
        #
        # base -> Astra -> Kinect
        #
        # ------------------------------------------------------------

        T_base_kinect = (
            T_base_astra @ T_A_K
        )

        self.print_transform(
            'FINAL base_link <- camera_base',
            T_base_kinect
        )

        self.save_yaml(
            T_base_kinect
        )

    # ================================================================
    # TF conversion
    # ================================================================

    def transform_msg_to_matrix(self, tf_msg):

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation

        T = cv2.Rodrigues(
            np.zeros(3)
        )[0]

        from tf_transformations import quaternion_matrix

        T = quaternion_matrix([
            q.x,
            q.y,
            q.z,
            q.w
        ])

        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z

        return T

    # ================================================================
    # Printing
    # ================================================================

    def print_transform(self, name, T):

        translation = T[:3, 3]

        from tf_transformations import quaternion_from_matrix

        q = quaternion_from_matrix(T)

        self.get_logger().info(
            f'\n{name}\n'
            f'  Translation [m]:\n'
            f'    x = {translation[0]:.5f}\n'
            f'    y = {translation[1]:.5f}\n'
            f'    z = {translation[2]:.5f}\n'
            f'  Quaternion:\n'
            f'    x = {q[0]:.6f}\n'
            f'    y = {q[1]:.6f}\n'
            f'    z = {q[2]:.6f}\n'
            f'    w = {q[3]:.6f}\n'
        )

    # ================================================================
    # YAML
    # ================================================================

    def save_yaml(self, T):

        from tf_transformations import quaternion_from_matrix

        q = quaternion_from_matrix(T)

        data = {
            'parent_frame': self.base_frame,
            'child_frame': self.kinect_frame,

            'translation': {
                'x': float(T[0, 3]),
                'y': float(T[1, 3]),
                'z': float(T[2, 3])
            },

            'rotation': {
                'x': float(q[0]),
                'y': float(q[1]),
                'z': float(q[2]),
                'w': float(q[3])
            }
        }

        with open(
            self.output_file,
            'w'
        ) as f:

            yaml.dump(
                data,
                f,
                sort_keys=False
            )

        self.get_logger().info(
            f'Saved calibration to: '
            f'{self.output_file}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = KinectExtrinsicCalibrator()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()