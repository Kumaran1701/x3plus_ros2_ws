# ROS2 Launch file for Yahboom X3Plus robot bringup.
# Provides base drivers, state estimation, TFs, and optional to enable camera-arm stream. 

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # Load URDF and EKF configuration paths.
    pkg_path_description = get_package_share_directory('x3plus_description')
    urdf_file = os.path.join(pkg_path_description, 'urdf', 'x3plus.gazebo.urdf.xacro')

    pkg_path_bringup = get_package_share_directory('x3plus_bringup')
    ekf_config = os.path.join(pkg_path_bringup, 'config', 'x3plus_ekf.yaml')

    # Optional: enable/disbale camera-arm stream
    enable_camera_arm = DeclareLaunchArgument(
        'enable_camera_arm',
        default_value='false'
    )

    # Robot Description (URDF -> XML)
    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )

    # Base hardware driver: motors, IMU, and other sensors.
    driver_node = Node(
        package='x3plus_bringup',
        executable='x3plus_driver.py',
        name='driver_node',
        output='screen'
    )

    # Wheel odometry publisher.
    odom_node = Node(
        package='x3plus_bringup',
        executable='odom_node.py',
        name='odometry_publisher_node',
        output='screen'
    )

    # IMU filtering (Madgwick)
    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter_madgwick',
        output='screen',

        parameters=[{
            'fixed_frame': 'base_link',
            'use_mag': False,
            'publish_tf': False,
            'use_magnetic_field_msg': False,
            'world_frame': 'enu',
            'orientation_stddev': 0.05,
            'angular_scale': 1.03,
        }],

        remappings=[
            ('imu/data_raw', '/imu/raw'),
            ('imu/mag', '/mag/raw'),
        ]
    )

    # EKF state estimator (fuses odom + IMU)
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config]
    )

    # Optional: Node to enable/disable camera-arm stream
    camera_arm_node = Node(
        package='x3plus_camera_arm',
        executable='x3plus_camera_arm.py',
        name='camera_arm_node',
        output='screen',
        condition=IfCondition(
            LaunchConfiguration('enable_camera_arm')
        )
    )

    # Publishes TF from URDF
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'robot_description': robot_description}
        ],
        output='screen'
    )

    # Static TF: base_link(robot) -> camera_base(azure_kinect)
    kinect_extrinsic_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='kinect_extrinsic_tf',
        arguments=[
            '0.404',
            '0.020',
            '0.437',
            '0.000',
            '0.000',
            '0.000',
            '1.000',
            'base_link',
            'camera_base'
        ],
        output='screen'
    )
    

    return LaunchDescription([
        enable_camera_arm,
        driver_node,
        odom_node,
        imu_filter_node,
        ekf_node,
        camera_arm_node,
        robot_state_publisher_node,
        kinect_extrinsic_tf_node,
    ])