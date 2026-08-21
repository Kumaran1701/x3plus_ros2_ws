from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    pkg_path_description = get_package_share_directory('x3plus_description')
    urdf_file = os.path.join(pkg_path_description, 'urdf', 'x3plus.gazebo.urdf.xacro')

    pkg_path_bringup = get_package_share_directory('x3plus_bringup')
    ekf_config = os.path.join(pkg_path_bringup, 'config', 'x3plus_ekf.yaml')

    enable_camera_arm = DeclareLaunchArgument(
        'enable_camera_arm',
        default_value='false'
    )

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )
    
    driver_node = Node(
        package='x3plus_bringup',
        executable='x3plus_driver.py',
        name='driver_node',
        output='screen'
    )

    odom_node = Node(
        package='x3plus_bringup',
        executable='odom_node.py',
        name='odometry_publisher_node',
        output='screen'
    )

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

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config]
    )
    
    camera_arm_node = Node(
        package='x3plus_camera_arm',
        executable='x3plus_camera_arm.py',
        name='camera_arm_node',
        output='screen',
        condition=IfCondition(
            LaunchConfiguration('enable_camera_arm')
        )
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'robot_description': robot_description}
        ],
        output='screen'
    )

    kinect_extrinsic_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='kinect_extrinsic_tf',
        arguments=[
            '0.00786856847',
            '-0.04501331105',
            '0.44389381664',
            '-0.50269719517',
            '0.49786415606',
            '-0.49916166179',
            '0.50026437758',
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