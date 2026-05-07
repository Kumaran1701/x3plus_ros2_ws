import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.substitutions import Command, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    x3plus_description = get_package_share_directory("x3plus_description")

    model_arg = DeclareLaunchArgument(name="model", default_value=os.path.join(
                                        x3plus_description, "urdf", "x3plus.gazebo.urdf.xacro"
                                        )
                                      )
    
    
    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            str(Path(x3plus_description).parent.resolve())
            ]
        )
    
    robot_description = ParameterValue(Command([
            "xacro ",
            LaunchConfiguration("model"),
        ]),
        value_type=str
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description,
                     "use_sim_time": True}]
    )

    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory("ros_gz_sim"), "launch"), "/gz_sim.launch.py"]),
                launch_arguments=[
                    ("gz_args", [" -v 4", " -r", " empty.sdf"]
                    )
                ]
             )

    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description",
                   "-name", "x3plus"],
    )

    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/world/empty/model/x3plus/link/arm_link4/sensor/rgb_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/world/empty/model/x3plus/link/arm_link4/sensor/rgb_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/world/empty/model/x3plus/link/base_footprint/sensor/rgbd_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/world/empty/model/x3plus/link/base_footprint/sensor/rgbd_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/world/empty/model/x3plus/link/base_footprint/sensor/rgbd_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/world/empty/model/x3plus/link/base_footprint/sensor/rgbd_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        remappings=[
            ("/world/empty/model/x3plus/link/arm_link4/sensor/rgb_camera/camera_info", "/camera_arm/camera_info"),
            ("/world/empty/model/x3plus/link/arm_link4/sensor/rgb_camera/image", "/camera_arm/image_raw"),
            ("/world/empty/model/x3plus/link/base_footprint/sensor/rgbd_camera/camera_info", "/depth_camera/camera_info"),
            ("/world/empty/model/x3plus/link/base_footprint/sensor/rgbd_camera/depth_image", "/depth_camera/depth_image_raw"),
            ("/world/empty/model/x3plus/link/base_footprint/sensor/rgbd_camera/image", "/depth_camera/image_raw"),
            ("/world/empty/model/x3plus/link/base_footprint/sensor/rgbd_camera/points", "/depth_camera/points"),
            ("/imu", "/imu"),
            ("/scan", "/scan"),
            ("/scan/points", "/scan/points")
        ]
    )


    return LaunchDescription([
        model_arg,
        gazebo_resource_path,
        gazebo,
        gz_spawn_entity,
        gz_ros2_bridge,
        robot_state_publisher_node,
    ])