from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    pkg_path = get_package_share_directory('x3plus_description')
    urdf_file = os.path.join(pkg_path, 'urdf', 'x3plus.gazebo.urdf.xacro')

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

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'robot_description': robot_description}
        ],
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen'
    )


    return LaunchDescription([
        driver_node,
        robot_state_publisher_node,
        rviz_node,
    ])