from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    driver_node = Node(
        package='x3plus_bringup',
        executable='x3plus_driver.py',
        name='driver_node',
        output='screen'
    )

    teleop_node = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_keybaord',
        prefix='xterm -e',
        output='screen'
    )

    return LaunchDescription([
        driver_node,
        teleop_node
    ])