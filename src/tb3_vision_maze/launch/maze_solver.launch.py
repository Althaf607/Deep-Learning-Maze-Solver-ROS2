import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tb3_vision_maze',
            executable='vision_node',
            name='vision_node',
            output='screen'
        ),
        Node(
            package='tb3_vision_maze',
            executable='planner_node',
            name='planner_node',
            output='screen'
        ),
        Node(
            package='tb3_vision_maze',
            executable='motion_node',
            name='motion_node',
            output='screen'
        )
    ])
