import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_tb3_vision_maze = get_package_share_directory('tb3_vision_maze')
    
    # Simulation Launch
    simulation_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_vision_maze, 'launch', 'simulation.launch.py')
        )
    )

    # Vision Node
    vision_node = Node(
        package='tb3_vision_maze',
        executable='vision_node',
        name='vision_node',
        output='screen'
    )
    
    # Training Node (TD3)
    # We run this in a separate terminal usually to see logs clearly, but can launch here.
    # We use xterm to spawn a new window for the training logs if possible, 
    # but for simplicity in this environment we just launch it.
    train_node = Node(
        package='tb3_vision_maze',
        executable='train_node',
        name='train_node',
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        simulation_cmd,
        vision_node,
        train_node
    ])
