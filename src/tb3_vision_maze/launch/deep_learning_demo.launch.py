import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_tb3_vision_maze = get_package_share_directory('tb3_vision_maze')
    
    # World File
    world_path = '/home/althaf/Case_Study/tb3-vision-maze-project-main/maze_world.world'

    # 1. Start Gazebo World
    gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_path}.items(),
    )

    # 2. Spawn Robot at Start (0, 0)
    spawn_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, 'launch', 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pos': '0.0',
            'y_pos': '0.0',
            'z_pos': '0.01'
        }.items()
    )

    # 3. Vision Node (Processes Camera -> Goal Vector)
    vision_node = Node(
        package='tb3_vision_maze',
        executable='vision_node',
        name='vision_node',
        output='screen'
    )
    
    # 4. Autonomous Navigation Node (Expert System)
    # Launches the robust controller that solves the maze.
    main_node = Node(
        package='tb3_vision_maze',
        executable='train_node',
        name='train_node',
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        gazebo_cmd,
        spawn_cmd, 
        vision_node,
        main_node
    ])
