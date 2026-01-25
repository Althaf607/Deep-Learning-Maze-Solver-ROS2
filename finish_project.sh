#!/bin/bash
echo "Building Project..."
colcon build --packages-select tb3_vision_maze
source install/setup.bash

echo "Launching TB3 Vision Maze Solver (TD3 + Teacher Policy)..."
echo "The robot will start solving the maze immediately."
ros2 launch tb3_vision_maze train_maze.launch.py
