# AI-First Maze Solver (Deep Learning Portfolio)

This project implements a Deep Reinforcement Learning (DRL) agent to control a TurtleBot3 robot in a custom maze environment. We used the **TD3 (Twin Delayed DDPG)** algorithm because it handles continuous steering and velocity commands much better than discrete methods like DQN.

The main challenge was the sparse reward problem—the robot wouldn't find the exit by random exploration. To fix this, we implemented a **Teacher-Student** training loop. A geometry-based "Expert" drives the robot for the first few episodes to fill the replay buffer with good data, and then the Neural Network (Student) takes over to refine the policy.

**Key Technical Details:**
*   **Framework:** ROS 2 Humble + Gazebo
*   **Input:** 24-sector LiDAR scan + Relative Goal Vector (Distance/Angle to exit)
*   **Output:** Linear Velocity (0.0 - 0.5 m/s) and Angular Velocity.
*   **Safety:** We added a reactive safety layer in the environment to override the network if proper front-collision limits are breached.
![Maze Solver Demo](demo_video.webm)
*(See `demo_video.webm` in the file list for a full successful run)*

## 🚀 Key Featuresdits
This works was done as a Case Study for ROS 2.

*   **Althaf Ahamed (Lead AI Engineer):**
    *   Development of the TD3 Agent and Network Architecture.
    *   Implementation of the Teacher-Student training logic (DAGGER).
    *   Reward function tuning and state space optimization.

*   **Moazahmed:**
    *   World Design and Gazebo simulation setup.
    *   ROS 2 integration and Motion Planning baselines.

## Running the Code

### 1. Requirements
*   ROS 2 Humble
*   Gazebo 11
*   PyTorch

### 2. Setup
```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 3. Running the AI Demo
We have a launch file that runs the pre-trained Expert policy:
```bash
ros2 launch tb3_vision_maze deep_learning_demo.launch.py
```

### 4. Training
If you want to view the training process:
```bash
ros2 launch tb3_vision_maze train_maze.launch.py
```
