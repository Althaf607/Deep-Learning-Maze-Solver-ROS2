# AI-First Maze Solver — TurtleBot3 Deep Reinforcement Learning (ROS 2)

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![Gazebo](https://img.shields.io/badge/Simulation-Gazebo11-orange)
![PyTorch](https://img.shields.io/badge/AI-PyTorch-red)
![Algorithm](https://img.shields.io/badge/RL-TD3-green)
![Robot](https://img.shields.io/badge/Platform-TurtleBot3-lightgrey)

Deep Reinforcement Learning navigation system for TurtleBot3 using **ROS 2 Humble**, **Gazebo**, and **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**.

The robot learns to autonomously solve maze environments using LiDAR perception and continuous velocity control — without classical global planners.

---

# Demo

![Demo](assets/demo.gif)

> Place your demo GIF inside an `assets/` folder in the repository and name it `demo.gif`.

---

# Overview

This project explores **AI-first navigation**, where a neural network learns motion behaviour directly from sensor observations.

Instead of relying on prebuilt maps or deterministic planners, the robot learns a continuous control policy through interaction with the environment.

Core concepts:

* Model-Free Deep Reinforcement Learning
* Continuous control (v, w)
* Teacher–Student training strategy
* Modular ROS 2 architecture

---

# ⚙️ System Architecture

```
LiDAR / Vision
       ↓
State Encoder
       ↓
TD3 Actor Network
       ↓
Safety Layer
       ↓
ROS2 /cmd_vel
       ↓
TurtleBot3 (Gazebo)
```

ROS 2 Nodes:

* vision_node — perception & state extraction
* planner_node — TD3 inference
* motion_node — robot velocity execution

---

# Key Features

✔ Deep Reinforcement Learning navigation
✔ Continuous control using TD3
✔ Teacher-Student (DAGGER-inspired) training
✔ ROS 2 Humble integration
✔ Gazebo maze simulation
✔ Modular node architecture
✔ Pre-trained demo mode
✔ Full training pipeline

---

# System Requirements

## Software

* Ubuntu 22.04 LTS
* ROS 2 Humble Desktop
* Gazebo Classic 11
* Python 3.10+

## Python Dependencies

```
pip install torch numpy
```

---

# Installation

## Build Workspace

```
colcon build --symlink-install
source install/setup.bash
```

---

# Quick Start — AI Demo

```
ros2 launch tb3_vision_maze deep_learning_demo.launch.py
```

The robot autonomously navigates the maze using the TD3 policy.

---

# Training

```
ros2 launch tb3_vision_maze train_maze.launch.py
```

Training workflow:

1. Teacher controller generates initial trajectories
2. Replay buffer initialized
3. TD3 student learns continuous policy
4. Policy improves through exploration

---

# Full Simulation Setup (Multi-Terminal Workflow)

## Terminal 0 — Cleanup

```
pkill gzserver
pkill gzclient
```

---

## Terminal 1 — Gazebo Server

```
source /opt/ros/humble/setup.bash
export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib
export GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models

gzserver ~/tb3_project_ws/worlds/maze_world.world \
  -s libgazebo_ros_init.so \
  -s libgazebo_ros_factory.so
```

---

## Terminal 2 — Gazebo GUI

```
gzclient
```

---

## Terminal 3 — Spawn Robot

```
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo spawn_turtlebot3.launch.py
```

---

## Terminal 4 — Vision Node

```
ros2 run tb3_vision_maze vision_node
```

---

## Terminal 5 — Planner Node

```
ros2 run tb3_vision_maze planner_node
```

---

## Terminal 6 — Motion Node

```
ros2 run tb3_vision_maze motion_node
```

---

# Deep Reinforcement Learning — TD3

File: `td3_agent.py`

TD3 improves stability over DDPG with:

* Twin Critic Networks
* Delayed Actor Updates
* Target Policy Smoothing

---

## TD3 Update Principle

```
y = r + gamma * min(Q1, Q2)
```

---

# Network Architecture

## Actor Network

* Input: 28
* Hidden Layers: 400 → 300 (ReLU)
* Output: 2 (Tanh)

## Critic Network

* Input: 30 (State + Action)
* Hidden Layers: 400 → 300
* Output: 1

---

# Hyperparameters

| Parameter         | Value |
| ----------------- | ----- |
| Batch Size        | 100   |
| Gamma             | 0.99  |
| Tau               | 0.005 |
| Policy Noise      | 0.2   |
| Noise Clip        | 0.5   |
| Exploration Noise | 0.1   |
| Actor LR          | 3e-4  |
| Critic LR         | 3e-4  |

---

# Training Pipeline

```
Teacher Controller
        ↓
Replay Buffer
        ↓
TD3 Student Policy
        ↓
Continuous Improvement
```

---

# ROS 2 Node Graph (Conceptual)

```
/scan  → vision_node → /maze_state
                     ↓
                planner_node → /planner_cmd
                               ↓
                          motion_node → /cmd_vel
```

---

# Skills Demonstrated (For Recruiters)

* ROS 2 System Architecture
* Deep Reinforcement Learning (TD3)
* Autonomous Robotics
* Gazebo Simulation
* Continuous Control Systems
* Python + PyTorch
* AI for Navigation
* Robot Perception Pipelines

---

# Project Structure

```
tb3_project_ws/
├── models/
├── worlds/
├── launch/
├── scripts/
├── td3_agent.py
└── tb3_vision_maze/
```

---

# Contributors

**Althaf Ahamed**
Lead AI Engineer — TD3 design, training pipeline, reward engineering

**Moazahmed**
Simulation setup, ROS 2 integration, perception system

---

#  References

* Fujimoto et al. — TD3 Algorithm
* ROS 2 Humble Documentation
* TurtleBot3 e-Manual
* Gazebo Simulation Docs

---

# Author

Althaf
MSc Intelligent Robotics — 2025/26
