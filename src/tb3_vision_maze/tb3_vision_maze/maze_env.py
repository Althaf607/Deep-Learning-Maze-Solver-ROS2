import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import numpy as np
import math
import time

from std_msgs.msg import String

class MazeEnv(Node):
    def __init__(self):
        super().__init__('maze_env')
        
        # --- Config ---
        # State: 24 Laser + 2 Action + 2 Goal Vector (Dist, Angle) = 28
        self.state_dim = 28 
        self.action_dim = 2  # linear, angular
        
        # Waypoints to guide the robot out of the maze (prevent local minima)
        # Refined coordinates based on World Analysis:
        # Gap exists between wall_v2 (x=2, y>=0) and wall_h2 (x>-2, y=-1). Gap at y=[-1, 0].
        self.waypoints = [
            [2.5, -0.6],   # Point 1: Align with gap (Go South-East) - Shifted x=2.5 to clear Wall V2(x=2)
            [3.5, -0.6],   # Point 2: Through the gap (Move East)
            [3.5, 4.5],    # Point 3: Up the right corridor (Move North) - Shifted y=4.5 (Center of Top Gap y=4-5)
            [0.0, 4.5],    # Point 4: Align with Exit (Move West) - Shifted y=4.5
            [0.0, 5.5]     # EXIT
        ]
        self.current_waypoint_idx = 0
        self.target_coords = self.waypoints[self.current_waypoint_idx]
        
        # --- ROS Communication ---
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_vision = self.create_subscription(String, '/maze_state', self.vision_callback, 10) # NEW
        
        self.reset_proxy = self.create_client(Empty, '/reset_simulation')
        
        # --- Data Buffers ---
        self.scan_data = None
        self.odom_data = None
        self.current_pose = None
        self.prev_dist = 0.0
        self.current_waypoint_idx = 0
        self.exit_visible = False # NEW
        
        self.get_logger().info("Maze Environment Initialized with Vision Support")

    def scan_callback(self, msg):
        self.scan_data = msg

    def odom_callback(self, msg):
        self.odom_data = msg
        self.current_pose = msg.pose.pose

    def vision_callback(self, msg):
        if msg.data == "exit_visible":
            self.exit_visible = True
        else:
            self.exit_visible = False

    def step(self, action):
        """
        Execute action and return (state, reward, done, info)
        Action: [linear(0~0.2), angular(-1~1)]
        """
        # 1. Execute Action
        # Safety Layer & Action Execution
        scan = np.array(self.get_scan_ranges())
        # Correct Front Sector Check (Indices 0, 1, 23)
        # Note: scan has 24 elements. 0 is front.
        min_front = min(scan[0], scan[1], scan[23]) * 3.5
        
        real_action = list(action)
        
        # Virtual Bumper: Prevent forward collision
        if min_front < 0.25 and real_action[0] > 0: # Reduced safety margin slightly to allow close maneuvers
            real_action[0] = 0.0 # Stop linear
            
        vel_cmd = Twist()
        vel_cmd.linear.x = float(np.clip(real_action[0], 0.0, 0.5)) # Overclocked Speed
        vel_cmd.angular.z = float(np.clip(real_action[1], -1.0, 1.0))
        self.pub_cmd_vel.publish(vel_cmd)
        
        # Wait for simulation (simplified sync)
        time.sleep(0.002) # Max Speed: 500Hz logic loop possible
        
        # 2. Get State
        state = self.get_state(action)
        
        # 3. Calculate Reward
        reward, done = self.calculate_reward(state, action)
        
        return state, reward, done, {}

    def reset(self):
        """
        Reset simulation and return initial state
        """
        # Stop robot
        self.pub_cmd_vel.publish(Twist())
        
        # Call Reset Service
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset service not available, waiting...')
        self.reset_proxy.call_async(Empty.Request())
        
        time.sleep(1.0) # Wait for reset stability
        
        # Wait for valid scan data
        while self.scan_data is None:
            self.get_logger().info("Waiting for LaserScan data...")
            rclpy.spin_once(self, timeout_sec=0.1)
            
        # Initialize prev_dist
        self.current_waypoint_idx = 0
        self.target_coords = self.waypoints[self.current_waypoint_idx]
        self.exit_visible = False
        
        if self.current_pose:
            x = self.current_pose.position.x
            y = self.current_pose.position.y
            self.prev_dist = math.sqrt((x - self.target_coords[0])**2 + (y - self.target_coords[1])**2)
        
        # Return initial state (zeros for action)
        return self.get_state([0.0, 0.0])

    def get_state(self, action):
        """
        State: [24 laser readings, current_linear_vel, current_angular_vel, dist_to_goal, angle_to_goal]
        """
        if self.scan_data is None or self.current_pose is None:
            return np.zeros(self.state_dim)
            
        # Downsample 360 -> 24
        ranges = self.scan_data.ranges
        num_readings = len(ranges)
        # Handle Inf/Nan
        ranges = [r if not math.isinf(r) and not math.isnan(r) else 3.5 for r in ranges]
        
        # Sample 24 points evenly
        step = num_readings // 24
        reduced_ranges = []
        for i in range(24):
            idx = (i * step) % num_readings
            r = ranges[idx]
            reduced_ranges.append(min(r, 3.5)) # Clip max range
            
        # Normalize Laser
        state = np.array(reduced_ranges) / 3.5
        
        # Append Action
        state = np.append(state, action)
        
        # Append Goal Vector (Distance, Angle)
        # Robot Pose
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        
        # Robot Orientation (Quaternion to Euler Yaw)
        q = self.current_pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Target
        tx = self.target_coords[0]
        ty = self.target_coords[1]
        
        dist_to_target = math.sqrt((tx - rx)**2 + (ty - ry)**2)
        angle_to_target = math.atan2(ty - ry, tx - rx) - yaw
        
        # Normalize Angle to [-pi, pi]
        while angle_to_target > math.pi: angle_to_target -= 2*math.pi
        while angle_to_target < -math.pi: angle_to_target += 2*math.pi
        
        # Normalize state features
        # Dist max approx 10m
        dist_norm = min(dist_to_target, 10.0) / 10.0
        angle_norm = angle_to_target / math.pi # [-1, 1]
        
        state = np.append(state, [dist_norm, angle_norm])
        
        return state
        
    def get_scan_ranges(self):
        if self.scan_data is None: return [3.5]*24
        ranges = self.scan_data.ranges
        ranges = [r if not math.isinf(r) and not math.isnan(r) else 3.5 for r in ranges]
        step = len(ranges) // 24
        return [min(ranges[(i*step)%len(ranges)], 3.5) for i in range(24)]

    def calculate_reward(self, state, action):
        reward = 0.0
        done = False
        
        # 1. VISION CHECK (Primary Goal)
        if self.exit_visible:
            reward = 200.0
            done = True
            self.get_logger().info("EXIT VISIBLE! GOAL REACHED (VISION)!")
            return reward, done

        # 2. Collision Check (min laser range)
        min_range = np.min(state[:24]) * 3.5
        if min_range < 0.18: 
            reward = -100.0
            done = True
            return reward, done
            
        # 3. Waypoint Check (Backup if Vision fails or for guidance)
        if self.current_pose:
            x = self.current_pose.position.x
            y = self.current_pose.position.y
            dist_to_exit = math.sqrt((x - self.target_coords[0])**2 + (y - self.target_coords[1])**2)
            
            # Check if reached current waypoint
            if dist_to_exit < 0.8: # Increased from 0.5 to prevent looping
                # If valid waypoint, switch to next
                if self.current_waypoint_idx < len(self.waypoints) - 1:
                    self.current_waypoint_idx += 1
                    self.target_coords = self.waypoints[self.current_waypoint_idx]
                    reward += 50.0 # Reward for reaching waypoint
                    self.get_logger().info(f"WAYPOINT {self.current_waypoint_idx} REACHED!")
                else: 
                    # Last waypoint is exit
                    reward = 100.0
                    done = True
                    self.get_logger().info("GOAL REACHED (COORDS)!")
                    return reward, done
        
        # 4. Basic Navigation Reward
        reward += action[0] * 2.0  # Encourage moving forward
        reward -= abs(action[1]) * 0.1 # Discourage excessive spinning, but allow turning
        
        # 5. Dense Reward: Distance Delta
        if self.current_pose:
            x = self.current_pose.position.x
            y = self.current_pose.position.y
            dist_to_exit = math.sqrt((x - self.target_coords[0])**2 + (y - self.target_coords[1])**2)
            
            dist_delta = self.prev_dist - dist_to_exit
            reward += dist_delta * 50.0 
            
            self.prev_dist = dist_to_exit
        
        return reward, done
