import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

class MotionNode(Node):
    def __init__(self):
        super().__init__('motion_node')

        self.planner_cmd = 'STOP'
        self.front_blocked = False
        self.exit_reached = False
        
        # Distance measurements
        self.min_left = 999.0
        self.min_front = 999.0
        self.min_right = 999.0

        # Subscriptions
        self.create_subscription(String, '/planner_cmd', self.planner_cb, 10)
        self.create_subscription(String, '/maze_state', self.exit_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        # Publisher
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_pub = self.create_publisher(String, '/scan_state', 10)

        # Control loop
        self.timer = self.create_timer(0.05, self.control_loop) # 20Hz

        self.get_logger().info("Motion node: safe distance + exit stop + Wall Follower enabled")

    def planner_cb(self, msg):
        self.planner_cmd = msg.data

    def exit_cb(self, msg):
        if msg.data == "exit_visible":
            self.exit_reached = True

    def scan_cb(self, msg):
        """
        Analyze ranges to detect walls: Left, Front, Right.
        Publishes state string: 'L1_F1_R1' (1=Free, 0=Blocked)
        Threshold: 0.6 meters
        """
        threshold = 0.6  
        
        # Helper to find min distance in an arc
        def get_min_in_sector(start_angle_deg, end_angle_deg):
            min_d = 999.0
            
            # Angles in ROS LaserScan are usually -pi to pi (or 0 to 2pi). 
            # msg.angle_min often -3.14, increment 0.017.
            # We map deg to index.
            
            start_rad = start_angle_deg * (math.pi / 180.0)
            end_rad = end_angle_deg * (math.pi / 180.0)
            
            # Create a list of indices to check
            # We must be careful with wrap around if needed, but here sectors are well defined if -180 to 180
            
            start_idx = int((start_rad - msg.angle_min) / msg.angle_increment)
            end_idx = int((end_rad - msg.angle_min) / msg.angle_increment)
            
            # Swap if start > end (should not happen if we pass correct range)
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
                
            # Clamp
            start_idx = max(0, start_idx)
            end_idx = min(len(msg.ranges)-1, end_idx)
            
            for i in range(start_idx, end_idx + 1):
                d = msg.ranges[i]
                if msg.range_min < d < msg.range_max:
                    if d < min_d: min_d = d
            return min_d

        # Front cone: -20 to +20 deg
        self.min_front = get_min_in_sector(-20, 20)

        # Left cone: +70 to +110 deg
        self.min_left = get_min_in_sector(70, 110)
            
        # Right cone: -110 to -70 deg
        self.min_right = get_min_in_sector(-110, -70)

        l_state = '1' if self.min_left > threshold else '0'
        f_state = '1' if self.min_front > threshold else '0'
        r_state = '1' if self.min_right > threshold else '0'

        scan_state = f"L{l_state}_F{f_state}_R{r_state}"
        self.scan_pub.publish(String(data=scan_state))
        
        # Keep basic safety stop
        if self.min_front < 0.35:
            self.front_blocked = True
        else:
            self.front_blocked = False

    def control_loop(self):
        cmd = Twist()

        # ---------- FINAL STOP AT EXIT ----------
        if self.exit_reached:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub.publish(cmd)
            self.get_logger().info("EXIT REACHED - HALTING")
            return

        # ---------- SAFETY OVERRIDE ----------
        if self.front_blocked and self.planner_cmd != 'STOP':
            # Emergency rotation if too close
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5 # Turn Right (Left Hand Rule safety)
            self.pub.publish(cmd)
            return

        # ---------- ACTION EXECUTION ----------
        if self.planner_cmd == 'FOLLOW_LEFT':
            # --- Continuous Left Wall Follower ---
            # Desired distance to left wall: 0.5m
            desired_dist = 0.5
            
            # 1. Collision Ahead? -> Turn Right Sharp
            if self.min_front < 0.6:
                cmd.linear.x = 0.1
                cmd.angular.z = -0.8 # Turn Right
            
            # 2. Wall Lost? (Opening on Left) -> Seek Left
            elif self.min_left > 0.8:
                cmd.linear.x = 0.15
                cmd.angular.z = 0.6 # Turn Left to find wall/corner
                
            # 3. Maintain Distance (P-Controller)
            else:
                error = desired_dist - self.min_left
                # If error > 0 (Too Far): Turn Left (+)
                # If error < 0 (Too Close): Turn Right (-)
                
                # Gain
                Kp = 3.0
                cmd.angular.z = Kp * error
                
                # Clip angular velocity
                cmd.angular.z = max(min(cmd.angular.z, 0.8), -0.8)
                
                # Linearity: Slow down if turning fast
                cmd.linear.x = 0.2 - (abs(cmd.angular.z) * 0.1)
                if cmd.linear.x < 0.05: cmd.linear.x = 0.05
    
        elif self.planner_cmd == 'MOVE_FORWARD':
            cmd.linear.x = 0.2
        elif self.planner_cmd == 'TURN_LEFT':
            cmd.angular.z = 0.5
        elif self.planner_cmd == 'TURN_RIGHT':
            cmd.angular.z = -0.5
        elif self.planner_cmd == 'STOP':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        self.pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    rclpy.spin(node)
    rclpy.shutdown()
