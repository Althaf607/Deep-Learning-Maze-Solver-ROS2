import rclpy
from rclpy.node import Node
import numpy as np
import torch
import os
import sys
import glob

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maze_env import MazeEnv
from td3_agent import TD3

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        
        self.state_dim = 28
        self.action_dim = 2
        self.max_action = 1.0
        
        self.env = MazeEnv()
        self.policy = TD3(self.state_dim, self.action_dim, self.max_action)
        
        # Load latest actor model
        models_dir = "./models_safe_success"
        if not os.path.exists(models_dir):
            self.get_logger().error("No models directory found! Train first.")
            return

        list_of_files = glob.glob(f'{models_dir}/actor_*.pth')
        if not list_of_files:
            self.get_logger().error("No model files found in ./models/")
            return
            
        latest_file = max(list_of_files, key=os.path.getctime)
        self.get_logger().info(f"Loading model: {latest_file}")
        
        self.policy.actor.load_state_dict(torch.load(latest_file))
        self.policy.actor.eval() # Set to evaluation mode
        
    def run_test(self):
        state = self.env.reset()
        done = False
        episode_reward = 0
        
        self.get_logger().info("Starting Evaluation Loop...")
        
        while rclpy.ok():
            rclpy.spin_once(self.env, timeout_sec=0.01)
            
            # Select action *without* noise
            action = self.policy.select_action(np.array(state))
            
            # Perform action
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            episode_reward += reward
            
            if done:
                self.get_logger().info(f"Episode Finished. Reward: {episode_reward:.3f}")
                state = self.env.reset()
                done = False
                episode_reward = 0

def main(args=None):
    rclpy.init(args=args)
    tester = TestNode()
    try:
        tester.run_test()
    except KeyboardInterrupt:
        pass
    tester.env.destroy_node()
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
