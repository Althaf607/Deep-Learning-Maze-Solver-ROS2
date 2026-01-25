import rclpy
from rclpy.node import Node
import numpy as np
import torch
import os
import sys
import time
import math

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maze_env import MazeEnv
from td3_agent import TD3, ReplayBuffer

class TrainingNode(Node):
    def load_robust(self, target_model, source_state_dict, input_offset=0):
        """
        Loads weights from source_state_dict into target_model, handling shape mismatches.
        Assumes mismatches are due to appended features in usage (e.g. State increased).
        For Critic, we need to be careful about where the mismatch is (State vs Action).
        """
        model_dict = target_model.state_dict()
        
        for name, param in source_state_dict.items():
            if name not in model_dict:
                continue
                
            local_param = model_dict[name]
            
            if local_param.shape == param.shape:
                # Perfect match
                model_dict[name].copy_(param)
            else:
                self.get_logger().warn(f"Shape mismatch for {name}: Checkpoint {param.shape} vs Current {local_param.shape}. Attempting adaptation...")
                
                # Check for Linear layer weight mismatch
                if len(param.shape) == 2 and len(local_param.shape) == 2:
                    # e.g. [400, 26] vs [400, 28]
                    # Assume rows match (units), cols differ (inputs)
                    if local_param.shape[0] == param.shape[0]:
                        # Copy common columns
                        min_cols = min(local_param.shape[1], param.shape[1])
                        
                        # Special handling for Critic (State + Action cat)
                        # If mismatch is in State, Action is shifted.
                        # We assume:
                        # Old: [State_Old (26) | Action (2)]
                        # New: [State_Old (26) | New_Feats (2) | Action (2)]
                        # We need to copy 0..25 -> 0..25 AND 26..27 -> 28..29
                        
                        # Heuristic: Detect if Action is at the end?
                        # Critic L1 input is state_dim + action_dim.
                        # Actor L1 input is state_dim.
                        
                        is_critic_l1 = "l1.weight" in name and local_param.shape[1] > self.state_dim
                        
                        if is_critic_l1:
                             # Critic Adaptation
                             # Copy State Part (First 26)
                             old_state_dim = param.shape[1] - self.action_dim
                             
                             # Copy State (0 -> old_state_dim)
                             local_param.data[:, :old_state_dim].copy_(param.data[:, :old_state_dim])
                             
                             # Zero out the NEW state features (indices old_state_dim : self.state_dim)
                             # This ensures the new features don't contribute noise initially
                             local_param.data[:, old_state_dim:self.state_dim].fill_(0.0)
                             
                             # Copy Action (old_state_dim -> end) to (new_state_dim -> end)
                             local_param.data[:, self.state_dim:].copy_(param.data[:, old_state_dim:])
                             
                             self.get_logger().info(f"Adapted Critic layer {name}: Copied State[:{old_state_dim}], Zeroed New Feats, Copied Action")
                        else:
                             # Actor or other layers (Append mode)
                             # Just copy the first N columns
                             local_param.data[:, :min_cols].copy_(param.data[:, :min_cols])
                             
                             # Zero out the rest (new columns)
                             if local_param.shape[1] > min_cols:
                                 local_param.data[:, min_cols:].fill_(0.0)
                                 self.get_logger().info(f"Adapted Layer {name}: Copied first {min_cols} cols, Zeroed rest")
                             else:
                                 self.get_logger().info(f"Adapted Layer {name}: Copied first {min_cols} columns")
                    else:
                         self.get_logger().error(f"Cannot adapt {name}: Rows mismatch.")
                else:
                    self.get_logger().error(f"Cannot adapt {name}: Not a 2D weight matrix.")
                    
    def __init__(self):
        super().__init__('training_node')
        
        # --- Hyperparameters ---
        self.state_dim = 28 # Updated for Goal Vector
        self.action_dim = 2
        self.max_action = 1.0 # Angular max, linear is clipped in env
        self.max_timesteps = 1000000 
        self.start_timesteps = 100000 # Extended Teacher Phase for robust demo
        self.batch_size = 100
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.expl_noise = 0.1 
        
        # --- Objects ---
        self.env = MazeEnv() 
        self.policy = TD3(self.state_dim, self.action_dim, self.max_action)
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
        
        # Create models dir
        if not os.path.exists("./models"):
            os.makedirs("./models")
            
        # --- Resume Logic ---
        self.start_episode = 0
        self.initial_timestep = 0
        
        try:
            # Check for existing models
            model_files = [f for f in os.listdir("./models") if f.startswith("actor_") and f.endswith(".pth")]
            if model_files:
                ep_nums = [int(f.split("_")[1].split(".")[0]) for f in model_files]
                if ep_nums:
                    latest_ep = max(ep_nums)
                    actor_path = f"./models/actor_{latest_ep}.pth"
                    critic_path = f"./models/critic_{latest_ep}.pth"
                    
                    if os.path.exists(actor_path) and os.path.exists(critic_path):
                        self.get_logger().info(f"FOUND CHECKPOINT: Resuming training from Episode {latest_ep}")
                        
                        # Robust Load
                        self.load_robust(self.policy.actor, torch.load(actor_path))
                        self.load_robust(self.policy.critic, torch.load(critic_path))
                        
                        # Sync target networks (since we only saved online networks)
                        self.policy.actor_target.load_state_dict(self.policy.actor.state_dict())
                        self.policy.critic_target.load_state_dict(self.policy.critic.state_dict())
                        
                        self.start_episode = latest_ep
                        
                        # Expert Demonstration Phase (DAGGER-like initialization)
                        # We use the corrected expert policy to fill the buffer with high-quality samples.
                        
                        current_steps = latest_ep * 2100 # Estimated steps
                        self.initial_timestep = current_steps
                        
                        # Expert Phase for Robust Feature Learning
                        self.start_timesteps = current_steps + 100000 
                             
                        self.get_logger().info(f"Initializing Expert Demonstration Phase (DAGGER) until timestep {self.start_timesteps}.")
                        
                        # Estimate timestep based on episode count. 
                        # If we are deep into training, ensure we skip teacher phase (start_timesteps).
                        # Assuming at least ~200 steps per episode roughly.
                        estimated_steps = latest_ep * 200
                        if estimated_steps > self.start_timesteps:
                             self.initial_timestep = estimated_steps
                        else:
                             # If we are resuming very early, maybe still in teacher phase, but let's be safe
                             self.initial_timestep = estimated_steps
                             
                        self.get_logger().info(f"Set initial timestep to {self.initial_timestep}")
        except Exception as e:
            self.get_logger().warn(f"Failed to resume from checkpoint: {e}")

        self.get_logger().info("Training Node Initialized")
        
    def train(self):
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = self.start_episode
        
        self.get_logger().info(f"Starting Training Loop from timestep {self.initial_timestep}...")
        
        for t in range(int(self.initial_timestep), int(self.max_timesteps)):
            episode_timesteps += 1
            
            # Spin 'env' node to get latest data
            rclpy.spin_once(self.env, timeout_sec=0.01)

            # Select action
            if t < self.start_timesteps: 
                # Expert Heuristic Controller (Data Collection)
                # Steering is guided by the goal vector with reactive obstacle avoidance.
                
                scan_ranges = self.env.get_scan_ranges()
                # Safety Checks: Front and Corner detection
                front_sector = [scan_ranges[0], scan_ranges[1], scan_ranges[2], 
                                scan_ranges[21], scan_ranges[22], scan_ranges[23]]
                min_front = min(front_sector) * 3.5
                min_left = min(scan_ranges[5], scan_ranges[6], scan_ranges[7]) * 3.5
                min_right = min(scan_ranges[17], scan_ranges[18], scan_ranges[19]) * 3.5
                
                angle_to_target = state[27] * math.pi # Un-normalize (was / pi)
                
                # Base Action: Steer to Target
                speed = 0.45 
                heading = angle_to_target * 1.5
                
                # Lane Centering Logic
                if min_left < 0.5:
                    heading -= 0.8 * (0.6 - min_left) 
                if min_right < 0.5:
                    heading += 0.8 * (0.6 - min_right) 
                
                # Obstacle Avoidance Override (Front + CORNERS)
                if min_front < 0.85:  # Increased lookahead for high speed braking
                    speed = 0.0
                    # Turn away from obstacle
                    if min_left < min_right:
                        heading = -1.0 # Turn Right
                    else:
                        heading = 1.0 # Turn Left
                        
                # REVERSE Logic (Anti-Stuck)
                if min_front < 0.25:
                    speed = -0.15
                    heading = -heading # Reverse direction
                
                # Add Noise for robustness
                heading += np.random.normal(0, 0.1)
                
                heading = float(np.clip(heading, -1.0, 1.0))
                speed = float(np.clip(speed, -0.2, 0.5))
                
                action = np.array([speed, heading])
                
            else:
                action = (
                    self.policy.select_action(np.array(state))
                    + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                ).clip(-self.max_action, self.max_action)

            # Perform action
            next_state, reward, done, _ = self.env.step(action) 
            
            # Store data in replay buffer
            done_bool = float(done) if episode_timesteps < 500 else 0 
            self.replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting enough data
            if t >= self.start_timesteps:
                self.policy.train(self.replay_buffer, self.batch_size, self.discount, self.tau, self.policy_noise, self.noise_clip, self.policy_freq)

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                self.get_logger().info(f"Ep: {episode_num+1} | T: {episode_timesteps} | R: {episode_reward:.3f} | Last Act: {action} | Mode: {'TEACHER' if t < self.start_timesteps else 'AGENT'}")
                
                # Save model every 20 episodes (more frequent)
                if (episode_num + 1) % 20 == 0:
                    self.get_logger().info("Saving model...")
                    torch.save(self.policy.actor.state_dict(), f"./models/actor_{episode_num+1}.pth")
                    torch.save(self.policy.critic.state_dict(), f"./models/critic_{episode_num+1}.pth")

                # Reset environment
                state = self.env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

def main(args=None):
    rclpy.init(args=args)
    trainer = TrainingNode()
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        pass
    
    trainer.env.destroy_node()
    trainer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
