import os
import sys
import numpy as np
import tensorflow as tf
from collections import deque
import pickle
import time
from command import Command
from buttons import Buttons
from rl_environment import StreetFighterEnv

# GPU Configuration
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision for RTX cards
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"[GPU] GPU acceleration enabled with mixed precision.")
            return True
        except RuntimeError as e:
            print(f"[GPU] Error: {e}. Using CPU.")
            return False
    else:
        print("[GPU] No GPU found. Using CPU.")
        return False

GPU_AVAILABLE = configure_gpu()

# Add the train_models directory to the Python path
train_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_models'))
sys.path.insert(0, train_models_path)

from train_dqn import DQNAgent

class DQNTrainer:
    def __init__(self, player_id=0, episodes=1000):
        self.player_id = player_id
        self.env = StreetFighterEnv(player_id=player_id)
        self.agent = DQNAgent(state_size=self.env.state_size, action_size=self.env.action_size)
        self.gpu_available = GPU_AVAILABLE
        
        # Training state - adjusted for GPU
        self.episode = 0
        self.max_episodes = episodes
        self.step_count = 0
        self.episode_reward = 0
        self.episode_steps = 0
        self.max_steps_per_episode = 3600  # 60 seconds at 60 FPS
        
        # Training frequency - more frequent training with GPU
        self.train_frequency = 2 if self.gpu_available else 5
        
        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.prev_state = None
        self.prev_action = None
        self.episode_start_time = time.time()
        
        # Create save directory
        self.model_dir = os.path.abspath(os.path.join('..', 'rl_models'))
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"[DQN Trainer] Initialized for character {player_id}")
        print(f"[DQN Trainer] GPU Acceleration: {'Enabled' if self.gpu_available else 'Disabled'}")
        print(f"[DQN Trainer] State size: {self.env.state_size}, Action size: {self.env.action_size}")
        print(f"[DQN Trainer] Training frequency: Every {self.train_frequency} steps")
        print(f"[DQN Trainer] Batch size: {self.agent.batch_size}")
        print(f"[DQN Trainer] Training for {episodes} episodes")
    
    def train_step(self, gs, player_id):
        """Process one training step during live gameplay with GPU acceleration"""
        
        # Get current state
        current_state, reward, done = self.env.step(gs, self.prev_action if self.prev_action is not None else 0)
        
        # Store experience if we have a previous state
        if self.prev_state is not None and self.prev_action is not None:
            self.agent.remember(self.prev_state, self.prev_action, reward, current_state, done)
            self.episode_reward += reward
            
            # Train more frequently with GPU, less frequently with CPU
            if (len(self.agent.memory) > self.agent.batch_size and 
                self.step_count % self.train_frequency == 0):
                
                # Time the training step to monitor GPU performance
                if self.gpu_available and self.step_count % 300 == 0:
                    start_time = time.time()
                    self.agent.replay()
                    train_time = time.time() - start_time
                    print(f"[GPU] Training step completed in {train_time:.3f}s")
                else:
                    self.agent.replay()
        
        # Choose next action with GPU acceleration
        action_idx = self.agent.act(current_state)
        
        # Convert action to button combination
        button_combination = self.env.action_to_buttons(action_idx)
        
        # Create command
        cmd = Command()
        button_map = {
            'UP': False, 'DOWN': False, 'RIGHT': False, 'LEFT': False,
            'Y': False, 'B': False, 'X': False, 'A': False, 'L': False, 'R': False
        }
        button_map.update(button_combination)
        
        if player_id == "1":
            cmd.player_buttons = Buttons(button_map)
        else:
            cmd.player2_buttons = Buttons(button_map)
        
        # Update tracking variables
        self.prev_state = current_state
        self.prev_action = action_idx
        self.step_count += 1
        self.episode_steps += 1
        
        # Check for episode end
        if done or self.episode_steps >= self.max_steps_per_episode or gs.is_round_over:
            self._end_episode()
        
        # Debug output with GPU performance info
        if self.step_count % 300 == 0:  # Every 5 seconds
            action_name = self.env.action_space[action_idx]['name']
            print(f"\n[DQN Trainer] Episode {self.episode + 1}/{self.max_episodes}")
            print(f"[DQN Trainer] Step {self.step_count}, Episode Step {self.episode_steps}")
            print(f"[DQN Trainer] Action: {action_name}, Reward: {reward:.2f}")
            print(f"[DQN Trainer] Epsilon: {self.agent.epsilon:.3f}")
            print(f"[DQN Trainer] Episode Reward: {self.episode_reward:.2f}")
            print(f"[DQN Trainer] Memory Size: {len(self.agent.memory)}")
            if self.gpu_available:
                print(f"[DQN Trainer] GPU Training: Every {self.train_frequency} steps")
        
        return cmd
    
    def _end_episode(self):
        """Handle end of episode with GPU performance logging"""
        self.episode_rewards.append(self.episode_reward)
        
        episode_time = time.time() - self.episode_start_time
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        
        print(f"\n[DQN Trainer] ===== EPISODE {self.episode + 1} COMPLETE =====")
        print(f"[DQN Trainer] Episode Reward: {self.episode_reward:.2f}")
        print(f"[DQN Trainer] Episode Steps: {self.episode_steps}")
        print(f"[DQN Trainer] Episode Time: {episode_time:.1f}s")
        print(f"[DQN Trainer] Average Reward (last 100): {avg_reward:.2f}")
        print(f"[DQN Trainer] Epsilon: {self.agent.epsilon:.3f}")
        print(f"[DQN Trainer] Memory Size: {len(self.agent.memory)}")
        if self.gpu_available:
            print(f"[DQN Trainer] GPU Acceleration: Active")
        
        # Save model more frequently with GPU (faster saves)
        save_frequency = 25 if self.gpu_available else 50
        if (self.episode + 1) % save_frequency == 0:
            save_start = time.time()
            model_path = os.path.join(self.model_dir, f'dqn_model_{self.player_id}.keras')
            self.agent.save(model_path)
            save_time = time.time() - save_start
            print(f"[DQN Trainer] Model saved to {model_path} in {save_time:.2f}s")
        
        # Reset for next episode
        self.episode += 1
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.env.reset_buffer()
        self.prev_state = None
        self.prev_action = None
        
        # Check if training is complete
        if self.episode >= self.max_episodes:
            print(f"\n[DQN Trainer] Training complete! {self.max_episodes} episodes finished.")
            # Save final model
            model_path = os.path.join(self.model_dir, f'dqn_model_{self.player_id}_final.keras')
            self.agent.save(model_path)
            print(f"[DQN Trainer] Final model saved to {model_path}")
            
            # Save training statistics with GPU info
            stats_path = os.path.join(self.model_dir, f'training_stats_{self.player_id}.txt')
            with open(stats_path, 'w') as f:
                f.write(f"DQN Training Statistics for Character {self.player_id}\n")
                f.write(f"GPU Acceleration: {'Enabled' if self.gpu_available else 'Disabled'}\n")
                f.write(f"Batch Size: {self.agent.batch_size}\n")
                f.write(f"Training Frequency: Every {self.train_frequency} steps\n")
                f.write(f"Total Episodes: {self.episode}\n")
                f.write(f"Total Steps: {self.step_count}\n")
                f.write(f"Final Epsilon: {self.agent.epsilon:.3f}\n")
                f.write(f"Average Reward (last 100): {avg_reward:.2f}\n")
                f.write(f"Memory Size: {len(self.agent.memory)}\n")
            
            print(f"[DQN Trainer] Training statistics saved to {stats_path}")