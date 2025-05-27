import os
import sys
import numpy as np
import tensorflow as tf
from collections import deque
import pickle
import time
import joblib
import random
from command import Command
from buttons import Buttons
from rl_environment import StreetFighterEnv

# GPU Configuration (same as your DQN)
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
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

class RNNDQNHybridAgent:
    """Hybrid agent that combines pre-trained RNN with DQN fine-tuning"""
    
    def __init__(self, state_size, action_size, window_size=6, rnn_model_path=None, player_id=0):
        self.state_size = state_size
        self.action_size = action_size
        self.window_size = window_size
        self.player_id = player_id
        self.gpu_available = GPU_AVAILABLE
        
        # DQN parameters (more conservative for fine-tuning)
        self.memory = deque(maxlen=50000)  # Smaller memory for faster training
        self.epsilon = 0.7  # Higher exploration
        self.epsilon_min = 0.1  # Don't decay too much
        self.epsilon_decay = 0.9995  # Slower decay
        self.learning_rate = 0.0005  # Lower learning rate for fine-tuning
        self.gamma = 0.95
        self.batch_size = 32 if self.gpu_available else 16
        self.update_target_freq = 500  # More frequent updates
        self.step_count = 0
        
        # Load pre-trained RNN model
        self.rnn_model, self.scaler = self._load_rnn_model(rnn_model_path, player_id)
        
        # Build hybrid Q-network based on RNN architecture
        with tf.device('/GPU:0' if self.gpu_available else '/CPU:0'):
            self.q_network = self._build_hybrid_model()
            self.target_network = self._build_hybrid_model()
            self._transfer_rnn_weights()
            self.update_target_network()
        
        # Button mapping for RNN compatibility
        self.BUTTONS = ['UP', 'DOWN', 'RIGHT', 'LEFT', 'Y', 'B', 'X', 'A', 'L', 'R']
        
        print(f"[Hybrid Agent] RNN-DQN agent initialized for character {player_id}")
        print(f"[Hybrid Agent] Using pre-trained RNN as baseline")
        print(f"[Hybrid Agent] Starting epsilon: {self.epsilon}")
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def _load_rnn_model(self, rnn_model_path, player_id):
        """Load pre-trained RNN model and scaler"""
        if rnn_model_path is None:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RNN_models'))
            rnn_model_path = os.path.join(base, f'model_{player_id}.keras')
        
        print(f"[Hybrid Agent] Loading RNN model from: {rnn_model_path}")
        
        try:
            rnn_model = tf.keras.models.load_model(rnn_model_path)
            scaler_path = rnn_model_path + '.scaler'
            scaler = joblib.load(scaler_path)
            print(f"[Hybrid Agent] RNN model and scaler loaded successfully")
            return rnn_model, scaler
        except Exception as e:
            print(f"[Hybrid Agent] Error loading RNN model: {e}")
            print(f"[Hybrid Agent] Will use random initialization")
            return None, None
    
    def _build_hybrid_model(self):
        """Build Q-network using similar architecture to RNN but with Q-value output"""
        model = tf.keras.Sequential()
        
        if self.rnn_model is not None:
            # Copy architecture from RNN model but change output layer
            for i, layer in enumerate(self.rnn_model.layers[:-1]):  # All layers except output
                if isinstance(layer, tf.keras.layers.LSTM):
                    if i == 0:  # First LSTM layer
                        model.add(tf.keras.layers.LSTM(
                            layer.units,
                            input_shape=(self.window_size, self.state_size),
                            return_sequences=layer.return_sequences,
                            dropout=getattr(layer, 'dropout', 0.0),
                            recurrent_dropout=getattr(layer, 'recurrent_dropout', 0.0)
                        ))
                    else:
                        model.add(tf.keras.layers.LSTM(
                            layer.units,
                            return_sequences=layer.return_sequences,
                            dropout=getattr(layer, 'dropout', 0.0),
                            recurrent_dropout=getattr(layer, 'recurrent_dropout', 0.0)
                        ))
                elif isinstance(layer, tf.keras.layers.Dense):
                    model.add(tf.keras.layers.Dense(layer.units, activation=layer.activation))
                elif isinstance(layer, tf.keras.layers.Dropout):
                    model.add(tf.keras.layers.Dropout(layer.rate))
        else:
            # Fallback architecture
            model.add(tf.keras.layers.LSTM(64, input_shape=(self.window_size, self.state_size), return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.LSTM(32, return_sequences=False))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
        
        # Q-value output layer (different from RNN classification)
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear', dtype='float32'))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model
    
    def _transfer_rnn_weights(self):
        """Transfer weights from pre-trained RNN to Q-network (except output layer)"""
        if self.rnn_model is None:
            print("[Hybrid Agent] No RNN model to transfer weights from")
            return
        
        try:
            rnn_layers = self.rnn_model.layers[:-1]  # Exclude output layer
            q_layers = self.q_network.layers[:-1]    # Exclude Q-value output layer
            
            transferred = 0
            for i, (rnn_layer, q_layer) in enumerate(zip(rnn_layers, q_layers)):
                if (isinstance(rnn_layer, (tf.keras.layers.LSTM, tf.keras.layers.Dense)) and
                    isinstance(q_layer, type(rnn_layer))):
                    
                    try:
                        q_layer.set_weights(rnn_layer.get_weights())
                        transferred += 1
                        print(f"[Hybrid Agent] Transferred weights for layer {i}: {type(rnn_layer).__name__}")
                    except Exception as e:
                        print(f"[Hybrid Agent] Could not transfer layer {i}: {e}")
            
            print(f"[Hybrid Agent] Successfully transferred {transferred} layers from RNN")
            
            # Freeze early layers to preserve RNN knowledge (optional)
            for i, layer in enumerate(self.q_network.layers[:-2]):  # Keep last 2 layers trainable
                layer.trainable = False
            print(f"[Hybrid Agent] Froze {len(self.q_network.layers)-2} layers to preserve RNN knowledge")
            
        except Exception as e:
            print(f"[Hybrid Agent] Error transferring weights: {e}")
    
    def get_rnn_action_probabilities(self, state_3d):
        """Get action probabilities from original RNN model"""
        if self.rnn_model is None or self.scaler is None:
            return np.zeros(len(self.BUTTONS))
        
        try:
            # Scale the data for RNN
            n_samples, n_timesteps, n_features = state_3d.shape
            X_reshaped = state_3d.reshape(-1, n_features)
            X_scaled = self.scaler.transform(X_reshaped)
            X_scaled_3d = X_scaled.reshape(n_samples, n_timesteps, n_features)
            
            # Get RNN predictions
            rnn_preds = self.rnn_model.predict(X_scaled_3d, verbose=0)[0]
            return rnn_preds
        except Exception as e:
            print(f"[Hybrid Agent] Error getting RNN predictions: {e}")
            return np.zeros(len(self.BUTTONS))
    
    def act(self, state):
        """Hybrid action selection with aggressive exploration bias"""
        if np.random.random() <= self.epsilon:
            # Exploration: bias toward aggressive actions
            if np.random.random() < 0.6:  # 40% chance for aggressive action
                # Choose from aggressive actions (punches, kicks, specials)
                aggressive_actions = []
                for i, action in enumerate(self.action_space):
                    action_name = action['name'].upper()
                    if any(keyword in action_name for keyword in ['PUNCH', 'KICK', 'SPECIAL', 'FORWARD']):
                        aggressive_actions.append(i)
                
                if aggressive_actions:
                    return np.random.choice(aggressive_actions)
            
            # Use RNN guidance for remaining exploration
            rnn_probs = self.get_rnn_action_probabilities(state)
            if np.sum(rnn_probs) > 0:
                rnn_probs = rnn_probs / np.sum(rnn_probs)
                action_idx = self._rnn_to_action_space(rnn_probs)
            else:
                action_idx = np.random.randint(self.action_size)
            
            return action_idx
        else:
            # Exploitation: use Q-network
            with tf.device('/GPU:0' if self.gpu_available else '/CPU:0'):
                q_values = self.q_network.predict(state, verbose=0)
                action_idx = np.argmax(q_values[0])
            
            return action_idx
    
    def _rnn_to_action_space(self, rnn_probs):
        """Convert RNN button probabilities to action space index - FIXED VERSION"""
        # Get the action space from environment
        if not hasattr(self, 'action_space'):
            print("[Hybrid Agent] Warning: No action space available, using fallback")
            return 0
        
        # Method 1: Find best matching combination
        best_match_score = -1
        best_action_idx = 0
        
        for i, action in enumerate(self.action_space):
            score = 0
            total_buttons = len(action['buttons'])
            
            for button_name, button_active in action['buttons'].items():
                if button_name in self.BUTTONS:
                    button_idx = self.BUTTONS.index(button_name)
                    button_prob = rnn_probs[button_idx]
                    
                    if button_active and button_prob > 0.01:  # Button should be active and has high prob
                        score += button_prob
                    elif not button_active and button_prob <= 0.01:  # Button should be inactive and has low prob
                        score += (1.0 - button_prob)
            
            # Normalize score by number of buttons
            normalized_score = score / total_buttons if total_buttons > 0 else 0
            
            if normalized_score > best_match_score:
                best_match_score = normalized_score
                best_action_idx = i
        
        # Method 2: If no good match, use highest probability button
        if best_match_score < 0.3:  # Threshold for "good enough" match
            max_button_idx = np.argmax(rnn_probs)
            button_name = self.BUTTONS[max_button_idx]
            
            # Find simplest action using this button
            for i, action in enumerate(self.action_space):
                if (button_name in action['buttons'] and 
                    action['buttons'][button_name] and 
                    len([b for b in action['buttons'].values() if b]) == 1):  # Single button action
                    return i
        
        return best_action_idx
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0][0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3][0] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch])
        
        with tf.device('/GPU:0' if self.gpu_available else '/CPU:0'):
            current_q_values = self.q_network.predict(states, verbose=0)
            next_q_values = self.target_network.predict(next_states, verbose=0)
            
            targets = current_q_values.copy()
            batch_indices = np.arange(self.batch_size)
            
            # Q-learning update
            max_next_q = np.max(next_q_values, axis=1)
            target_values = rewards + (self.gamma * max_next_q * (1 - dones))
            targets[batch_indices, actions] = target_values
            
            # Train only the unfrozen layers
            self.q_network.fit(states, targets, verbose=0, batch_size=self.batch_size)
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save the hybrid model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.q_network.save(filepath)
        
        params = {
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'window_size': self.window_size,
            'gpu_available': self.gpu_available,
            'player_id': self.player_id
        }
        
        with open(filepath + '.params', 'wb') as f:
            pickle.dump(params, f)

class RNNDQNHybridTrainer:
    """Trainer for the hybrid RNN-DQN approach"""
    
    def __init__(self, player_id=0, episodes=100):
        self.player_id = player_id
        self.env = StreetFighterEnv(player_id=player_id)
        self.agent = RNNDQNHybridAgent(
            state_size=self.env.state_size, 
            action_size=self.env.action_size,
            player_id=player_id
        )
        # Store action space reference in agent for RNN mapping
        self.agent.action_space = self.env.action_space
        
        self.gpu_available = GPU_AVAILABLE
        
        # Training state
        self.episode = 0
        self.max_episodes = episodes
        self.step_count = 0
        self.episode_reward = 0
        self.episode_steps = 0
        self.max_steps_per_episode = 1800  # 30 seconds (shorter episodes)
        
        # Training frequency
        self.train_frequency = 10  # Train every 3 steps
        
        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.prev_state = None
        self.prev_action = None
        self.episode_start_time = time.time()
        
        # Round state management to prevent multiple episode ends
        self.round_ended = False
        self.prev_round_over = False
        self.round_start_cooldown = 0
        self.training_complete = False  # Flag to prevent multiple completions
        
        # Action diversity tracking
        self.recent_actions = deque(maxlen=30)  # Track last 30 actions
        self.action_diversity_bonus = 0.1
                
        # Create save directory
        self.model_dir = os.path.abspath(os.path.join('..', 'rl_models'))
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"[Hybrid Trainer] Initialized for character {player_id}")
        print(f"[Hybrid Trainer] GPU Acceleration: {'Enabled' if self.gpu_available else 'Disabled'}")
        print(f"[Hybrid Trainer] Using RNN baseline + DQN fine-tuning")
        print(f"[Hybrid Trainer] Episodes: {episodes} (much fewer than pure DQN)")
        print(f"[Hybrid Trainer] Max steps per episode: {self.max_steps_per_episode}")
    
    def train_step(self, gs, player_id):
        """Training step with RNN-DQN hybrid approach"""
        
        # Prevent training if already complete
        if self.training_complete:
            # Return idle command
            cmd = Command()
            button_map = {
                'UP': False, 'DOWN': False, 'RIGHT': False, 'LEFT': False,
                'Y': False, 'B': False, 'X': False, 'A': False, 'L': False, 'R': False
            }
            if player_id == "1":
                cmd.player_buttons = Buttons(button_map)
            else:
                cmd.player2_buttons = Buttons(button_map)
            return cmd
        
        # Round state management
        current_round_over = gs.is_round_over
        
        # Detect round start (transition from round over to not round over)
        if self.prev_round_over and not current_round_over:
            print(f"[Hybrid Trainer] New round detected - resetting episode state")
            self.round_ended = False
            self.round_start_cooldown = 60  # 1 second cooldown at 60 FPS
        
        # Countdown cooldown
        if self.round_start_cooldown > 0:
            self.round_start_cooldown -= 1
        
        # Update round state tracking
        self.prev_round_over = current_round_over
        

        # Get current state and base reward
        current_state, base_reward, done = self.env.step(gs, self.prev_action if self.prev_action is not None else 0)
        
        # Choose action using hybrid approach (MOVE THIS UP)
        action_idx = self.agent.act(current_state)
        
        # Apply contextual reward system (NOW action_idx is available)
        enhanced_reward = self.env.calculate_contextual_reward(gs, action_idx, base_reward)
        
        # Add diversity bonus to enhanced reward
        if len(self.recent_actions) > 10:
            unique_actions = len(set(self.recent_actions))
            diversity_bonus = self.action_diversity_bonus * (unique_actions / len(self.recent_actions))
            final_reward = enhanced_reward + diversity_bonus
        else:
            final_reward = enhanced_reward
        
        # Store experience if we have a previous state (use final_reward)
        if self.prev_state is not None and self.prev_action is not None:
            self.agent.remember(self.prev_state, self.prev_action, final_reward, current_state, done)
            self.episode_reward += final_reward
        
        # Track action for diversity (ONLY ONCE)
        self.recent_actions.append(action_idx)
        
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
        
        # Check for episode end with proper state management
        should_end_episode = (
            # Only end on actual round completion or max steps
            (self.episode_steps >= self.max_steps_per_episode or 
             (current_round_over )) and
            not self.round_ended and  
            self.round_start_cooldown == 0 and  
            self.episode_steps > 60  # Minimum 5 minutes per episode
        )
        
        if should_end_episode:
            self.round_ended = True  # Mark this round as processed
            self._end_episode()
        
        # Debug output
        if self.step_count % 180 == 0:  # Every 3 seconds
            action_name = self.env.action_space[action_idx]['name']
            print(f"\n[Hybrid Trainer] Episode {self.episode + 1}/{self.max_episodes}")
            print(f"[Hybrid Trainer] Step {self.step_count}, Episode Step {self.episode_steps}")
            print(f"[Hybrid Trainer] Action: {action_name}, Reward: {final_reward:.2f}")
            print(f"[Hybrid Trainer] Epsilon: {self.agent.epsilon:.3f}")
            print(f"[Hybrid Trainer] Episode Reward: {self.episode_reward:.2f}")
            print(f"[Hybrid Trainer] Memory Size: {len(self.agent.memory)}")
            print(f"[Hybrid Trainer] Round Over: {current_round_over}, Round Ended Flag: {self.round_ended}")
            print(f"[Hybrid Trainer] Mode: {'Exploration (RNN-guided)' if np.random.random() <= self.agent.epsilon else 'Exploitation (Q-network)'}")
            if self.gpu_available:
                print(f"[Hybrid Trainer] GPU Training: Every {self.train_frequency} steps")
        
        return cmd
    

    def _end_episode(self):
        """Handle end of episode"""
        
        # Additional safety check
        if self.training_complete:
            return
        
        self.episode_rewards.append(self.episode_reward)
        
        episode_time = time.time() - self.episode_start_time
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        
        print(f"\n[Hybrid Trainer] ===== EPISODE {self.episode + 1} COMPLETE =====")
        print(f"[Hybrid Trainer] Episode Reward: {self.episode_reward:.2f}")
        print(f"[Hybrid Trainer] Episode Steps: {self.episode_steps}")
        print(f"[Hybrid Trainer] Episode Time: {episode_time:.1f}s")
        print(f"[Hybrid Trainer] Average Reward (last 100): {avg_reward:.2f}")
        print(f"[Hybrid Trainer] Epsilon: {self.agent.epsilon:.3f}")
        print(f"[Hybrid Trainer] Memory Size: {len(self.agent.memory)}")
        if self.gpu_available:
            print(f"[Hybrid Trainer] GPU Acceleration: Active")
        
        # Save model frequently (fast training)
        save_frequency = 10 if self.gpu_available else 20
        if (self.episode + 1) % save_frequency == 0:
            save_start = time.time()
            model_path = os.path.join(self.model_dir, f'hybrid_rnn_dqn_model_{self.player_id}.keras')
            self.agent.save(model_path)
            save_time = time.time() - save_start
            print(f"[Hybrid Trainer] Model saved to {model_path} in {save_time:.2f}s")
        
        # CRITICAL FIX: Reset for next episode (ALWAYS, not just when saving)
        self.episode += 1
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.env.reset_buffer()
        self.prev_state = None
        self.prev_action = None
        self.recent_actions.clear()  # Reset action diversity tracking
        self.round_ended = False  # Reset round flag for new episode
        
        # Check if training is complete (AFTER incrementing episode)
        if self.episode >= self.max_episodes and not self.training_complete:
            self.training_complete = True  # Set flag to prevent multiple completions
            print(f"\n[Hybrid Trainer] Training complete! {self.max_episodes} episodes finished.")
            
            # Save final model
            model_path = os.path.join(self.model_dir, f'hybrid_rnn_dqn_model_{self.player_id}_final.keras')
            self.agent.save(model_path)
            print(f"[Hybrid Trainer] Final hybrid model saved to {model_path}")
            
            # Save training statistics
            stats_path = os.path.join(self.model_dir, f'hybrid_training_stats_{self.player_id}.txt')
            with open(stats_path, 'w') as f:
                f.write(f"Hybrid RNN-DQN Training Statistics for Character {self.player_id}\n")
                f.write(f"GPU Acceleration: {'Enabled' if self.gpu_available else 'Disabled'}\n")
                f.write(f"Total Episodes: {self.episode}\n")
                f.write(f"Total Steps: {self.step_count}\n")
                f.write(f"Final Epsilon: {self.agent.epsilon:.3f}\n")
                f.write(f"Average Reward (last 100): {avg_reward:.2f}\n")
                f.write(f"Memory Size: {len(self.agent.memory)}\n")
            
            print(f"[Hybrid Trainer] Training statistics saved to {stats_path}")