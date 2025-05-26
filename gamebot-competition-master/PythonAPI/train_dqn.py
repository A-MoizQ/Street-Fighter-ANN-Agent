import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
import joblib
import pickle

# GPU Configuration - Add this at the top
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    # Check for GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent allocation of all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision for better performance on RTX cards
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"[GPU] Found {len(gpus)} GPU(s). Using GPU acceleration with mixed precision.")
            print(f"[GPU] GPU Details: {[gpu.name for gpu in gpus]}")
            return True
        except RuntimeError as e:
            print(f"[GPU] Error configuring GPU: {e}")
            print("[GPU] Falling back to CPU")
            return False
    else:
        print("[GPU] No GPU found. Using CPU.")
        return False

# Configure GPU at module load
GPU_AVAILABLE = configure_gpu()

# Add PythonAPI to path for environment import
pythonapi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PythonAPI'))
sys.path.insert(0, pythonapi_path)

class DQNAgent:
    def __init__(self, state_size, action_size, window_size=6):
        self.state_size = state_size
        self.action_size = action_size
        self.window_size = window_size
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount factor
        self.batch_size = 64 if GPU_AVAILABLE else 32  # Larger batch size for GPU
        self.update_target_freq = 1000
        self.step_count = 0
        self.gpu_available = GPU_AVAILABLE
        
        # Build networks with device placement
        with tf.device('/GPU:0' if self.gpu_available else '/CPU:0'):
            print(f"[DQN] Building networks on {'GPU' if self.gpu_available else 'CPU'}")
            self.q_network = self._build_model()
            self.target_network = self._build_model()
            self.update_target_network()
    
    def _build_model(self):
        """Build the DQN model using RNN architecture with GPU optimization"""
        model = models.Sequential([
            layers.LSTM(64, input_shape=(self.window_size, self.state_size), 
                       return_sequences=True, 
                       activation='tanh',  # Better for mixed precision
                       recurrent_activation='sigmoid'),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False,
                       activation='tanh',
                       recurrent_activation='sigmoid'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear', dtype='float32')  # Ensure float32 output
        ])
        
        # Use different optimizers based on GPU availability
        if self.gpu_available:
            optimizer = optimizers.Adam(learning_rate=self.learning_rate, 
                                      epsilon=1e-7)  # Better for mixed precision
        else:
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(optimizer=optimizer, loss='mse')
        return model
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy with GPU acceleration"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Ensure tensor is on correct device
        with tf.device('/GPU:0' if self.gpu_available else '/CPU:0'):
            q_values = self.q_network.predict(state, verbose=0)
            return np.argmax(q_values[0])
    
    def replay(self):
        """Train the model on a batch of experiences with GPU acceleration"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = np.array([e[0][0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3][0] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch])
        
        # Use GPU for batch processing
        with tf.device('/GPU:0' if self.gpu_available else '/CPU:0'):
            # Current Q values
            current_q_values = self.q_network.predict(states, verbose=0)
            
            # Next Q values from target network
            next_q_values = self.target_network.predict(next_states, verbose=0)
            
            # Update Q values using vectorized operations for better GPU utilization
            targets = current_q_values.copy()
            
            # Vectorized Q-learning update
            max_next_q = np.max(next_q_values, axis=1)
            target_values = rewards + (self.gamma * max_next_q * (1 - dones))
            
            # Update targets
            batch_indices = np.arange(self.batch_size)
            targets[batch_indices, actions] = target_values
            
            # Train the model
            if self.gpu_available:
                # Use larger batch processing for GPU
                self.q_network.fit(states, targets, 
                                 batch_size=self.batch_size,
                                 verbose=0,
                                 epochs=1)
            else:
                self.q_network.fit(states, targets, verbose=0)
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.update_target_network()
            if self.gpu_available:
                print(f"[DQN] Target network updated at step {self.step_count}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save the model and agent parameters"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.q_network.save(filepath)
        
        # Save agent parameters including GPU info
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
            'gpu_available': self.gpu_available
        }
        
        with open(filepath + '.params', 'wb') as f:
            pickle.dump(params, f)
        
        print(f"[DQN] Model saved to {filepath} ({'GPU' if self.gpu_available else 'CPU'} trained)")
    
    def load(self, filepath):
        """Load the model and agent parameters"""
        # Load with appropriate device
        with tf.device('/GPU:0' if self.gpu_available else '/CPU:0'):
            self.q_network = tf.keras.models.load_model(filepath)
            self.target_network = tf.keras.models.load_model(filepath)
        
        # Load agent parameters
        with open(filepath + '.params', 'rb') as f:
            params = pickle.load(f)
            for key, value in params.items():
                setattr(self, key, value)
        
        print(f"[DQN] Model loaded from {filepath} (Running on {'GPU' if self.gpu_available else 'CPU'})")

def train_dqn_agent(character_id, episodes=1000):
    """Train DQN agent for specific character with GPU acceleration"""
    from rl_environment import StreetFighterEnv
    
    # Print GPU status
    print(f"\n[Training] GPU Acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")
    if GPU_AVAILABLE:
        print(f"[Training] Batch size increased to 64 for GPU efficiency")
    
    # Initialize environment
    env = StreetFighterEnv(player_id=character_id)
    
    # Initialize agent
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    
    # Training metrics
    scores = deque(maxlen=100)
    
    print(f"Training DQN agent for character {character_id}")
    print(f"State size: {env.state_size}, Action size: {env.action_size}")
    print(f"Batch size: {agent.batch_size}")
    
    for episode in range(episodes):
        env.reset_buffer()
        state = env.get_state_vector()
        total_reward = 0
        steps = 0
        
        # Note: This template shows the structure for offline training
        # For live training, use the DQNTrainer class through controller.py
        
        print(f"Episode {episode + 1}/{episodes}")
        print(f"Average Score: {np.mean(scores) if scores else 0:.2f}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        
        # Save model periodically
        if (episode + 1) % 100 == 0:
            model_dir = os.path.abspath(os.path.join('..', 'rl_models'))
            model_path = os.path.join(model_dir, f'dqn_model_{character_id}.keras')
            agent.save(model_path)
            print(f"Model saved to {model_path}")

if __name__ == '__main__':
    # Print system info
    print("=== System Information ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {GPU_AVAILABLE}")
    
    if GPU_AVAILABLE:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f"GPU Details: {[gpu.name for gpu in gpus]}")
        print("Mixed Precision: Enabled")
    
    # Train for specific characters
    characters_to_train = [7, 10]  # Dhalsim and Balrog
    
    for character_id in characters_to_train:
        print(f"\n=== Training DQN for character {character_id} ===")
        train_dqn_agent(character_id, episodes=500)