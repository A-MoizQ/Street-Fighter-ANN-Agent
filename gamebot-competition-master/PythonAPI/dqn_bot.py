import os
import pickle
import numpy as np
import tensorflow as tf
from collections import deque
from command import Command
from buttons import Buttons
from rl_environment import StreetFighterEnv

class DQNBot:
    def __init__(self, player_id=0, model_path=None):
        self.player_id = player_id
        self.env = StreetFighterEnv(player_id=player_id)
        
        # Load model and parameters
        if model_path is None:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rl_models'))
            model_path = os.path.join(base, f'dqn_model_{player_id}.keras')
        
        print(f"[DQN Bot] Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load agent parameters
        with open(model_path + '.params', 'rb') as f:
            self.params = pickle.load(f)
        
        # Set epsilon to 0 for pure exploitation during gameplay
        self.epsilon = 0.0
        
        print(f"[DQN Bot] Model loaded successfully")
        print(f"[DQN Bot] Action space size: {self.env.action_size}")
    
    def fight(self, gs, player_id):
        """Main method called by controller"""
        # Process game state through environment
        next_state, reward, done = self.env.step(gs, 0)  # Action doesn't matter for state processing
        
        # Get current state
        state = self.env.get_state_vector()
        
        # Choose action using trained model
        action_idx = self.act(state)
        
        # Convert action to button combination
        button_combination = self.env.action_to_buttons(action_idx)
        
        # Create command
        cmd = Command()
        
        # Map buttons to command
        button_map = {
            'UP': False, 'DOWN': False, 'RIGHT': False, 'LEFT': False,
            'Y': False, 'B': False, 'X': False, 'A': False, 'L': False, 'R': False
        }
        
        # Update with selected buttons
        button_map.update(button_combination)
        
        if player_id == "1":
            cmd.player_buttons = Buttons(button_map)
        else:
            cmd.player2_buttons = Buttons(button_map)
        
        # Debug output
        action_name = self.env.action_space[action_idx]['name']
        active_buttons = [btn for btn, state in button_combination.items() if state]
        
        print(f"\n[DQN Bot] Selected action: {action_name}")
        print(f"[DQN Bot] Active buttons: {', '.join(active_buttons) if active_buttons else 'None'}")
        print(f"[DQN Bot] Reward: {reward:.2f}")
        
        return cmd
    
    def act(self, state):
        """Choose action using the trained model"""
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.env.action_size)
        
        q_values = self.model.predict(state, verbose=0)
        action = np.argmax(q_values[0])
        
        print(f"[DQN Bot] Q-values preview: {q_values[0][:5]}...")  # Show first 5 Q-values
        print(f"[DQN Bot] Selected action index: {action}")
        
        return action