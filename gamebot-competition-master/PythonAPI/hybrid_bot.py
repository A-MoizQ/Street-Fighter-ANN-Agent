import os
import pickle
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
from command import Command
from buttons import Buttons
from rl_environment import StreetFighterEnv

class HybridBot:
    """Bot that uses hybrid RNN-DQN models - fixed version"""
    
    def __init__(self, player_id=0, model_path=None):
        self.player_id = player_id
        self.env = StreetFighterEnv(player_id=player_id)
        
        # Track previous action for proper state updates
        self.prev_action = 0
        
        # Load hybrid model
        if model_path is None:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rl_models'))
            model_path = os.path.join(base, f'hybrid_rnn_dqn_model_{player_id}_final.keras')
            
            # Fallback to regular hybrid model if final doesn't exist
            if not os.path.exists(model_path):
                model_path = os.path.join(base, f'hybrid_rnn_dqn_model_{player_id}.keras')
        
        print(f"[Hybrid Bot] Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Warm up the model (first prediction is always slow)
        print(f"[Hybrid Bot] Warming up model...")
        dummy_state = np.zeros((1, self.env.window_size, self.env.state_size))
        _ = self.model.predict(dummy_state, verbose=0)
        print(f"[Hybrid Bot] Model warmed up")
        
        # Load model parameters if available
        try:
            with open(model_path + '.params', 'rb') as f:
                params = pickle.load(f)
                print(f"[Hybrid Bot] Loaded parameters: epsilon={params.get('epsilon', 'N/A')}")
        except:
            print(f"[Hybrid Bot] No parameter file found")
        
        # Pure exploitation during gameplay
        self.epsilon = 0.0
        
        print(f"[Hybrid Bot] Model loaded successfully")
        print(f"[Hybrid Bot] Action space size: {self.env.action_size}")
        
        # Debug counters
        self.step_count = 0
        self.last_debug_time = 0
        self.action_counts = {}
    
    def fight(self, gs, player_id):
        """Main method called by controller - simplified version"""
        import time
        start_time = time.time()
        
        # Update environment with previous action (CRITICAL FIX)
        current_state, reward, done = self.env.step(gs, self.prev_action)
        
        # Choose action using hybrid model (NO ACTION PERSISTENCE)
        action_idx = self.act(current_state)
        
        # Store action for next iteration
        self.prev_action = action_idx
        
        # Track action diversity
        action_name = self.env.action_space[action_idx]['name']
        self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1
        
        # Convert action to button combination
        button_combination = self.env.action_to_buttons(action_idx)
        
        # Create command - SAME PATTERN AS RNN/ANN BOTS
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
        
        # Performance monitoring
        inference_time = time.time() - start_time
        self.step_count += 1
        current_time = time.time()
        
        # Debug output every 3 seconds (like RNN bot pattern)
        if current_time - self.last_debug_time >= 3.0:
            self.last_debug_time = current_time
            
            active_buttons = [btn for btn, state in button_combination.items() if state]
            
            print(f"\n[Hybrid Bot] === STEP {self.step_count} REPORT ===")
            print(f"[Hybrid Bot] Current action: {action_name}")
            print(f"[Hybrid Bot] Active buttons: {', '.join(active_buttons) if active_buttons else 'None'}")
            print(f"[Hybrid Bot] Reward: {reward:.2f}")
            print(f"[Hybrid Bot] Round over: {gs.is_round_over}")
            print(f"[Hybrid Bot] Inference time: {inference_time*1000:.1f}ms")
            
            # Show action diversity (like RNN bot)
            print(f"[Hybrid Bot] Action diversity (last {self.step_count} steps):")
            sorted_actions = sorted(self.action_counts.items(), key=lambda x: x[1], reverse=True)
            for action, count in sorted_actions[:5]:  # Top 5 actions
                percentage = (count / self.step_count) * 100
                print(f"  {action}: {count} times ({percentage:.1f}%)")
            
            # Show Q-values for current state
            if self.step_count % 9 == 0:  # Every 9 debug cycles (27 seconds)
                self._show_q_values(current_state)
        
        return cmd
    
    def act(self, state):
        """Choose action using the hybrid trained model - pure exploitation"""
        # Get Q-values from trained hybrid model
        q_values = self.model.predict(state, verbose=0)
        
        # Pure exploitation - choose best action
        action = np.argmax(q_values[0])
        
        return action
    
    def _show_q_values(self, state):
        """Show Q-value distribution (like RNN bot probability display)"""
        try:
            q_vals = self.model.predict(state, verbose=0)[0]
            
            # Get top 5 Q-values
            top_5_indices = np.argsort(q_vals)[-5:][::-1]
            
            print(f"[Hybrid Bot] Top 5 Q-values:")
            for i in top_5_indices:
                action_name = self.env.action_space[i]['name']
                q_value = q_vals[i]
                print(f"  {action_name}: {q_value:.2f}")
                
        except Exception as e:
            print(f"[Hybrid Bot] Error showing Q-values: {e}")