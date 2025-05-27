import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import time
from collections import deque
from command import Command
from buttons import Buttons

# Define constants the same way as RNN bot
WINDOW_SIZE = 6
STATE_FEATURES = ['timer', 'fight_result', 'has_round_started', 'is_round_over',
                 'player1_id', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move_id',
                 'player2_id', 'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id',
                 'diff_x', 'diff_y', 'diff_health']

BUTTONS = ['UP', 'DOWN', 'RIGHT', 'LEFT', 'Y', 'B', 'X', 'A', 'L', 'R']

class GameStateTracker:
    """Tracks meaningful changes in game state"""
    
    def __init__(self):
        self.last_state_hash = None
        self.last_action_time = 0
        self.forced_action_interval = 1.0  # Force action every 1 second regardless
        self.minimal_change_threshold = 2.0  # Minimum 2 seconds between actions on tiny changes
        
    def get_state_signature(self, gs, player_id):
        """Create a signature of important state elements"""
        # Determine player and opponent
        if player_id == gs.player1.player_id:
            player = gs.player1
            opponent = gs.player2
        else:
            player = gs.player2
            opponent = gs.player1
        
        # Key state elements that matter for decision making
        signature = {
            'timer': gs.timer,
            'fight_result': gs.fight_result,
            'round_started': gs.has_round_started,
            'round_over': gs.is_round_over,
            
            # Player state (rounded to reduce noise)
            'p_health': player.health,
            'p_x': round(player.x_coord / 10) * 10,  # Round to nearest 10 pixels
            'p_y': round(player.y_coord / 10) * 10,
            'p_jumping': player.is_jumping,
            'p_crouching': player.is_crouching,
            'p_in_move': player.is_player_in_move,
            'p_move_id': player.move_id,
            
            # Opponent state (rounded to reduce noise)
            'o_health': opponent.health,
            'o_x': round(opponent.x_coord / 10) * 10,
            'o_y': round(opponent.y_coord / 10) * 10,
            'o_jumping': opponent.is_jumping,
            'o_crouching': opponent.is_crouching,
            'o_in_move': opponent.is_player_in_move,
            'o_move_id': opponent.move_id,
            
            # Relative positions (rounded)
            'distance': round(abs(player.x_coord - opponent.x_coord) / 20) * 20,  # Round to nearest 20
            'height_diff': round((player.y_coord - opponent.y_coord) / 10) * 10,
            'health_diff': player.health - opponent.health,
        }
        
        return signature
    
    def has_significant_change(self, gs, player_id):
        """Check if there's been a significant state change worth acting on"""
        current_time = time.time()
        current_signature = self.get_state_signature(gs, player_id)
        
        # Always act on first call
        if self.last_state_hash is None:
            self.last_state_hash = current_signature
            self.last_action_time = current_time
            return True, "INITIAL"
        
        # Force action if too much time has passed (safety mechanism)
        time_since_last_action = current_time - self.last_action_time
        if time_since_last_action >= self.forced_action_interval:
            self.last_state_hash = current_signature
            self.last_action_time = current_time
            return True, "FORCED_TIMEOUT"
        
        # Check for significant state changes
        changes = []
        
        # CRITICAL CHANGES (always act)
        if current_signature['fight_result'] != self.last_state_hash['fight_result']:
            changes.append("FIGHT_RESULT")
        
        if current_signature['round_started'] != self.last_state_hash['round_started']:
            changes.append("ROUND_STATE")
        
        if current_signature['p_health'] != self.last_state_hash['p_health']:
            changes.append("PLAYER_HEALTH")
        
        if current_signature['o_health'] != self.last_state_hash['o_health']:
            changes.append("OPPONENT_HEALTH")
        
        if current_signature['p_in_move'] != self.last_state_hash['p_in_move']:
            changes.append("PLAYER_MOVE_STATE")
        
        if current_signature['o_in_move'] != self.last_state_hash['o_in_move']:
            changes.append("OPPONENT_MOVE_STATE")
        
        if current_signature['p_move_id'] != self.last_state_hash['p_move_id']:
            changes.append("PLAYER_MOVE_ID")
        
        if current_signature['o_move_id'] != self.last_state_hash['o_move_id']:
            changes.append("OPPONENT_MOVE_ID")
        
        # MOVEMENT CHANGES (act if significant)
        distance_change = abs(current_signature['distance'] - self.last_state_hash['distance'])
        if distance_change >= 20:  # Moved at least 20 pixels
            changes.append("DISTANCE_CHANGE")
        
        position_change = (
            abs(current_signature['p_x'] - self.last_state_hash['p_x']) >= 20 or
            abs(current_signature['p_y'] - self.last_state_hash['p_y']) >= 20 or
            abs(current_signature['o_x'] - self.last_state_hash['o_x']) >= 20 or
            abs(current_signature['o_y'] - self.last_state_hash['o_y']) >= 20
        )
        if position_change:
            changes.append("POSITION_CHANGE")
        
        # STATE CHANGES
        if (current_signature['p_jumping'] != self.last_state_hash['p_jumping'] or
            current_signature['p_crouching'] != self.last_state_hash['p_crouching'] or
            current_signature['o_jumping'] != self.last_state_hash['o_jumping'] or
            current_signature['o_crouching'] != self.last_state_hash['o_crouching']):
            changes.append("STANCE_CHANGE")
        
        # TIMER CHANGES (every few seconds)
        timer_change = abs(current_signature['timer'] - self.last_state_hash['timer'])
        if timer_change >= 3:  # Every 3 game timer units
            changes.append("TIMER_CHANGE")
        
        # Decide if we should act
        should_act = False
        change_reason = ""
        
        if changes:
            # Don't act too frequently on minor changes
            if time_since_last_action >= self.minimal_change_threshold or any(
                change in ["FIGHT_RESULT", "ROUND_STATE", "PLAYER_HEALTH", "OPPONENT_HEALTH", 
                          "PLAYER_MOVE_STATE", "OPPONENT_MOVE_STATE", "PLAYER_MOVE_ID", "OPPONENT_MOVE_ID"]
                for change in changes
            ):
                should_act = True
                change_reason = ",".join(changes)
                self.last_state_hash = current_signature
                self.last_action_time = current_time
        
        return should_act, change_reason

class NeuroevolutionBot:
    """Bot that uses evolved neural network weights with state-change based actions"""
    
    def __init__(self, player_id=0, weights_path=None, fps=3.0):
        self.player_id = player_id
        self.weights_path = weights_path
        self.fps = fps  # Keep for compatibility but don't rely on it
        self.buttons = Buttons()
        self.cmd = Command()
        
        # State-change based action system
        self.state_tracker = GameStateTracker()
        self.last_button_map = {}
        self.last_significant_action = None
        self.action_cache = None  # Cache last action for repeated calls
        
        # Move state filtering (simplified)
        self.move_inhibit_until = 0
        self.last_move_end_time = 0
        
        # Load base RNN model architecture (same path as RNN bot)
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RNN_models'))
        model_path = os.path.join(base, f'model_{player_id}.keras')
        
        print(f"[NeuroBot] Loading base model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        scaler_path = model_path + '.scaler'
        self.scaler = joblib.load(scaler_path)
        
        # Load evolved weights if provided
        if weights_path and os.path.exists(weights_path):
            self.load_evolved_weights(weights_path)
            print(f"[NeuroBot] Loaded evolved weights from {weights_path}")
        else:
            print(f"[NeuroBot] Using base RNN weights")
        
        # Initialize frame buffer (same as RNN bot)
        self.buffer = deque(maxlen=WINDOW_SIZE)
        empty = {feat: 0 for feat in STATE_FEATURES}
        empty['fight_result'] = 'NOT_OVER'
        for _ in range(WINDOW_SIZE):
            self.buffer.append(empty.copy())
        
        print(f"[NeuroBot] Initialized for character {player_id} with STATE-CHANGE based actions")
        print(f"[NeuroBot] Will only act when game state changes significantly")
    
    def load_evolved_weights(self, weights_path):
        """Load evolved weights into the model"""
        try:
            with open(weights_path, 'rb') as f:
                evolved_weights = pickle.load(f)
            
            # Set the evolved weights
            self.model.set_weights(evolved_weights)
            print(f"[NeuroBot] Successfully loaded {len(evolved_weights)} weight matrices")
        except Exception as e:
            print(f"[NeuroBot] Error loading weights: {e}")
    
    def _frame_to_dict(self, gs):
        """Convert GameState to raw feature dict (same as RNN bot)"""
        p1, p2 = gs.player1, gs.player2
        d = {
            'timer': gs.timer,
            'fight_result': gs.fight_result,
            'has_round_started': int(gs.has_round_started),
            'is_round_over': int(gs.is_round_over),
            'player1_id': p1.player_id,
            'p1_health': p1.health,
            'p1_x': p1.x_coord,
            'p1_y': p1.y_coord,
            'p1_jumping': int(p1.is_jumping),
            'p1_crouching': int(p1.is_crouching),
            'p1_in_move': int(p1.is_player_in_move),
            'p1_move_id': p1.move_id,
            'player2_id': p2.player_id,
            'p2_health': p2.health,
            'p2_x': p2.x_coord,
            'p2_y': p2.y_coord,
            'p2_jumping': int(p2.is_jumping),
            'p2_crouching': int(p2.is_crouching),
            'p2_in_move': int(p2.is_player_in_move),
            'p2_move_id': p2.move_id,
            'diff_x': p1.x_coord - p2.x_coord,
            'diff_y': p1.y_coord - p2.y_coord,
            'diff_health': p1.health - p2.health,
        }
        return d
    
    def _get_raw_prediction(self, gs):
        """Get raw prediction from neural network (same structure as RNN bot)"""
        # 1. Append new frame (same as RNN bot)
        raw = self._frame_to_dict(gs)
        self.buffer.append(raw)

        # 2. Prepare data for RNN (3D shape) - same as RNN bot
        X_3d = []
        FIGHT_MAP = {'NOT_OVER': 0, 'P1': 1, 'P2': 2, 'DRAW': 3}

        # Create a sequence for each time step
        for t in range(WINDOW_SIZE):
            frame = self.buffer[t]
            # Extract features for this frame
            features = []
            for feat in STATE_FEATURES:
                val = frame[feat]
                if feat == 'fight_result':
                    features.append(FIGHT_MAP.get(val, 0))
                else:
                    features.append(float(val))
            X_3d.append(features)
        
        # Convert to numpy array with shape (1, WINDOW_SIZE, n_features)
        X_3d = np.array([X_3d])
        
        # 3. Scale the data (reshape to 2D for scaling, then back to 3D) - same as RNN bot
        n_samples, n_timesteps, n_features = X_3d.shape
        X_reshaped = X_3d.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled_3d = X_scaled.reshape(n_samples, n_timesteps, n_features)

        # 4. Predict (same as RNN bot)
        preds = self.model.predict(X_scaled_3d, verbose=0)[0]

        # 5. Map to Buttons, resolving opposing directions (same as RNN bot)
        probs = {b: float(preds[i]) for i, b in enumerate(BUTTONS)}
        
        # Remove pairs that cancel each other out (same logic as RNN bot)
        if probs['LEFT'] and probs['RIGHT']:
            if probs['LEFT'] > probs['RIGHT']:
                probs['RIGHT'] = 0.0
            else:
                probs['LEFT'] = 0.0
        if probs['UP'] and probs['DOWN']:
            if probs['UP'] > probs['DOWN']:
                probs['DOWN'] = 0.0
            else:
                probs['UP'] = 0.0

        # Final button map (same threshold as RNN bot)
        btn_map = {b: (probs[b] > 0.01) for b in BUTTONS}
        
        return btn_map
    
    def _filter_buttons_by_move_state(self, gs, raw_buttons):
        """Apply move-state based filtering (simplified)"""
        current_time = time.time()
        
        # Get player state
        player = gs.player1 if self.player_id == gs.player1.player_id else gs.player2
        
        # Initialize filtered buttons
        filtered_buttons = raw_buttons.copy()
        
        # 1. Critical: Don't attack during move execution
        if player.is_player_in_move and player.move_id != 0:
            # Player is executing a move - only allow movement, no new attacks
            attack_buttons = {'Y', 'B', 'X', 'A'}
            for btn in attack_buttons:
                filtered_buttons[btn] = False
            
            # Set move inhibit period
            self.move_inhibit_until = current_time + 0.3  # 300ms inhibit after move
            return filtered_buttons
        
        # 2. Move inhibit period (prevent immediate follow-up attacks)
        if current_time < self.move_inhibit_until:
            attack_buttons = {'Y', 'B', 'X', 'A'}
            
            # Only suppress some attacks during inhibit period
            for btn in attack_buttons:
                if filtered_buttons[btn] and np.random.random() < 0.6:  # 60% chance to suppress
                    filtered_buttons[btn] = False
        
        # 3. Prevent excessive button repetition
        current_action_string = str(sorted([k for k, v in filtered_buttons.items() if v]))
        last_action_string = str(sorted([k for k, v in self.last_button_map.items() if v]))
        
        if (current_action_string == last_action_string and 
            current_action_string != "[]" and
            current_action_string == self.last_significant_action):
            
            # Same action repeated - add some variation
            active_buttons = [k for k, v in filtered_buttons.items() if v]
            if active_buttons and len(active_buttons) > 1:
                # Randomly disable one button to create variation
                button_to_disable = np.random.choice(active_buttons)
                filtered_buttons[button_to_disable] = False
        
        # Store for next comparison
        self.last_button_map = filtered_buttons.copy()
        
        return filtered_buttons
    
    def act(self, gs):
        """Get action from evolved neural network with state-change based logic"""
        # Check if we should act based on state changes
        should_act, change_reason = self.state_tracker.has_significant_change(gs, self.player_id)
        
        if should_act:
            # Significant change detected - compute new action
            raw_buttons = self._get_raw_prediction(gs)
            filtered_buttons = self._filter_buttons_by_move_state(gs, raw_buttons)
            
            # Cache this action
            self.action_cache = filtered_buttons.copy()
            self.last_significant_action = str(sorted([k for k, v in filtered_buttons.items() if v]))
            
            # Debug output for significant actions
            active_buttons = [btn for btn, state in filtered_buttons.items() if state]
            if active_buttons:
                print(f"[NeuroBot] Action update ({change_reason}): {', '.join(active_buttons)}")
            
            return filtered_buttons
        else:
            # No significant change - return cached action or neutral
            if self.action_cache is not None:
                return self.action_cache
            else:
                # Return neutral action
                return {btn: False for btn in BUTTONS}
    
    def fight(self, gs, player_id):
        """Main method called by controller (same interface as RNN bot)"""
        button_map = self.act(gs)
        
        cmd = Command()
        if player_id == "1" or player_id == 1:
            cmd.player_buttons = Buttons(button_map)
        else:
            cmd.player2_buttons = Buttons(button_map)
        
        return cmd
    
    def get_command(self, gs, player_id):
        """Alternative interface for controller compatibility"""
        return self.fight(gs, player_id)