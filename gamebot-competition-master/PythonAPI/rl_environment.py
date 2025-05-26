import numpy as np
import tensorflow as tf
from collections import deque
from game_state import GameState

class StreetFighterEnv:
    """RL Environment wrapper for Street Fighter II Turbo"""
    
    def __init__(self, player_id=0):
        self.player_id = player_id
        self.window_size = 6
        self.state_features = ['timer', 'fight_result', 'has_round_started', 'is_round_over',
                              'player1_id', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move_id',
                              'player2_id', 'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id',
                              'diff_x', 'diff_y', 'diff_health']
        
        # Define action space (meaningful button combinations)
        self.action_space = self._create_action_space()
        self.action_size = len(self.action_space)
        self.state_size = len(self.state_features)
        
        # Initialize state buffer
        self.state_buffer = deque(maxlen=self.window_size)
        self.reset_buffer()
        
        # Previous game state for reward calculation
        self.prev_health = {'p1': 100, 'p2': 100, 'timer': 6000}
        self.prev_distance = 0
        self.combo_count = 0
        self.prev_move_id = {'p1': 0, 'p2': 0}
        self.consecutive_idle = 0
        
        print(f"[RL Environment] Initialized with {self.action_size} actions")
    
    def _create_action_space(self):
        """Create meaningful action combinations"""
        actions = [
            # Basic movements
            {'name': 'IDLE', 'buttons': {}},
            {'name': 'LEFT', 'buttons': {'LEFT': True}},
            {'name': 'RIGHT', 'buttons': {'RIGHT': True}},
            {'name': 'UP', 'buttons': {'UP': True}},
            {'name': 'DOWN', 'buttons': {'DOWN': True}},
            {'name': 'CROUCH_LEFT', 'buttons': {'DOWN': True, 'LEFT': True}},
            {'name': 'CROUCH_RIGHT', 'buttons': {'DOWN': True, 'RIGHT': True}},
            {'name': 'JUMP_LEFT', 'buttons': {'UP': True, 'LEFT': True}},
            {'name': 'JUMP_RIGHT', 'buttons': {'UP': True, 'RIGHT': True}},
            
            # Basic attacks
            {'name': 'LIGHT_PUNCH', 'buttons': {'X': True}},
            {'name': 'MEDIUM_PUNCH', 'buttons': {'Y': True}},
            {'name': 'HEAVY_PUNCH', 'buttons': {'L': True}},
            {'name': 'LIGHT_KICK', 'buttons': {'A': True}},
            {'name': 'MEDIUM_KICK', 'buttons': {'B': True}},
            {'name': 'HEAVY_KICK', 'buttons': {'R': True}},
            
            # Moving attacks
            {'name': 'FORWARD_LIGHT_PUNCH', 'buttons': {'RIGHT': True, 'X': True}},
            {'name': 'FORWARD_MEDIUM_PUNCH', 'buttons': {'RIGHT': True, 'Y': True}},
            {'name': 'FORWARD_HEAVY_PUNCH', 'buttons': {'RIGHT': True, 'L': True}},
            {'name': 'FORWARD_LIGHT_KICK', 'buttons': {'RIGHT': True, 'A': True}},
            {'name': 'FORWARD_MEDIUM_KICK', 'buttons': {'RIGHT': True, 'B': True}},
            {'name': 'FORWARD_HEAVY_KICK', 'buttons': {'RIGHT': True, 'R': True}},
            
            # Backward attacks
            {'name': 'BACKWARD_LIGHT_PUNCH', 'buttons': {'LEFT': True, 'X': True}},
            {'name': 'BACKWARD_MEDIUM_PUNCH', 'buttons': {'LEFT': True, 'Y': True}},
            {'name': 'BACKWARD_LIGHT_KICK', 'buttons': {'LEFT': True, 'A': True}},
            
            # Crouching attacks
            {'name': 'CROUCH_LIGHT_PUNCH', 'buttons': {'DOWN': True, 'X': True}},
            {'name': 'CROUCH_MEDIUM_PUNCH', 'buttons': {'DOWN': True, 'Y': True}},
            {'name': 'CROUCH_HEAVY_PUNCH', 'buttons': {'DOWN': True, 'L': True}},
            {'name': 'CROUCH_LIGHT_KICK', 'buttons': {'DOWN': True, 'A': True}},
            {'name': 'CROUCH_MEDIUM_KICK', 'buttons': {'DOWN': True, 'B': True}},
            {'name': 'CROUCH_HEAVY_KICK', 'buttons': {'DOWN': True, 'R': True}},
            
            # Jumping attacks
            {'name': 'JUMP_LIGHT_PUNCH', 'buttons': {'UP': True, 'X': True}},
            {'name': 'JUMP_MEDIUM_PUNCH', 'buttons': {'UP': True, 'Y': True}},
            {'name': 'JUMP_HEAVY_PUNCH', 'buttons': {'UP': True, 'L': True}},
            {'name': 'JUMP_LIGHT_KICK', 'buttons': {'UP': True, 'A': True}},
            {'name': 'JUMP_MEDIUM_KICK', 'buttons': {'UP': True, 'B': True}},
            {'name': 'JUMP_HEAVY_KICK', 'buttons': {'UP': True, 'R': True}},
        ]
        return actions
    
    def reset_buffer(self):
        """Reset the state buffer"""
        empty_state = {feat: 0 for feat in self.state_features}
        empty_state['fight_result'] = 'NOT_OVER'
        for _ in range(self.window_size):
            self.state_buffer.append(empty_state.copy())
    
    def game_state_to_features(self, gs):
        """Convert GameState to feature dictionary"""
        p1, p2 = gs.player1, gs.player2
        return {
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
    
    def get_state_vector(self):
        """Convert state buffer to 3D array for RNN"""
        X_3d = []
        FIGHT_MAP = {'NOT_OVER': 0, 'P1': 1, 'P2': 2}
        
        for t in range(self.window_size):
            frame = self.state_buffer[t]
            features = []
            for feat in self.state_features:
                val = frame[feat]
                if feat == 'fight_result':
                    features.append(FIGHT_MAP[val])
                else:
                    features.append(float(val))
            X_3d.append(features)
        
        return np.array([X_3d])  # Shape: (1, window_size, n_features)
    
    def calculate_reward(self, gs, action_idx):
        """Calculate reward based on game state changes"""
        p1, p2 = gs.player1, gs.player2
        reward = 0
        
        # Health-based rewards (most important)
        p1_health_change = p1.health - self.prev_health['p1']
        p2_health_change = p2.health - self.prev_health['p2']
        
        if self.player_id == p1.player_id:
            # We are player 1
            reward += (p2_health_change * -15)  # Reward for dealing damage to opponent
            reward += (p1_health_change * -10)  # Penalty for taking damage
        else:
            # We are player 2
            reward += (p1_health_change * -15)  # Reward for dealing damage to opponent
            reward += (p2_health_change * -10)  # Penalty for taking damage
        
        # Distance control reward
        current_distance = abs(p1.x_coord - p2.x_coord)
        optimal_distance_min = 80
        optimal_distance_max = 150
        
        if optimal_distance_min <= current_distance <= optimal_distance_max:
            reward += 2  # Good distance
        else:
            distance_penalty = min(abs(current_distance - optimal_distance_min), 
                                 abs(current_distance - optimal_distance_max)) / 20
            reward -= distance_penalty
        
        # Combo detection and rewards
        if self.player_id == p1.player_id:
            if p1.is_player_in_move and p1.move_id != self.prev_move_id['p1']:
                if p2_health_change < 0:  # Opponent took damage
                    self.combo_count += 1
                    reward += 25  # Attack bonus
                    if self.combo_count > 1:
                        reward += 15 * self.combo_count  # Combo bonus
                else:
                    self.combo_count = 0
            else:
                if self.combo_count > 0:
                    self.combo_count = max(0, self.combo_count - 1)
        else:
            if p2.is_player_in_move and p2.move_id != self.prev_move_id['p2']:
                if p1_health_change < 0:
                    self.combo_count += 1
                    reward += 25
                    if self.combo_count > 1:
                        reward += 15 * self.combo_count
                else:
                    self.combo_count = 0
            else:
                if self.combo_count > 0:
                    self.combo_count = max(0, self.combo_count - 1)
        
        # Action-based rewards
        action_name = self.action_space[action_idx]['name']
        if action_name == 'IDLE':
            self.consecutive_idle += 1
            if self.consecutive_idle > 30:  # 0.5 seconds of idle
                reward -= 5  # Penalty for excessive idling
        else:
            self.consecutive_idle = 0
        
        # Round end rewards
        if gs.fight_result == 'P1' and self.player_id == p1.player_id:
            reward += 500
        elif gs.fight_result == 'P2' and self.player_id == p2.player_id:
            reward += 500
        elif gs.fight_result != 'NOT_OVER':
            reward -= 500
        
        # Time management
        timer_change = gs.timer - self.prev_health['timer']
        if timer_change < 0:  # Time is running out
            reward -= 0.5  # Small penalty to encourage decisive play
        
        # Update previous states
        self.prev_health = {'p1': p1.health, 'p2': p2.health, 'timer': gs.timer}
        self.prev_distance = current_distance
        self.prev_move_id = {'p1': p1.move_id, 'p2': p2.move_id}
        
        return reward
    
    def step(self, gs, action_idx):
        """Process one step in the environment"""
        # Add new state to buffer
        features = self.game_state_to_features(gs)
        self.state_buffer.append(features)
        
        # Calculate reward
        reward = self.calculate_reward(gs, action_idx)
        
        # Check if episode is done
        done = gs.is_round_over or gs.fight_result != 'NOT_OVER'
        
        # Get next state
        next_state = self.get_state_vector()
        
        return next_state, reward, done
    
    def action_to_buttons(self, action_idx):
        """Convert action index to button combination"""
        if 0 <= action_idx < len(self.action_space):
            return self.action_space[action_idx]['buttons']
        return {}