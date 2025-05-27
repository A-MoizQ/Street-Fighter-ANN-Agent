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


    def calculate_contextual_reward(self, gs, action_idx, base_reward):
        """Calculate enhanced rewards based on combat effectiveness"""
        
        # Get action details
        action_info = self.action_space[action_idx]
        action_name = action_info['name'].upper()
        
        # Initialize contextual reward
        contextual_reward = base_reward
        reward_breakdown = []
        
        # Get player states
        if self.player_id == 0:
            player = gs.player1
            opponent = gs.player2
        else:
            player = gs.player2  
            opponent = gs.player1
        
        # Calculate distance between players
        distance = abs(player.x_coord - opponent.x_coord)
        close_range = distance < 50   # Very close combat range
        medium_range = 50 <= distance < 120  # Medium range
        far_range = distance >= 120   # Far range
        
        # Get opponent activity state
        opponent_attacking = self._opponent_attacking(gs)
        opponent_move_type = self._get_opponent_move_type(gs)
        
        # Check for damage dealt/received this frame
        damage_dealt = 0
        damage_received = 0
        if hasattr(self, 'prev_opponent_health') and hasattr(self, 'prev_player_health'):
            damage_dealt = max(0, self.prev_opponent_health - opponent.health)
            damage_received = max(0, self.prev_player_health - player.health)
        
        # 1. ATTACK EFFECTIVENESS BONUSES
        if any(keyword in action_name for keyword in ['PUNCH', 'KICK', 'SPECIAL']):
            if damage_dealt > 0:
                # SUCCESSFUL HIT BONUS
                if 'HEAVY' in action_name or 'SPECIAL' in action_name:
                    contextual_reward += 8.0 + (damage_dealt * 0.2)  # Heavy hit bonus
                    reward_breakdown.append(f"Heavy Hit (+{damage_dealt} dmg): +{8.0 + (damage_dealt * 0.2):.1f}")
                else:
                    contextual_reward += 5.0 + (damage_dealt * 0.15)  # Light/medium hit bonus
                    reward_breakdown.append(f"Attack Hit (+{damage_dealt} dmg): +{5.0 + (damage_dealt * 0.15):.1f}")
                
                # COUNTER-ATTACK BONUS: Hit while opponent was attacking
                if opponent_attacking:
                    contextual_reward += 4.0
                    reward_breakdown.append("Counter-Attack: +4.0")
                    
            else:
                # MISSED ATTACK PENALTY (range and situation based)
                miss_penalty = 0.5
                
                if close_range:
                    miss_penalty = 0.3  # Light penalty for close miss (normal)
                elif far_range and any(keyword in action_name for keyword in ['LIGHT', 'MEDIUM']):
                    miss_penalty = 2.5  # Heavy penalty for attacking at wrong range
                elif opponent_attacking and close_range:
                    miss_penalty = 1.5  # Penalty for missing counter-attack opportunity
                
                contextual_reward -= miss_penalty
                reward_breakdown.append(f"Attack Miss: -{miss_penalty}")
        
        # 2. DEFENSIVE EFFECTIVENESS
        elif action_name in ['CROUCH', 'DOWN'] or 'BLOCK' in action_name:
            if opponent_attacking:
                # SUCCESSFUL DEFENSE: Defended while opponent attacked
                defense_bonus = 2.0
                
                # Use generic move categories instead of hardcoded names
                if "projectile_special" in opponent_move_type:
                    defense_bonus = 4.5  # Major bonus for defending against projectiles
                    reward_breakdown.append("Defended Projectile: +4.5")
                elif "super_move" in opponent_move_type:
                    defense_bonus = 5.0  # Huge bonus for defending against supers
                    reward_breakdown.append("Defended Super Move: +5.0")
                elif "command_special" in opponent_move_type:
                    defense_bonus = 3.5  # Good bonus for defending against command moves
                    reward_breakdown.append("Defended Special: +3.5")
                elif opponent_move_type == "jump_attack":
                    defense_bonus = 3.0  # Good bonus for anti-air defense
                    reward_breakdown.append("Anti-Air Defense: +3.0")
                else:
                    reward_breakdown.append("Good Defense: +2.0")

                    
                contextual_reward += defense_bonus
                
                # Extra bonus if we avoided damage
                if damage_received == 0:
                    contextual_reward += 1.0
                    reward_breakdown.append("Perfect Defense: +1.0")
                    
            else:
                contextual_reward -= 0.8  # Penalty for unnecessary defense
                reward_breakdown.append("Unnecessary Defense: -0.8")
        
        # 3. MOVEMENT EFFECTIVENESS
        elif action_name in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
            # STRATEGIC MOVEMENT
            if far_range:
                # Approaching when far away
                if ((action_name == 'RIGHT' and player.x_coord < opponent.x_coord) or 
                    (action_name == 'LEFT' and player.x_coord > opponent.x_coord)):
                    contextual_reward += 1.2  # Bonus for closing distance
                    reward_breakdown.append("Closing Distance: +1.2")
            
            # EVASIVE MOVEMENT: Moving away when opponent attacks
            elif opponent_attacking:
                if ((action_name == 'LEFT' and player.x_coord > opponent.x_coord) or
                    (action_name == 'RIGHT' and player.x_coord < opponent.x_coord)):
                    evasion_bonus = 1.5
                    
                    # Generic evasion bonuses
                    if "projectile_special" in opponent_move_type:
                        evasion_bonus = 3.0  # Big bonus for avoiding projectiles
                        reward_breakdown.append("Evaded Projectile: +3.0")
                    elif "super_move" in opponent_move_type:
                        evasion_bonus = 3.5  # Huge bonus for avoiding supers
                        reward_breakdown.append("Evaded Super: +3.5")
                    elif "command_special" in opponent_move_type:
                        evasion_bonus = 2.5  # Good bonus for avoiding command moves
                        reward_breakdown.append("Evaded Special: +2.5")
                    else:
                        reward_breakdown.append("Evasive Movement: +1.5")
                    contextual_reward += evasion_bonus
            
            # MOVEMENT SPAM PENALTY
            if hasattr(self, 'recent_movements'):
                recent_same_moves = sum(1 for m in self.recent_movements if m == action_name)
                if recent_same_moves >= 4:  # 4+ same movements
                    contextual_reward -= 3.0  # Heavy spam penalty
                    reward_breakdown.append("Movement Spam: -3.0")
            
            # BAD MOVEMENT PENALTIES
            if close_range and not opponent_attacking:
                if action_name in ['UP', 'DOWN']:
                    contextual_reward -= 1.5  # Bad movement when should be attacking
                    reward_breakdown.append("Poor Close Movement: -1.5")
        
        # 4. JUMPING EFFECTIVENESS
        elif 'JUMP' in action_name:
            if opponent.is_crouching and close_range:
                contextual_reward += 2.5  # Good jump over crouching opponent
                reward_breakdown.append("Jump Over Crouch: +2.5")
            elif opponent_move_type in ["crouch_attack", "basic_attack"] and close_range:
                contextual_reward += 2.0  # Jump to avoid low attacks
                reward_breakdown.append("Jump Evasion: +2.0")
            elif far_range:
                contextual_reward -= 1.0  # Risky jump from far away
                reward_breakdown.append("Risky Far Jump: -1.0")
            else:
                contextual_reward -= 0.3  # General jump penalty (slightly risky)
                reward_breakdown.append("Risky Jump: -0.3")
        
        # 5. IDLE PENALTIES
        elif action_name in ['IDLE', 'NO_ACTION', '']:
            idle_penalty = 0.5
            if close_range and not opponent_attacking:
                idle_penalty = 3.5  # Heavy penalty for inaction in combat range
                reward_breakdown.append("Idle in Combat: -3.5")
            elif medium_range and not opponent_attacking:
                idle_penalty = 1.5  # Medium penalty for inaction at medium range
                reward_breakdown.append("Idle at Medium Range: -1.5")
            else:
                reward_breakdown.append("General Idle: -0.5")
            
            contextual_reward -= idle_penalty
        
        # Store state for next frame comparison (IMPORTANT: Track positions)
        self.prev_opponent_health = opponent.health
        self.prev_player_health = player.health
        self.prev_opponent_x = opponent.x_coord
        self.prev_player_x = player.x_coord
        
        # Track recent actions for spam detection
        if not hasattr(self, 'recent_movements'):
            self.recent_movements = deque(maxlen=8)
        if not hasattr(self, 'recent_actions_detailed'):
            self.recent_actions_detailed = deque(maxlen=12)
        
        if action_name in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
            self.recent_movements.append(action_name)
        self.recent_actions_detailed.append(action_name)
        
        # Debug output for significant rewards/penalties
        if len(reward_breakdown) > 0 and abs(contextual_reward - base_reward) > 1.2:
            print(f"[Reward] {action_name}: Base={base_reward:.2f}, Final={contextual_reward:.2f}")
            for breakdown in reward_breakdown:
                print(f"  {breakdown}")
            print(f"  Distance: {distance:.0f}, Opponent: {'Attacking' if opponent_attacking else 'Passive'} ({opponent_move_type})")
            if damage_dealt > 0 or damage_received > 0:
                print(f"  Damage: Dealt={damage_dealt}, Received={damage_received}")
        
        return contextual_reward

    def _opponent_attacking(self, gs):
        """Check if opponent is currently in an attacking animation/state"""
        if self.player_id == 0:
            opponent = gs.player2
        else:
            opponent = gs.player1
        
        # Check if opponent is performing a move (attacking)
        is_in_move = opponent.is_player_in_move
        move_id = opponent.move_id
        
        # METHOD 1: Special moves (has move_id)
        if is_in_move and move_id != 0:
            # This is definitely a special move or complex animation
            return True
        
        # METHOD 2: Infer basic attacks from state changes
        # Since basic punches/kicks happen too fast for move_id to register,
        # we need to detect them differently
        
        # Check if opponent's health dealt damage recently (they're likely attacking)
        if hasattr(self, 'prev_player_health'):
            current_player = gs.player1 if self.player_id == 0 else gs.player2
            if current_player.health < self.prev_player_health:
                # Player took damage this frame - opponent was likely attacking
                return True
        
        # METHOD 3: Position-based attack detection
        # If opponent is very close and not moving defensively, they might be attacking
        if self.player_id == 0:
            player = gs.player1
            distance = abs(player.x_coord - opponent.x_coord)
        else:
            player = gs.player2
            distance = abs(player.x_coord - opponent.x_coord)
        
        # If opponent is in very close range and not crouching/jumping defensively
        if distance < 40 and not opponent.is_crouching and not opponent.is_jumping:
            # Check if opponent's position suggests aggressive action
            if hasattr(self, 'prev_opponent_x'):
                x_change = abs(opponent.x_coord - self.prev_opponent_x)
                if x_change > 2:  # Opponent moved toward player
                    return True
        
        return False

    def _get_opponent_move_type(self, gs):
        """Determine what type of move the opponent is performing (ADVANCED GENERIC)"""
        if self.player_id == 0:
            opponent = gs.player2
        else:
            opponent = gs.player1
        
        move_id = opponent.move_id
        is_in_move = opponent.is_player_in_move
        
        # Special moves detection (character-agnostic)
        if is_in_move and move_id != 0:
            
            # Analyze move characteristics based on opponent state + move_id
            move_category = self._categorize_move_by_id_and_state(move_id, opponent)
            
            if opponent.is_jumping:
                return f"jump_{move_category}"
            elif opponent.is_crouching:
                return f"crouch_{move_category}"
            else:
                return move_category
        
        # Basic attacks (inferred from context)
        elif self._opponent_attacking(gs):
            if opponent.is_jumping:
                return "jump_attack"
            elif opponent.is_crouching:
                return "crouch_attack"
            else:
                return "basic_attack"
        
        # Movement states
        elif opponent.is_jumping:
            return "jumping"
        elif opponent.is_crouching:
            return "crouching"
        else:
            return "idle"

    def _categorize_move_by_id_and_state(self, move_id, opponent):
        """Categorize moves based on ID ranges and character state (CHARACTER-AGNOSTIC)"""
        
        # Track move IDs we've seen to build dynamic categories
        if not hasattr(self, 'observed_move_patterns'):
            self.observed_move_patterns = {
                'projectile_ids': set(),
                'command_ids': set(),
                'super_ids': set(),
                'special_ids': set()
            }
        
        # Dynamic pattern recognition based on move_id characteristics
        
        # Very high IDs (30M+) are usually projectiles/fireballs
        if move_id >= 30000000:
            self.observed_move_patterns['projectile_ids'].add(move_id)
            return "projectile_special"
        
        # High IDs (1M-30M) are usually command moves (DP, HK, etc.)
        elif 1000000 <= move_id < 30000000:
            self.observed_move_patterns['command_ids'].add(move_id)
            return "command_special"
        
        # Medium IDs (100K-1M) might be supers or complex moves
        elif 100000 <= move_id < 1000000:
            self.observed_move_patterns['super_ids'].add(move_id)
            return "super_move"
        
        # Lower IDs might be special normals
        else:
            self.observed_move_patterns['special_ids'].add(move_id)
            return "special_move"

    def _get_move_threat_level(self, move_type, distance):
        """Determine how threatening a move is based on type and distance (UNIVERSAL)"""
        
        threat_levels = {
            # High threat moves
            "projectile_special": 0.9 if distance > 100 else 0.3,  # Dangerous at range
            "command_special": 0.8 if distance < 80 else 0.2,     # Dangerous up close
            "super_move": 0.95,  # Always dangerous
            
            # Medium threat moves  
            "jump_special": 0.7 if distance < 100 else 0.4,
            "crouch_special": 0.6 if distance < 60 else 0.2,
            "special_move": 0.6,
            
            # Basic attacks
            "jump_attack": 0.5 if distance < 80 else 0.1,
            "crouch_attack": 0.4 if distance < 50 else 0.1,
            "basic_attack": 0.4 if distance < 60 else 0.1,
            
            # Non-threatening
            "jumping": 0.2,
            "crouching": 0.1,
            "idle": 0.0
        }
        
        return threat_levels.get(move_type, 0.3)  # Default medium threat
        
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