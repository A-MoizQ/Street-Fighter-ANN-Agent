import numpy as np
import tensorflow as tf
import os
import pickle
import random
import time
from collections import deque
import joblib

class GeneticAlgorithm:
    """Genetic Algorithm for evolving neural network weights"""
    
    def __init__(self, player_id=0, population_size=20, mutation_rate=0.1, crossover_rate=0.7, fps=3.0):
        self.player_id = player_id
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
        self.fps = fps
        
        # Load base model to get weight structure (same path as RNN bot)
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RNN_models'))
        base_model_path = os.path.join(base, f'model_{player_id}.keras')
        
        print(f"[Genetic Algorithm] Loading base model from: {base_model_path}")
        self.base_model = tf.keras.models.load_model(base_model_path)
        self.base_weights = self.base_model.get_weights()
        
        print(f"[Genetic Algorithm] Base model loaded with {len(self.base_weights)} weight matrices")
        for i, w in enumerate(self.base_weights):
            print(f"  Weight matrix {i}: shape {w.shape}, dtype {w.dtype}")
        
        # Initialize population
        self.population = self._create_initial_population()
        self.fitness_scores = np.zeros(population_size)
        self.best_individual = None
        self.best_fitness = -float('inf')
        
        # Global best tracking (across all generations)
        self.global_best_individual = None
        self.global_best_fitness = -float('inf')
        self.global_best_generation = 0
        
        # Evolution history
        self.fitness_history = []
        self.best_fitness_history = []
        
        # Save directory
        self.save_dir = os.path.join('..', 'neuroevolution_models')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"[Genetic Algorithm] Initialized for character {player_id}")
        print(f"[Genetic Algorithm] Population size: {population_size}")
        print(f"[Genetic Algorithm] Mutation rate: {mutation_rate}")
        print(f"[Genetic Algorithm] Crossover rate: {crossover_rate}")
        print(f"[Genetic Algorithm] Target FPS: {fps}")
    
    def _create_initial_population(self):
        """Create initial population with variations of base weights"""
        population = []
        
        for i in range(self.population_size):
            if i == 0:
                # First individual is the original model
                individual = [w.copy() for w in self.base_weights]
            else:
                # Create variations by adding noise to base weights
                individual = []
                for weight_matrix in self.base_weights:
                    noise_scale = 0.1  # 10% noise
                    noise = np.random.normal(0, noise_scale, weight_matrix.shape)
                    mutated_weights = weight_matrix + noise
                    individual.append(mutated_weights.astype(weight_matrix.dtype))
            
            population.append(individual)
        
        print(f"[Genetic Algorithm] Created population with {len(population)} individuals")
        return population
    
    def _mutate(self, individual):
        """Apply mutation to an individual"""
        mutated = []
        for weight_matrix in individual:
            if np.random.random() < self.mutation_rate:
                # Apply Gaussian noise to random subset of weights
                mutation_mask = np.random.random(weight_matrix.shape) < 0.1  # Mutate 10% of weights
                noise = np.random.normal(0, 0.05, weight_matrix.shape)  # Small mutations
                mutated_weights = weight_matrix.copy()
                mutated_weights[mutation_mask] += noise[mutation_mask]
                mutated.append(mutated_weights.astype(weight_matrix.dtype))
            else:
                mutated.append(weight_matrix.copy())
        return mutated
    
    def _crossover(self, parent1, parent2):
        """Create offspring through crossover"""
        if np.random.random() > self.crossover_rate:
            return parent1, parent2
        
        offspring1, offspring2 = [], []
        
        for w1, w2 in zip(parent1, parent2):
            # Random crossover point for each weight matrix
            if len(w1.shape) > 1:  # Matrix
                crossover_point = np.random.randint(0, w1.shape[0])
                child1 = w1.copy()
                child2 = w2.copy()
                child1[crossover_point:] = w2[crossover_point:]
                child2[crossover_point:] = w1[crossover_point:]
            else:  # Vector
                crossover_point = np.random.randint(0, len(w1))
                child1 = w1.copy()
                child2 = w2.copy()
                child1[crossover_point:] = w2[crossover_point:]
                child2[crossover_point:] = w1[crossover_point:]
            
            offspring1.append(child1)
            offspring2.append(child2)
        
        return offspring1, offspring2
    
    def _select_parents(self):
        """Tournament selection for choosing parents"""
        tournament_size = 3
        parents = []
        
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(
                self.population_size, tournament_size, replace=False
            )
            tournament_fitness = self.fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append([w.copy() for w in self.population[winner_idx]])
        
        return parents
    
    def evolve_generation(self):
        """Evolve to next generation"""
        # Select parents
        parents = self._select_parents()
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Keep best individual (elitism)
        if self.best_individual is not None:
            new_population.append([w.copy() for w in self.best_individual])
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            offspring1, offspring2 = self._crossover(parent1, parent2)
            
            # Apply mutation
            offspring1 = self._mutate(offspring1)
            offspring2 = self._mutate(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        print(f"[Genetic Algorithm] Generation {self.generation} created")
    
    def update_fitness(self, individual_idx, fitness):
        """Update fitness score for an individual"""
        self.fitness_scores[individual_idx] = fitness
        
        # Update best individual for this generation
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_individual = [w.copy() for w in self.population[individual_idx]]
            print(f"[Genetic Algorithm] New generation best fitness: {fitness:.2f}")
        
        # Update global best individual across all generations
        if fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_individual = [w.copy() for w in self.population[individual_idx]]
            self.global_best_generation = self.generation
            print(f"[Genetic Algorithm] NEW GLOBAL BEST FITNESS: {fitness:.2f} (Generation {self.generation})")
    
    def get_individual_weights(self, individual_idx):
        """Get weights for a specific individual"""
        return self.population[individual_idx]
    
    def save_global_best(self):
        """Save the global best individual across all generations"""
        if self.global_best_individual is None:
            return None
        
        filepath = os.path.join(self.save_dir, f'global_best_char_{self.player_id}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(self.global_best_individual, f)
        
        # Also save metadata
        metadata = {
            'fitness': self.global_best_fitness,
            'generation': self.global_best_generation,
            'character_id': self.player_id,
            'save_time': time.time()
        }
        metadata_path = os.path.join(self.save_dir, f'global_best_char_{self.player_id}_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"[Genetic Algorithm] Global best individual saved to {filepath}")
        print(f"[Genetic Algorithm] Global best fitness: {self.global_best_fitness:.2f} from generation {self.global_best_generation}")
        return filepath
    
    def get_generation_stats(self):
        """Get statistics for current generation"""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'global_best_fitness': self.global_best_fitness,
            'global_best_generation': self.global_best_generation,
            'avg_fitness': np.mean(self.fitness_scores),
            'std_fitness': np.std(self.fitness_scores),
            'population_size': self.population_size
        }

class NeuroevolutionTrainer:
    """Trainer that uses genetic algorithm to evolve neural networks"""
    
    def __init__(self, player_id=0, population_size=20, generations=50, fps=3.0):
        self.player_id = player_id
        self.population_size = population_size
        self.max_generations = generations
        self.fps = fps
        
        # Initialize genetic algorithm
        self.genetic_algorithm = GeneticAlgorithm(
            player_id=player_id,
            population_size=population_size,
            mutation_rate=0.15,  # Higher mutation for exploration
            crossover_rate=0.8,   # High crossover rate
            fps=fps
        )
        
        # Current evaluation state
        self.current_individual = 0
        self.current_weights_path = None
        
        # Fitness tracking
        self.fitness_evaluator = FitnessEvaluator(fps=fps)
        
        # Training state
        self.episode_data = {
            'start_time': time.time(),
            'steps': 0,
            'start_health': None,
            'start_opponent_health': None,
            'total_damage_dealt': 0,
            'total_damage_received': 0,
            'actions_taken': deque(maxlen=100),  # Store button combinations
            'distances': deque(maxlen=100),
            'defensive_actions': 0,
            'aggressive_actions': 0,
            'movement_actions': 0,
            'idle_actions': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'rounds_completed': 0
        }
        
        print(f"[Neuroevolution Trainer] Initialized for character {player_id}")
        print(f"[Neuroevolution Trainer] Population: {population_size}, Generations: {generations}")
        print(f"[Neuroevolution Trainer] Target FPS: {fps}")
    
    def start_evaluation(self, individual_idx):
        """Start evaluating a specific individual"""
        self.current_individual = individual_idx
        
        # Save current individual's weights
        weights = self.genetic_algorithm.get_individual_weights(individual_idx)
        self.current_weights_path = os.path.join(
            self.genetic_algorithm.save_dir, 
            f'temp_individual_{individual_idx}.pkl'
        )
        
        with open(self.current_weights_path, 'wb') as f:
            pickle.dump(weights, f)
        
        # Reset episode data
        self.episode_data = {
            'start_time': time.time(),
            'steps': 0,
            'start_health': None,
            'start_opponent_health': None,
            'total_damage_dealt': 0,
            'total_damage_received': 0,
            'actions_taken': deque(maxlen=100),
            'distances': deque(maxlen=100),
            'defensive_actions': 0,
            'aggressive_actions': 0,
            'movement_actions': 0,
            'idle_actions': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'rounds_completed': 0
        }
        
        print(f"[Neuroevolution Trainer] Evaluating individual {individual_idx + 1}/{self.population_size}")
    
    def _analyze_button_combination(self, button_map):
        """Analyze button combination to determine action type (based on RNN bot button structure)"""
        active_buttons = [btn for btn, state in button_map.items() if state]
        
        # Convert to set for easier checking
        active_set = set(active_buttons)
        
        # AGGRESSIVE ACTIONS (Attack buttons)
        attack_buttons = {'Y', 'B', 'X', 'A'}  # Punch/Kick buttons
        if active_set.intersection(attack_buttons):
            return 'aggressive'
        
        # DEFENSIVE ACTIONS
        # Crouching (defensive)
        if 'DOWN' in active_set and len(active_set) == 1:
            return 'defensive'
        
        # Blocking combinations (backward + button or just backward when opponent close)
        if 'LEFT' in active_set and len(active_set) == 1:  # Backing away
            return 'defensive'
        
        # MOVEMENT ACTIONS
        movement_buttons = {'UP', 'DOWN', 'LEFT', 'RIGHT'}
        if active_set.intersection(movement_buttons) and not active_set.intersection(attack_buttons):
            return 'movement'
        
        # IDLE/NO ACTION
        if len(active_set) == 0:
            return 'idle'
        
        # DEFAULT: Mixed or complex combination
        return 'mixed'
    
    def update_episode_data(self, gs, action_taken):
        """Update episode data during evaluation"""
        self.episode_data['steps'] += 1
        
        # Determine player and opponent based on player_id
        if self.player_id == gs.player1.player_id:
            player = gs.player1
            opponent = gs.player2
        else:
            player = gs.player2
            opponent = gs.player1
        
        # Initialize starting health
        if self.episode_data['start_health'] is None:
            self.episode_data['start_health'] = player.health
            self.episode_data['start_opponent_health'] = opponent.health
        
        # Track damage
        if hasattr(self, 'prev_health'):
            damage_received = max(0, self.prev_health - player.health)
            self.episode_data['total_damage_received'] += damage_received
        
        if hasattr(self, 'prev_opponent_health'):
            damage_dealt = max(0, self.prev_opponent_health - opponent.health)
            self.episode_data['total_damage_dealt'] += damage_dealt
        
        # Store health for next frame
        self.prev_health = player.health
        self.prev_opponent_health = opponent.health
        
        # Track distance
        distance = abs(player.x_coord - opponent.x_coord)
        self.episode_data['distances'].append(distance)
        
        # Track and analyze actions
        self.episode_data['actions_taken'].append(action_taken)
        
        # Analyze button combination (action_taken is the button_map from RNN bot)
        action_type = self._analyze_button_combination(action_taken)
        
        if action_type == 'aggressive':
            self.episode_data['aggressive_actions'] += 1
        elif action_type == 'defensive':
            self.episode_data['defensive_actions'] += 1
        elif action_type == 'movement':
            self.episode_data['movement_actions'] += 1
        elif action_type == 'idle':
            self.episode_data['idle_actions'] += 1
        
        # Track wins/losses/draws with proper round completion tracking
        if gs.fight_result in ['P1', 'P2', 'DRAW']:
            if not hasattr(self, '_last_round_result') or self._last_round_result != gs.fight_result:
                # New round result detected
                self.episode_data['rounds_completed'] += 1
                
                if gs.fight_result == 'P1' and self.player_id == gs.player1.player_id:
                    self.episode_data['wins'] += 1
                elif gs.fight_result == 'P2' and self.player_id == gs.player2.player_id:
                    self.episode_data['wins'] += 1
                elif gs.fight_result == 'DRAW':
                    self.episode_data['draws'] += 1
                else:
                    self.episode_data['losses'] += 1
                
                self._last_round_result = gs.fight_result
        else:
            self._last_round_result = gs.fight_result
    
    def calculate_fitness(self):
        """Calculate fitness based on episode performance"""
        return self.fitness_evaluator.calculate_comprehensive_fitness(self.episode_data)
    
    def end_evaluation(self):
        """End evaluation and update fitness"""
        fitness = self.calculate_fitness()
        self.genetic_algorithm.update_fitness(self.current_individual, fitness)
        
        print(f"[Neuroevolution Trainer] Individual {self.current_individual + 1} fitness: {fitness:.2f}")
        
        # Print detailed stats
        print(f"  Wins: {self.episode_data['wins']}, Losses: {self.episode_data['losses']}, Draws: {self.episode_data['draws']}")
        print(f"  Rounds: {self.episode_data['rounds_completed']}")
        print(f"  Damage: Dealt={self.episode_data['total_damage_dealt']}, Received={self.episode_data['total_damage_received']}")
        print(f"  Actions: Aggressive={self.episode_data['aggressive_actions']}, Defensive={self.episode_data['defensive_actions']}, Movement={self.episode_data['movement_actions']}, Idle={self.episode_data['idle_actions']}")
        
        # Cleanup temporary file
        if self.current_weights_path and os.path.exists(self.current_weights_path):
            os.remove(self.current_weights_path)
        
        return fitness
    
    def should_evolve_generation(self):
        """Check if all individuals have been evaluated"""
        return self.current_individual >= self.population_size - 1
    
    def evolve_next_generation(self):
        """Evolve to next generation"""
        self.genetic_algorithm.evolve_generation()
        
        # Always save global best (not generation-specific)
        best_path = self.genetic_algorithm.save_global_best()
        
        # Reset for next generation
        self.current_individual = 0
        
        # Print generation statistics
        stats = self.genetic_algorithm.get_generation_stats()
        print(f"\n[Neuroevolution Trainer] ===== GENERATION {stats['generation']} COMPLETE =====")
        print(f"[Neuroevolution Trainer] Generation Best Fitness: {stats['best_fitness']:.2f}")
        print(f"[Neuroevolution Trainer] Global Best Fitness: {stats['global_best_fitness']:.2f} (Gen {stats['global_best_generation']})")
        print(f"[Neuroevolution Trainer] Average Fitness: {stats['avg_fitness']:.2f}")
        print(f"[Neuroevolution Trainer] Standard Deviation: {stats['std_fitness']:.2f}")
        if best_path:
            print(f"[Neuroevolution Trainer] Global best saved to: {best_path}")
        
        return stats['generation'] >= self.max_generations

class FitnessEvaluator:
    """Comprehensive fitness evaluation for neuroevolution"""
    
    def __init__(self, fps=3.0):
        self.fps = fps
    
    def calculate_comprehensive_fitness(self, episode_data):
        """Calculate fitness based on multiple criteria with FPS adjustments"""
        fitness = 0.0
        
        # Check for invalid short evaluations
        min_steps = int(10 * self.fps)  # At least 10 seconds of evaluation
        if episode_data['steps'] <= min_steps:
            print(f"[WARNING] Very short evaluation ({episode_data['steps']} steps), giving minimal fitness")
            return 10.0
        
        # 1. WIN/LOSS/DRAW OUTCOMES (50% of fitness - increased importance)
        total_rounds = episode_data['rounds_completed']
        if total_rounds > 0:
            win_rate = episode_data['wins'] / total_rounds
            loss_rate = episode_data['losses'] / total_rounds
            draw_rate = episode_data['draws'] / total_rounds
            
            # Heavily reward wins, moderately reward draws, penalize losses
            fitness += win_rate * 200      # Up to +200 for 100% win rate
            fitness += draw_rate * 100     # Up to +100 for 100% draw rate
            fitness -= loss_rate * 50      # Up to -50 for 100% loss rate
        
        # 2. DAMAGE EFFICIENCY (25% of fitness)
        damage_dealt = episode_data['total_damage_dealt']
        damage_received = episode_data['total_damage_received']
        
        # Scale damage rewards by FPS (more time = more opportunities)
        time_factor = min(episode_data['steps'] / (60 * self.fps), 1.0)  # Normalize to 1 minute
        
        fitness += damage_dealt * 1.0 * time_factor
        fitness -= damage_received * 0.5 * time_factor
        
        # Damage ratio bonus
        if damage_received > 0:
            damage_ratio = damage_dealt / damage_received
            fitness += min(damage_ratio * 20, 50)  # Cap at +50
        elif damage_dealt > 0:
            fitness += 30  # Bonus for dealing damage without taking any
        
        # 3. ACTION DIVERSITY (15% of fitness)
        total_actions = episode_data['steps']
        if total_actions > 0:
            # Calculate action entropy to measure diversity
            action_counts = [
                episode_data['aggressive_actions'],
                episode_data['defensive_actions'], 
                episode_data['movement_actions'],
                episode_data['idle_actions']
            ]
            
            # Penalize excessive repetition
            if episode_data['actions_taken']:
                action_strings = [str(sorted([k for k, v in action.items() if v])) 
                                for action in episode_data['actions_taken']]
                unique_actions = len(set(action_strings))
                total_recorded = len(action_strings)
                
                if total_recorded > 10:
                    diversity_ratio = unique_actions / total_recorded
                    if diversity_ratio > 0.6:  # Good diversity
                        fitness += 25
                    elif diversity_ratio < 0.2:  # Poor diversity (button mashing)
                        fitness -= 30
            
            # Penalize excessive idle time
            idle_ratio = episode_data['idle_actions'] / total_actions
            if idle_ratio > 0.4:  # More than 40% idle
                fitness -= (idle_ratio - 0.4) * 60
        
        # 4. COMBAT ENGAGEMENT (10% of fitness)
        if episode_data['distances']:
            avg_distance = np.mean(episode_data['distances'])
            optimal_distance = 120  # Slightly increased for low FPS
            distance_score = max(0, 25 - abs(avg_distance - optimal_distance) * 0.1)
            fitness += distance_score
        
        # 5. ROUND COMPLETION BONUS
        if episode_data['rounds_completed'] > 0:
            fitness += episode_data['rounds_completed'] * 15  # +15 per completed round
        
        # 6. LONGEVITY BONUS (for surviving evaluation period)
        evaluation_time = time.time() - episode_data['start_time']
        expected_time = episode_data['steps'] / self.fps
        if evaluation_time >= expected_time * 0.8:  # Survived most of evaluation
            longevity_bonus = min(evaluation_time * 0.2, 25)
            fitness += longevity_bonus
        
        return max(fitness, 0)  # Ensure non-negative fitness