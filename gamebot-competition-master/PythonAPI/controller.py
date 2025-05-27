import socket
import json
import sys
import time
from game_state import GameState
from command import Command
from buttons import Buttons
from listen_to_key import get_current_keypress
from make_dataset import record_frame

player_id = sys.argv[1]

# Enhanced mode detection
MODE = 'record' if len(sys.argv) > 2 and sys.argv[2] == 'record' else \
       'train_dqn' if len(sys.argv) > 2 and sys.argv[2] == 'train_dqn' else \
       'train_hybrid' if len(sys.argv) > 2 and sys.argv[2] == 'train_hybrid' else \
       'train_neuroevolution' if len(sys.argv) > 2 and sys.argv[2] == 'train_neuroevolution' else \
       'bot'

MODEL_TYPE = 'dqn' if len(sys.argv) > 2 and sys.argv[2] == 'dqn' else \
            'hybrid' if len(sys.argv) > 2 and sys.argv[2] == 'hybrid' else \
            'neuroevolution' if len(sys.argv) > 2 and sys.argv[2] == 'neuroevolution' else \
            'rnn' if len(sys.argv) > 2 and sys.argv[2] == 'rnn' else 'standard'

port = 9999 if player_id == '1' else 10000

def connect(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', port))
    s.listen(1)
    client, _ = s.accept()
    print('Connected to game!')
    return client

def send(sock, cmd):
    payload = json.dumps(cmd.object_to_dict())
    sock.sendall(payload.encode())

def receive(sock):
    data = json.loads(sock.recv(4096).decode())
    return GameState(data)

def send_command(sock, cmd):
    """Enhanced command sending with error handling"""
    try:
        send(sock, cmd)
    except Exception as e:
        print(f"[Controller] Error sending command: {e}")

def main():
    # Remove FPS measurement - use time-based approach instead
    print(f"[Controller] Starting in {MODE} mode with {MODEL_TYPE} model type")
    
    sock = connect(port)
    cmd = Command()
    player_id_set = False
    bot = None
    trainer = None
    
    # TIME-BASED neuroevolution variables (FPS independent)
    current_bot = None
    evaluation_start_time = time.time()
    max_evaluation_duration = 180.0  # 3 minutes per individual
    round_end_detected = False
    round_end_time = 0
    round_wait_duration = 30.0  # 30 seconds wait between rounds
    
    print(f"[Controller] Max evaluation duration: {max_evaluation_duration} seconds")
    print(f"[Controller] Round wait duration: {round_wait_duration} seconds")
    
    while True:
        try:
            gs = receive(sock)
            
            # Initialize trainer/bot based on mode
            if not player_id_set:
                character_id = gs.player1.player_id
                
                if MODE == 'train_dqn':
                    from dqn_trainer import DQNTrainer
                    trainer = DQNTrainer(player_id=character_id)
                    print(f"[Controller] DQN Training mode activated for character {character_id}")
                    
                elif MODE == 'train_hybrid':
                    from rnn_dqn_trainer import RNNDQNHybridTrainer
                    trainer = RNNDQNHybridTrainer(player_id=character_id, episodes=100)
                    print(f"[Controller] Hybrid RNN-DQN Training mode activated for character {character_id}")
                    
                elif MODE == 'train_neuroevolution':
                    from neuroevolution_trainer import NeuroevolutionTrainer
                    trainer = NeuroevolutionTrainer(
                        player_id=character_id, 
                        population_size=15,
                        generations=30,
                        fps=3.0  # Keep for compatibility but not used for timing
                    )
                    print(f"[Controller] Neuroevolution Training mode activated for character {character_id}")
                    print(f"[Controller] Population: {trainer.population_size}, Generations: {trainer.max_generations}")
                    
                elif MODE != 'record':
                    # Bot modes
                    if MODEL_TYPE == 'dqn':
                        from dqn_bot import DQNBot as Bot
                        print(f"[Controller] Using DQN reinforcement learning bot for character {character_id}")
                    elif MODEL_TYPE == 'hybrid':
                        from hybrid_bot import HybridBot as Bot
                        print(f"[Controller] Using Hybrid RNN-DQN bot for character {character_id}")
                    elif MODEL_TYPE == 'neuroevolution':
                        from neuroevolution_bot import NeuroevolutionBot as Bot
                        # Check for best evolved model
                        import os
                        evolved_models_dir = os.path.join('..', 'neuroevolution_models')
                        best_model_path = None
                        if os.path.exists(evolved_models_dir):
                            # Find the global best individual (across all generations)
                            model_files = [f for f in os.listdir(evolved_models_dir) 
                                         if f.startswith(f'global_best_char_{character_id}.pkl')]
                            if model_files:
                                best_model_path = os.path.join(evolved_models_dir, model_files[0])
                                print(f"[Controller] Found global best evolved model: {best_model_path}")
                        
                        bot = Bot(player_id=character_id, weights_path=best_model_path, fps=3.0)
                        print(f"[Controller] Using Neuroevolution bot for character {character_id}")
                    elif MODEL_TYPE == 'rnn':
                        from rnn_bot import Bot
                        print(f"[Controller] Using RNN model bot for character {character_id}")
                    else:
                        from bot import Bot
                        print(f"[Controller] Using standard model bot for character {character_id}")
                        
                    if MODEL_TYPE != 'neuroevolution':
                        bot = Bot(player_id=character_id)
                
                player_id_set = True

            # Handle different modes
            if MODE == 'record':
                keys = get_current_keypress()
                cmd.player_buttons = Buttons({k: True for k in keys})
                record_frame(gs, keys)
                
            elif MODE == 'train_dqn':
                cmd = trainer.train_step(gs, player_id)
                
            elif MODE == 'train_hybrid':
                cmd = trainer.train_step(gs, player_id)
                
            elif MODE == 'train_neuroevolution':
                # TIME-BASED neuroevolution training logic
                
                # Start new individual evaluation if needed
                if current_bot is None:
                    trainer.start_evaluation(trainer.current_individual)
                    from neuroevolution_bot import NeuroevolutionBot
                    current_bot = NeuroevolutionBot(
                        player_id=trainer.player_id, 
                        weights_path=trainer.current_weights_path,
                        fps=3.0  # Keep for compatibility
                    )
                    evaluation_start_time = time.time()
                    round_end_detected = False
                    round_end_time = 0
                    print(f"[Controller] Started evaluating individual {trainer.current_individual + 1}/{trainer.population_size}")
                    print(f"[Controller] Evaluation will run for {max_evaluation_duration} seconds")
                
                # Check round state and manage evaluation
                current_time = time.time()
                current_round_over = gs.fight_result in ['P1', 'P2', 'DRAW']
                
                # Detect first round end
                if current_round_over and not round_end_detected:
                    round_end_detected = True
                    round_end_time = current_time
                    print(f"[Controller] Round ended at {current_time - evaluation_start_time:.1f}s, result: {gs.fight_result}")
                    print(f"[Controller] Waiting {round_wait_duration} seconds for next round...")
                
                # If round ended, wait for new round to start with sufficient delay
                if round_end_detected:
                    time_since_round_end = current_time - round_end_time
                    
                    # Wait for both round restart AND sufficient delay
                    if (gs.has_round_started and 
                        not current_round_over and 
                        time_since_round_end >= round_wait_duration):
                        
                        print(f"[Controller] New round started after {time_since_round_end:.1f} seconds")
                        round_end_detected = False
                        round_end_time = 0
                        # Continue with normal evaluation
                    else:
                        # Still waiting - send neutral command
                        cmd = Command()
                        send_command(sock, cmd)
                        
                        # Show waiting progress every 10 seconds
                        if int(time_since_round_end) % 10 == 0 and time_since_round_end > 0:
                            remaining_wait = max(0, round_wait_duration - time_since_round_end)
                            print(f"[Controller] Waiting... {remaining_wait:.1f} seconds remaining")
                        
                        continue
                
                # Normal evaluation (when round is active)
                if gs.has_round_started and not current_round_over:
                    # Get action from current individual with STATE-CHANGE detection
                    cmd = current_bot.get_command(gs, player_id)
                    
                    # Update training data with the action taken
                    action_taken = current_bot.act(gs)
                    trainer.update_episode_data(gs, action_taken)
                else:
                    # Send neutral command when round not active
                    cmd = Command()
                
                # Check if evaluation should end (TIME-BASED)
                evaluation_duration = current_time - evaluation_start_time
                time_limit_reached = evaluation_duration >= max_evaluation_duration
                should_end = time_limit_reached
                
                if should_end:
                    # End current evaluation
                    fitness = trainer.end_evaluation()
                    current_bot = None
                    
                    end_reason = 'Time Limit' if time_limit_reached else 'Round Over'
                    print(f"[Controller] Individual {trainer.current_individual + 1} evaluation complete")
                    print(f"[Controller] Fitness: {fitness:.2f}, Duration: {evaluation_duration:.1f}s, Reason: {end_reason}")
                    
                    # Check if generation is complete
                    if trainer.should_evolve_generation():
                        training_complete = trainer.evolve_next_generation()
                        
                        if training_complete:
                            print("[Controller] Neuroevolution training complete!")
                            print(f"[Controller] Best fitness achieved: {trainer.genetic_algorithm.global_best_fitness:.2f}")
                            break
                    else:
                        trainer.current_individual += 1
                
                # Display progress periodically (TIME-BASED)
                if int(evaluation_duration) % 30 == 0 and evaluation_duration > 0:  # Every 30 seconds
                    progress = (trainer.current_individual / trainer.population_size) * 100
                    gen_progress = (trainer.genetic_algorithm.generation / trainer.max_generations) * 100
                    print(f"[Controller] Progress - Generation {trainer.genetic_algorithm.generation + 1}/{trainer.max_generations} ({gen_progress:.1f}%), "
                          f"Individual {trainer.current_individual + 1}/{trainer.population_size} ({progress:.1f}%), "
                          f"Duration: {evaluation_duration:.1f}s/{max_evaluation_duration}s")
                
            else:
                # Bot mode
                cmd = bot.fight(gs, player_id)
            
            # Send command to game
            send_command(sock, cmd)
            
            # Simple frame rate control - don't try to sync with variable FPS
            time.sleep(0.016)  # ~60 FPS target, but game will run at its own speed
            
        except KeyboardInterrupt:
            print("\n[Controller] Training interrupted by user")
            
            # Save progress for neuroevolution
            if MODE == 'train_neuroevolution' and trainer is not None:
                print("[Controller] Saving current progress...")
                trainer.genetic_algorithm.save_global_best()
                stats = trainer.genetic_algorithm.get_generation_stats()
                print(f"[Controller] Best fitness so far: {stats['global_best_fitness']:.2f}")
                print(f"[Controller] Generation: {stats['generation']}")
            
            break
            
        except Exception as e:
            print(f"[Controller] Error in main loop: {e}")
            time.sleep(0.1)

if __name__ == '__main__':
    main()