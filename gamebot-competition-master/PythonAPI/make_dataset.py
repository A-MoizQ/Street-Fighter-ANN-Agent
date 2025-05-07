import csv
import os
from game_state import GameState
from buttons import Buttons
from listen_to_key import get_current_keypress

# Define normalized fields
BUTTONS = ['UP', 'DOWN', 'RIGHT', 'LEFT', 'SELECT', 'START', 'Y', 'B', 'X', 'A', 'L', 'R']
FIELDNAMES = [
    'timer', 'fight_result', 'has_round_started', 'is_round_over',
    'player1_id', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move_id'
] + [f'player1_buttons_{b.lower()}' for b in BUTTONS] + [
    'player2_id', 'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id'
]  + [
    'diff_x', 'diff_y', 'diff_health'
]

_last_keys = None

def get_output_file(character_id):
    # Save to ../normalized_character_datasets/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'normalized_character_datasets'))
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f'normalized_dataset_{character_id}.csv')

def ensure_file_exists(filename):
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

def record_frame(gs: GameState, keys: list):
    global _last_keys
     
    # Optional: Skip frames with no input to balance dataset
    if not keys:
        return
        
    # Debug the raw keys received from listen_to_key.py
    print(f"Raw keys received: {keys}")
    # Store for next frame
    _last_keys = '+'.join(sorted(keys))
    
    # Build row dict with all game state data
    row = {
        'timer': gs.timer,
        'fight_result': gs.fight_result,
        'has_round_started': gs.has_round_started,
        'is_round_over': gs.is_round_over,
        'player1_id': gs.player1.player_id,
        'p1_health': gs.player1.health,
        'p1_x': gs.player1.x_coord,
        'p1_y': gs.player1.y_coord,
        'p1_jumping': gs.player1.is_jumping,
        'p1_crouching': gs.player1.is_crouching,
        'p1_in_move': gs.player1.is_player_in_move,
        'p1_move_id': gs.player1.move_id,
        'player2_id': gs.player2.player_id,
        'p2_health': gs.player2.health,
        'p2_x': gs.player2.x_coord,
        'p2_y': gs.player2.y_coord,
        'p2_jumping': gs.player2.is_jumping,
        'p2_crouching': gs.player2.is_crouching,
        'p2_in_move': gs.player2.is_player_in_move,
        'p2_move_id': gs.player2.move_id,
        'diff_x': gs.player1.x_coord - gs.player2.x_coord,
        'diff_y': gs.player1.y_coord - gs.player2.y_coord,
        'diff_health': gs.player1.health - gs.player2.health
    }

    # Initialize all button columns to False
    for b in BUTTONS:
        row[f'player1_buttons_{b.lower()}'] = False
        # row[f'player2_buttons_{b.lower()}'] = False
    
    # Map keys to Buttons for player1
    p1_buttons = Buttons({k: True for k in keys})  # No .upper() needed - already uppercase from get_current_keypress
    bd1 = p1_buttons.object_to_dict()
    
    # Debug the button mapping
    print(f"Button mapping: {bd1}")
    
    # Update the row with actual button values
    for b in BUTTONS:
        row[f'player1_buttons_{b.lower()}'] = bd1[b]
    
    # # For player2 from the game state (will be False for CPU - that's expected)
    # if hasattr(gs.player2, 'player_buttons'):
    #     bd2 = gs.player2.player_buttons.object_to_dict()
    #     for b in BUTTONS:
    #         row[f'player2_buttons_{b.lower()}'] = bd2[b]

    # Determine output file based on player1's character
    character_id = gs.player1.player_id
    output_file = get_output_file(character_id)
    ensure_file_exists(output_file)

    # Append row
    with open(output_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)