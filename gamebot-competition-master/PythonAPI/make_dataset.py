import csv
import os
from game_state import GameState
from listen_to_key import get_current_keypress

OUTPUT_FILE = 'dataset.csv'
FIELDNAMES = [
    'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move_id',
    'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id',
    'pressed_keys'
]

# Initialize the CSV file with header if it doesn't exist
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

_last_keys = None  # Cache previous key state

def record_frame(gs: GameState):
    """
    Reads a GameState instance, captures keypresses, and appends a row to the dataset
    only when keys change.
    """
    global _last_keys

    keys = get_current_keypress()
    key_str = '+'.join(sorted(keys))

    if not keys or key_str == _last_keys:
        return  # Skip empty or unchanged input

    _last_keys = key_str

    p1 = gs.player1
    p2 = gs.player2

    row = {
        'p1_health': p1.health,
        'p1_x': p1.x_coord,
        'p1_y': p1.y_coord,
        'p1_jumping': p1.is_jumping,
        'p1_crouching': p1.is_crouching,
        'p1_in_move': p1.is_player_in_move,
        'p1_move_id': p1.move_id,
        'p2_health': p2.health,
        'p2_x': p2.x_coord,
        'p2_y': p2.y_coord,
        'p2_jumping': p2.is_jumping,
        'p2_crouching': p2.is_crouching,
        'p2_in_move': p2.is_player_in_move,
        'p2_move_id': p2.move_id,
        'pressed_keys': key_str
    }

    # Append to CSV
    with open(OUTPUT_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)