import numpy as np
from command import Command
from buttons import Buttons
from collections import deque
import tensorflow as tf  # or your preferred ML framework

# Constants matching preprocess_windows.py
STATE_COLS = [
    'timer','fight_result','has_round_started','is_round_over',
    'player1_id','p1_health','p1_x','p1_y','p1_jumping','p1_crouching','p1_in_move','p1_move_id',
    'player2_id','p2_health','p2_x','p2_y','p2_jumping','p2_crouching','p2_in_move','p2_move_id',
    'diff_x', 'diff_y', 'diff_health'
]
BUTTONS = ['up', 'down', 'right', 'left', 'select', 'start', 'y', 'b', 'x', 'a', 'l', 'r']
BUTTON_COLS = [f'player1_buttons_{b}' for b in BUTTONS] + [f'player2_buttons_{b}' for b in BUTTONS]

class Bot:
    def __init__(self, model_path='model.h5', window_size=6):
        # Load the trained model
        print(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.window_size = window_size
        
        # Try to load the scaler if it exists
        scaler_path = model_path.replace('.h5', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler_info = pickle.load(f)
            print(f"Loaded normalization scaler")
        else:
            print("No scaler found - using default normalization")
            self.scaler_info = None
        
        # Default normalization ranges (used if no scaler is available)
        self.norm_ranges = {
            'timer': 99,
            'health': 176,
            'x_coord': 400,
            'y_coord': 300,
            'character_id': 8
        }
        
        # Initialize the frame buffer
        self.frame_buffer = deque(maxlen=window_size)
        
        # Pre-fill with empty frames for cold start
        empty_frame = self._create_empty_frame()
        for _ in range(window_size):
            self.frame_buffer.append(empty_frame)
    
    def _create_empty_frame(self):
        """Create an empty frame with zeros for all features"""
        frame_data = {}
        for col in STATE_COLS:
            frame_data[col] = 0
        for col in BUTTON_COLS:
            frame_data[col] = 0
        return frame_data
    
    def _gamestate_to_frame(self, gs, player_id):
        """Convert GameState to our frame format with required features"""
        frame = {}
        
        # Extract state values from game state
        frame['timer'] = gs.timer
        frame['fight_result'] = gs.fight_result
        frame['has_round_started'] = 1 if gs.has_round_started else 0
        frame['is_round_over'] = 1 if gs.is_round_over else 0
        
        # Player 1 data
        frame['player1_id'] = gs.p1.character_id
        frame['p1_health'] = gs.p1.health
        frame['p1_x'] = gs.p1.x
        frame['p1_y'] = gs.p1.y
        frame['p1_jumping'] = 1 if gs.p1.jumping else 0
        frame['p1_crouching'] = 1 if gs.p1.crouching else 0
        frame['p1_in_move'] = 1 if gs.p1.in_move else 0
        frame['p1_move_id'] = gs.p1.move_id
        
        # Player 2 data
        frame['player2_id'] = gs.p2.character_id
        frame['p2_health'] = gs.p2.health
        frame['p2_x'] = gs.p2.x
        frame['p2_y'] = gs.p2.y
        frame['p2_jumping'] = 1 if gs.p2.jumping else 0
        frame['p2_crouching'] = 1 if gs.p2.crouching else 0
        frame['p2_in_move'] = 1 if gs.p2.in_move else 0
        frame['p2_move_id'] = gs.p2.move_id
        
        # Derived features
        frame['diff_x'] = gs.p1.x - gs.p2.x
        frame['diff_y'] = gs.p1.y - gs.p2.y
        frame['diff_health'] = gs.p1.health - gs.p2.health
        
        # Button states (these would need to be tracked from previous frames)
        for button in BUTTONS:
            frame[f'player1_buttons_{button}'] = 0  # We don't have this info
            frame[f'player2_buttons_{button}'] = 0  # We don't have this info
        
        return frame
    
    def fight(self, gs, player_id):
        """Main method called by controller each frame"""
        # Convert game state to our frame format
        current_frame = self._gamestate_to_frame(gs, player_id)
        
        # Add to frame buffer
        self.frame_buffer.append(current_frame)
        
        # Prepare input for the model
        model_input = []
        for frame in self.frame_buffer:
            # Add state features
            for col in STATE_COLS:
                model_input.append(frame[col])
            # Add button features
            for col in BUTTON_COLS:
                model_input.append(frame[col])
        
        # Make prediction
        model_input = np.array([model_input])  # Add batch dimension
        prediction = self.model.predict(model_input, verbose=0)[0]
        
        # Convert prediction to buttons
        button_dict = {}
        button_subset = BUTTONS  # Only the buttons for our player
        
        # If we're player 1, use first 12 outputs, else use last 12
        start_idx = 0 if player_id == '1' else len(BUTTONS)
        
        for i, button in enumerate(button_subset):
            # Apply threshold to get binary button press
            button_dict[button] = prediction[start_idx + i] > 0.5
        
        # Create command with buttons
        cmd = Command()
        cmd.player_buttons = Buttons(button_dict)
        return cmd