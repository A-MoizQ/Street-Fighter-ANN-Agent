import pandas as pd
import numpy as np
import glob
import os

# Use lowercase button names to match CSV columns
BUTTONS = ['up', 'down', 'right', 'left', 'select', 'start', 'y', 'b', 'x', 'a', 'l', 'r']

STATE_COLS = [
    'timer','fight_result','has_round_started','is_round_over',
    'player1_id','p1_health','p1_x','p1_y','p1_jumping','p1_crouching','p1_in_move','p1_move_id',
    'player2_id','p2_health','p2_x','p2_y','p2_jumping','p2_crouching','p2_in_move','p2_move_id',
    'diff_x', 'diff_y', 'diff_health'
]
BUTTON_COLS = [f'player1_buttons_{b}' for b in BUTTONS] + [f'player2_buttons_{b}' for b in BUTTONS]

def clean_dataset(df):
    """Remove redundant frames and non-gameplay frames while preserving round results"""
    initial_size = len(df)
    clean_indices = []
    last_state_signature = None
    include_round_end = True  # Flag to include first round end frame
    
    for i, row in df.iterrows():
        # Skip menu states (both players at 0 health or unrealistic positions)
        if (row['p1_health'] == 0 and row['p2_health'] == 0) or \
           (row['p1_x'] == 0 and row['p1_y'] == 0 and row['p2_x'] == 0 and row['p2_y'] == 0):
            continue
            
        # For round-over states, only include the first occurrence
        if row['is_round_over'] == True:
            if include_round_end:
                clean_indices.append(i)
                include_round_end = False  # Don't include more round end frames
            continue
        else:
            include_round_end = True  # Reset flag for next round end
        
        # For normal gameplay, check for duplicate frames
        state_signature = tuple(row[STATE_COLS])
        if state_signature != last_state_signature:
            clean_indices.append(i)
            last_state_signature = state_signature
    
    # Create new dataframe with selected indices
    clean_df = df.loc[clean_indices]
    
    # Report cleaning results
    print(f"Original frames: {initial_size}")
    print(f"After cleaning: {len(clean_df)} ({len(clean_df)/initial_size*100:.1f}% of original)")
    
    return clean_df


def create_windowed_dataset(input_csv: str, window_size: int = 6, output_csv: str = None):
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded input CSV: {input_csv}, shape: {df.shape}")
    except Exception as e:
        print(f"Error loading input CSV: {e}")
        return
    
    df = clean_dataset(df)
    rows = []

    for i in range(window_size - 1, len(df)):
        window = df.iloc[i - window_size + 1:i + 1]
        features = window[STATE_COLS + BUTTON_COLS].values.flatten()
        target = window[BUTTON_COLS].iloc[-1].values
        rows.append(np.concatenate([features, target]))

    feature_names = []
    for t in range(window_size):
        suffix = f"t-{window_size - 1 - t}" if window_size > 1 else 't'
        for col in STATE_COLS + BUTTON_COLS:
            feature_names.append(f'{col}_{suffix}')
    target_names = BUTTON_COLS
    all_cols = feature_names + target_names

    out_df = pd.DataFrame(rows, columns=all_cols)
    if output_csv is None:
        # Default output filename based on input
        base = os.path.basename(input_csv)
        if base.startswith("normalized_dataset_"):
            char_id = base[len("normalized_dataset_"):-len(".csv")]
            # Save to ../flattened_window_datasets/
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'flattened_window_datasets'))
            os.makedirs(base_dir, exist_ok=True)
            output_csv = os.path.join(base_dir, f"windowed_dataset_{char_id}.csv")
        else:
            output_csv = "windowed_dataset.csv"
    out_df.to_csv(output_csv, index=False)
    print(f"Windowed dataset saved to {output_csv}, shape: {out_df.shape}")

def process_all_character_datasets(window_size=6):
    # Look for datasets in ../normalized_character_datasets/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'normalized_character_datasets'))
    pattern = os.path.join(base_dir, "normalized_dataset_*.csv")
    for file in glob.glob(pattern):
        create_windowed_dataset(file, window_size=window_size)

if __name__ == "__main__":
    # Process all per-character datasets
    process_all_character_datasets(window_size=6)