import pandas as pd
import os
def balance_button_distribution(df, threshold=0.55):
    """
    Balance the button press distribution in the dataframe by removing excess rows 
    where a button is pressed more than the threshold percentage.
    """
    button_cols = ['player1_buttons_up', 'player1_buttons_down', 
                  'player1_buttons_right', 'player1_buttons_left',
                  'player1_buttons_y', 'player1_buttons_b', 
                  'player1_buttons_x', 'player1_buttons_a',
                  'player1_buttons_l', 'player1_buttons_r']
    
    initial_rows = len(df)
    print(f"\nInitial rows: {initial_rows}")
    
    # Show initial distribution
    print("\nInitial button distribution:")
    for col in button_cols:
        press_ratio = df[col].mean()
        print(f"{col}: {press_ratio:.2%}")
    
    # Balance each button that exceeds threshold
    total_rows_removed = 0
    for col in button_cols:
        press_ratio = df[col].mean()
        if press_ratio > threshold:
            # Get rows where this button is pressed
            pressed_rows = df[df[col] == 1]
            current_count = len(pressed_rows)
            
            # Calculate target count to achieve threshold ratio
            target_count = int((threshold * len(df)) / (1 - threshold))
            rows_to_remove = current_count - target_count
            
            if rows_to_remove > 0:
                # Randomly sample rows to remove
                drop_indices = pressed_rows.sample(n=rows_to_remove, random_state=42).index
                df = df.drop(drop_indices)
                total_rows_removed += rows_to_remove
                
                print(f"\nBalanced {col}")
                print(f"Removed {rows_to_remove} rows")
                print(f"New ratio: {df[col].mean():.2%}")
    
    final_rows = len(df)
    kept_percent = (final_rows / initial_rows) * 100
    
    print(f"\nDataset Statistics:")
    print(f"Initial rows: {initial_rows}")
    print(f"Final rows: {final_rows}")
    print(f"Total rows removed: {total_rows_removed}")
    print(f"Kept {kept_percent:.1f}% of original data")
    
    return df

if __name__ == "__main__":
    # List of large datasets which are overfitting our models
    datasets_to_process = [0, 3, 8, 9]
    
    # Base directory containing datasets
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'flattened_window_datasets')
    )
    
    for idx in datasets_to_process:
        dataset_path = os.path.join(base_dir, f'windowed_dataset_{idx}.csv')
        
        if not os.path.exists(dataset_path):
            print(f"\nSkipping dataset {idx} - file not found")
            continue
            
        print(f"\nProcessing dataset {idx}")
        print(f"Loading {dataset_path}")
        
        # Read, balance and save back to same file
        df = pd.read_csv(dataset_path)
        df_balanced = balance_button_distribution(df)
        df_balanced.to_csv(dataset_path, index=False)
        print(f"Saved balanced dataset back to {dataset_path}")