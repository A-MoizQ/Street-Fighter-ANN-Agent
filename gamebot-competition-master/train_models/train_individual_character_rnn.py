import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam

# defining constants
WINDOW_SIZE = 6
STATE_FEATURES = ['timer', 'fight_result', 'has_round_started', 'is_round_over',
                 'player1_id', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move_id',
                 'player2_id', 'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id',
                 'diff_x', 'diff_y', 'diff_health']

# buttons to be predicted by the model
BUTTONS = ['up', 'down', 'right', 'left', 'y', 'b', 'x', 'a', 'l', 'r']
P1_BUTTON_COLS = [f'player1_buttons_{b}' for b in BUTTONS]

def load_and_preprocess_data(csv_path):
    # load dataset
    df = pd.read_csv(csv_path)

    # select rows where buttons are pressed
    df_pos = df[df[P1_BUTTON_COLS].sum(axis=1) > 0]
    # select rows where buttons are not pressed
    df_neg_all = df[df[P1_BUTTON_COLS].sum(axis=1) == 0]
    # get a 2:1 for positives to negatives to reduce class imbalance
    neg_count = min(len(df_neg_all), len(df_pos) * 2)
    # get samples of negatives
    df_neg = df_neg_all.sample(n=neg_count, random_state=42)
    # concatenate positives and negatives and shuffle
    df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # encoding the features
    FIGHT_MAP = {'NOT_OVER': 0, 'P1': 1, 'P2': 2}
    BOOL_MAP = {False: 0, True: 1, 'False': 0, 'True': 1}
    for t in range(WINDOW_SIZE):
        # map fight_result
        col_fr = f'fight_result_t-{t}'
        df[col_fr] = df[col_fr].map(FIGHT_MAP)
        # map booleans
        for bf in ['has_round_started','is_round_over','p1_jumping','p1_crouching','p1_in_move',
                  'p2_jumping','p2_crouching','p2_in_move']:
            col_b = f'{bf}_t-{t}'
            df[col_b] = df[col_b].map(BOOL_MAP)
    
    # Extract features and targets
    y = df[P1_BUTTON_COLS].astype(int)
    
    # Reshape data for RNN (samples, time_steps, features)
    X_3d = []
    for t in range(WINDOW_SIZE):
        # Get features for this timestep
        timestep_cols = [f"{feat}_t-{t}" for feat in STATE_FEATURES]
        X_t = df[timestep_cols].values
        X_3d.append(X_t)
    
    # Convert to numpy array and transpose to get (samples, timesteps, features)
    X_3d = np.array(X_3d).transpose(1, 0, 2)
    
    return X_3d, y

def create_rnn_model(lstm_units=64, dense_units=32, dropout_rate=0.3, learning_rate=0.001):
    model = models.Sequential()
    # Add LSTM layer
    model.add(layers.LSTM(lstm_units, input_shape=(WINDOW_SIZE, len(STATE_FEATURES)), 
                         return_sequences=True))
    model.add(layers.Dropout(dropout_rate))
    # Add another LSTM layer
    model.add(layers.LSTM(lstm_units // 2, return_sequences=False))
    model.add(layers.Dropout(dropout_rate))
    # Add dense layers
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dense(len(P1_BUTTON_COLS), activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss='binary_crossentropy',
                 metrics=['binary_accuracy'])
    
    return model

def grid_search_best_model(X_train, y_train):
    """Manual implementation of grid search to avoid compatibility issues."""
    # Define hyperparameters to search
    param_grid = {
        'lstm_units': [32, 64, 128],
        'dense_units': [32, 64],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [64, 128]
    }
    
    # Sample a smaller set for grid search to speed things up
    sample_size = min(len(X_train), 5000)  # Using an even smaller sample for quick results
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train[indices]
    y_sample = y_train.iloc[indices]
    
    # Split the sample into training and validation sets
    X_train_gs, X_val_gs, y_train_gs, y_val_gs = train_test_split(
        X_sample, y_sample, test_size=0.3, random_state=42)
    
    best_accuracy = 0.0
    best_params = {}
    
    # Track all results for reporting
    all_results = []
    
    print("Starting manual grid search...")
    
    # Total combinations to try
    total_combinations = (len(param_grid['lstm_units']) * 
                         len(param_grid['dense_units']) * 
                         len(param_grid['dropout_rate']) * 
                         len(param_grid['learning_rate']) * 
                         len(param_grid['batch_size']))
    
    print(f"Testing {total_combinations} hyperparameter combinations")
    
    combo_count = 0
    
    # Manual grid search
    for lstm_units in param_grid['lstm_units']:
        for dense_units in param_grid['dense_units']:
            for dropout_rate in param_grid['dropout_rate']:
                for learning_rate in param_grid['learning_rate']:
                    for batch_size in param_grid['batch_size']:
                        combo_count += 1
                        print(f"\nTrying combination {combo_count}/{total_combinations}:")
                        print(f"  LSTM units: {lstm_units}, Dense units: {dense_units}, " +
                              f"Dropout: {dropout_rate}, LR: {learning_rate}, Batch size: {batch_size}")
                        
                        # Create model with current hyperparameters
                        model = create_rnn_model(
                            lstm_units=lstm_units,
                            dense_units=dense_units,
                            dropout_rate=dropout_rate,
                            learning_rate=learning_rate
                        )
                        
                        # Train for just a few epochs
                        history = model.fit(
                            X_train_gs, y_train_gs,
                            epochs=3,  # Reduced epochs for quick evaluation
                            batch_size=batch_size,
                            validation_data=(X_val_gs, y_val_gs),
                            verbose=0
                        )
                        
                        # Evaluate on validation set
                        loss, accuracy = model.evaluate(X_val_gs, y_val_gs, verbose=0)
                        print(f"  Validation accuracy: {accuracy:.4f}")
                        
                        # Store results
                        result = {
                            'lstm_units': lstm_units,
                            'dense_units': dense_units,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'accuracy': accuracy,
                            'loss': loss
                        }
                        all_results.append(result)
                        
                        # Update best parameters if this model is better
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                'lstm_units': lstm_units,
                                'dense_units': dense_units,
                                'dropout_rate': dropout_rate,
                                'learning_rate': learning_rate,
                                'batch_size': batch_size
                            }
    
    # Sort results by accuracy
    all_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
    
    # Print top 3 results
    print("\nTop 3 hyperparameter combinations:")
    for i, result in enumerate(all_results[:3]):
        print(f"{i+1}. LSTM={result['lstm_units']}, Dense={result['dense_units']}, " +
              f"Dropout={result['dropout_rate']}, LR={result['learning_rate']}, " +
              f"Batch={result['batch_size']} → Accuracy: {result['accuracy']:.4f}")
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    
    return best_params

def train_model(csv_path, model_path, models_dir):
    # Load and preprocess data
    X_3d, y = load_and_preprocess_data(csv_path)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_3d, y, test_size=0.2, random_state=42)
    
    # Scale features
    # We need to scale each feature, but we want to fit the scaler only on training data
    # We'll scale each feature across all time steps
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_val = X_val.shape[0]
    
    # Reshape to 2D for scaling
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_val_reshaped = X_val.reshape(-1, n_features)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
    X_val_scaled = X_val_scaled.reshape(n_samples_val, n_timesteps, n_features)
    
    # Save scaler
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, os.path.basename(model_path) + '.scaler')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Find best hyperparameters
    print("Running grid search to find optimal hyperparameters...")
    best_params = grid_search_best_model(X_train_scaled, y_train)
    
    # Create model with best parameters
    model = create_rnn_model(
        lstm_units=best_params['lstm_units'],
        dense_units=best_params['dense_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )
    
    # Give more weight to samples with button presses
    sample_weight = (y_train.sum(axis=1) > 0).astype(float) * 4 + 1
    
    # Checkpoint to only save the best model
    os.makedirs(models_dir, exist_ok=True)
    model_path_full = os.path.join(models_dir, os.path.basename(model_path))
    ckpt = callbacks.ModelCheckpoint(
        filepath=model_path_full,
        monitor='val_binary_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    model.fit(
        X_train_scaled, y_train,
        sample_weight=sample_weight,
        validation_data=(X_val_scaled, y_val),
        epochs=50,
        batch_size=best_params['batch_size'],
        callbacks=[ckpt, early_stopping]
    )
    
    print(f"Training done. Best model at {model_path_full}")
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_val_scaled, y_val)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    return model_path_full

if __name__ == '__main__':
    # set this according to the characters you want to train
    characters_to_train = [9,1] 

    # get file paths for datasets and where to save the models
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'flattened_window_datasets'))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RNN_models'))
    os.makedirs(out, exist_ok=True)
    
    # process only specific characters using their datasets
    for cid in characters_to_train:
        fn = f'windowed_dataset_{cid}.csv'
        if not os.path.exists(os.path.join(base, fn)):
            print(f"\n=== Skipping character {cid} - dataset not found ===")
            continue
            
        inp = os.path.join(base, fn)
        mdl = f'model_{cid}.keras'
        print(f"\n=== Training RNN model for character {cid} ===")
        train_model(inp, mdl, out)