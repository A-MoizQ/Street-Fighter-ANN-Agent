# Street Fighter II Turbo AI Bot

This project implements a deep learning-based AI system that can play Street Fighter II Turbo. The system records human gameplay, trains neural networks on these recordings, and then uses the trained models to control characters in the game.

## Project Overview

The project consists of several key components:

1. **Data Collection** - Record human gameplay to create training datasets
2. **Data Processing** - Normalize and transform gameplay data into suitable format for DL
3. **Model Training** - Train neural networks to predict button presses based on game state
4. **Bot Execution** - Use trained models to play the game automatically

## Model Architectures

The system supports two types of neural network architectures:

1. **Standard ANN** - Feed-forward neural network with dense layers (original implementation)
2. **RNN/LSTM** - Recurrent neural network that better captures temporal game state sequences

## Character Support

The system supports all characters in Street Fighter II Turbo:

| ID | Character |  | ID | Character |
|:--:|-----------|--|:--:|-----------|
| 0  |     Ryu   |  | 6  |  Zangief  |
| 1  |  E. Honda |  | 7  |  Dhalsim  |
| 2  |   Blanka  |  | 8  |  M. Bison |
| 3  |    Guile  |  | 9  |   Sagat   |
| 4  |     Ken   |  | 10 |   Balrog  |
| 5  |  Chun-Li  |  | 11 |   Vega    |

## Setup Instructions

### Prerequisites

- Python 3.6+
- TensorFlow 2.x
- Street Fighter II Turbo ROM
- BizHawk Emulator (EmuHawk.exe)

### Installation

1. Clone this repository
2. Install required Python packages:
   ```
   pip install tensorflow pandas numpy scikit-learn joblib pynput scikeras
   ```

## Running the Game

### Recording Gameplay (Creating Training Data)

1. Navigate to either the `single-player` or `two-players` folder
2. Run `EmuHawk.exe`
3. From the File menu, choose **Open ROM** (Ctrl+O)
4. Select the `Street Fighter II Turbo (U).smc` ROM file
5. From Tools menu, open the **Tool Box** (Shift+T)
6. **Open a command prompt in the `PythonAPI` directory** and run:
   ```
   python controller.py "1" "record"
   ```
7. Select your character in the game after choosing normal mode
8. Click on the **Gyroscope Bot** icon (second icon in the top row)
9. The emulator will connect to your program and show "Connected to game"
10. Play the game - your moves will be recorded to create the dataset

The recorded data will be saved in the `normalized_character_datasets` folder.

### Running the AI Bot

#### Standard ANN Bot
1. Follow steps 1-5 from the Recording Gameplay section
2. **Open a command prompt in the `PythonAPI` directory** and run:
   ```
   python controller.py "1"
   ```

#### RNN Bot
To use the RNN-based models instead of standard ANN, run:
```
python controller.py "1" "rnn"
```

3. Select your character in the game after choosing normal mode
4. Click on the **Gyroscope Bot** icon (second icon in the top row)
5. The AI bot will now take control and play the game

For two-player mode with player 2 controlled by AI, use:
```
python controller.py "2"       # Standard ANN
python controller.py "2" "rnn" # RNN model
```

## Project Structure

- **`PythonAPI/`** - Main code directory
  - bot.py - Standard ANN implementation
  - rnn_bot.py - RNN implementation using LSTM models
  - buttons.py - Button state representation
  - command.py - Command objects to send to game
  - controller.py - Main interface between game and system
  - game_state.py - Game state representation
  - player.py - Player state representation
  - listen_to_key.py - Keyboard input detection
  - make_dataset.py - Dataset creation utilities

- **`normalized_character_datasets/`** - Raw datasets for each character
- **`flattened_window_datasets/`** - Processed datasets ready for training
- **`models/`** - Trained standard ANN models
- **`RNN_models/`** - Trained RNN/LSTM models
- **`train_models/`** - Training scripts
  - train_individual_character.py - Train standard ANN models
  - train_individual_character_rnn.py - Train RNN models

- **`single-player/`** - Single-player game files
- **`two-players/`** - Two-player game files

## Model Training

### Standard ANN Training
To train standard ANN models for specific characters:

1. Ensure you have recorded gameplay data for the characters
2. Process the normalized datasets into windowed datasets
3. Edit train_individual_character.py to specify which character IDs to train
4. Run the training script:
   ```
   python train_models/train_individual_character.py
   ```

### RNN Training
To train RNN/LSTM models:

1. Ensure you have the flattened window datasets ready
2. Edit train_individual_character_rnn.py to specify which character IDs to train
3. Run the RNN training script:
   ```
   python train_models/train_individual_character_rnn.py
   ```
4. The script will perform a grid search to find optimal hyperparameters
5. Models will be saved to the `RNN_models/` directory

## Troubleshooting

- **Game not responding to AI commands**: Ensure the game is properly connected to the controller
- **Character not moving as expected**: Check that the model for that character has been properly trained
- **Connection errors**: Make sure you're running the correct port (9999 for player 1, 10000 for player 2)
- **ImportError with KerasClassifier**: Make sure you have installed the `scikeras` package

## Technical Details

### Standard ANN Architecture
- Uses a sliding window of 6 frames flattened into a 1D feature vector
- Neural networks use 3 dense layers with dropout for regularization
- Models are trained with class weighting to handle imbalanced button presses

### RNN Architecture
- Processes the 6-frame window as a temporal sequence (not flattened)
- Uses LSTM layers to capture temporal relationships between frames
- Better captures action sequences and player movement patterns
- Hyperparameter optimization via grid search to find optimal architecture
- Provides improved gameplay prediction by understanding time-dependent patterns

## Comparison of Model Types

| Feature | Standard ANN | RNN/LSTM |
|---------|-------------|----------|
| Temporal Understanding | Limited | Strong |
| Training Speed | Faster | Slower |
| Model Size | Smaller | Larger |
| Sequence Learning | No | Yes |
| Hyperparameter Tuning | Manual | Automated Grid Search |
| Memory Usage | Lower | Higher |

## Reinforcement Learning (neuroevolutionary pipeline)

- https://github.com/A-MoizQ/Street-Fighter-Neuroevolution