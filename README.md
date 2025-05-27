# Street Fighter II Turbo AI Bot

This project implements multiple deep learning and reinforcement learning-based AI systems that can play Street Fighter II Turbo. The system records human gameplay, trains various types of neural networks on these recordings, and then uses the trained models to control characters in the game.

## Project Overview

The project consists of several key components:

1. **Data Collection** - Record human gameplay to create training datasets
2. **Data Processing** - Normalize and transform gameplay data into suitable format for machine learning
3. **Model Training** - Train various types of neural networks and RL agents
4. **Bot Execution** - Use trained models to play the game automatically

## AI Model Architectures

The system supports five different AI approaches:

### 1. **Standard ANN** ✅ (Working)
- Feed-forward neural network with dense layers
- Uses flattened 6-frame windows as input
- Simple and fast, good baseline performance

### 2. **RNN/LSTM** ✅ (Working) 
- Recurrent neural network that processes temporal sequences
- Better captures movement patterns and action sequences
- Uses 6-frame windows as temporal sequences (not flattened)

### 3. **DQN (Deep Q-Network)** ⚠️ (Computational Issues)
- Reinforcement learning approach using Q-learning
- Learns through trial and error during live gameplay
- GPU-accelerated training with experience replay

### 4. **Hybrid RNN-DQN** ⚠️ (Computational Issues)
- Combines pre-trained RNN knowledge with DQN fine-tuning
- Uses RNN as baseline, then improves through reinforcement learning
- Faster convergence than pure DQN

### 5. **Neuroevolution** ⚠️ (Computational Issues)
- Genetic algorithm that evolves neural network weights
- No gradient-based training, uses evolutionary selection
- State-change based action system for FPS independence

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
- **For RL methods**: NVIDIA GPU with CUDA support (recommended)

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

### Running AI Bots

#### 1. Standard ANN Bot ✅
1. Follow steps 1-5 from the Recording Gameplay section
2. **Open a command prompt in the `PythonAPI` directory** and run:
   ```
   python controller.py "1"
   ```
3. Select your character and click the Gyroscope Bot icon

#### 2. RNN Bot ✅
To use the RNN-based models:
```
python controller.py "1" "rnn"
```

#### 3. DQN Bot ⚠️
To use the Deep Q-Network bot (if trained):
```
python controller.py "1" "dqn"
```

#### 4. Hybrid RNN-DQN Bot ⚠️
To use the hybrid bot (if trained):
```
python controller.py "1" "hybrid"
```

#### 5. Neuroevolution Bot ⚠️
To use the evolved neural network bot (if evolved):
```
python controller.py "1" "neuroevolution"
```

For two-player mode with player 2 controlled by AI, use:
```
python controller.py "2" [model_type]
```

## Training Different AI Models

### 1. Standard ANN Training ✅

**Requirements**: Recorded gameplay data

**Process**:
1. Ensure you have recorded gameplay data for the characters
2. Process the normalized datasets into windowed datasets
3. Edit `train_models/train_individual_character.py` to specify character IDs
4. Run training:
   ```
   python train_models/train_individual_character.py
   ```

**Output**: Models saved to `models/` directory

### 2. RNN Training ✅

**Requirements**: Flattened window datasets

**Process**:
1. Ensure you have the processed datasets ready
2. Edit `train_models/train_individual_character_rnn.py` for character IDs
3. Run RNN training with hyperparameter optimization:
   ```
   python train_models/train_individual_character_rnn.py
   ```

**Features**:
- Automated grid search for optimal hyperparameters
- LSTM layers for temporal sequence learning
- Better action sequence prediction

**Output**: Models saved to `RNN_models/` directory

### 3. DQN Training ⚠️

**Requirements**: 
- NVIDIA GPU with CUDA (recommended)
- Live gameplay environment
- Significant computational resources

**Process**:
1. Setup GPU acceleration (if available)
2. Start DQN training mode:
   ```
   python controller.py "1" "train_dqn"
   ```
3. Let the agent play and learn for many episodes (1000+)

**How it Works**:
- **Experience Replay**: Stores game experiences in memory buffer
- **Q-Learning**: Learns optimal action values through trial and error
- **Epsilon-Greedy**: Balances exploration vs exploitation
- **Target Network**: Stabilizes training with periodic weight updates
- **GPU Acceleration**: Uses mixed precision and larger batch sizes

**Training Features**:
- Live learning during gameplay
- Contextual reward system based on combat effectiveness
- Adaptive action space (35+ meaningful button combinations)
- Health-based, distance-based, and strategic rewards

**Output**: Models saved to `rl_models/dqn_model_[character_id].keras`

**⚠️ Known Issues**:
- Requires extensive training time (hours to days)
- Sensitive to FPS variations in the game
- High computational overhead
- May not converge properly due to variable game timing

### 4. Hybrid RNN-DQN Training ⚠️

**Requirements**:
- Pre-trained RNN model for the character
- GPU acceleration (highly recommended)
- Stable game environment

**Process**:
1. Ensure RNN model exists for the character
2. Start hybrid training:
   ```
   python controller.py "1" "train_hybrid"
   ```
3. Training will combine RNN knowledge with RL fine-tuning

**How it Works**:
- **Weight Transfer**: Copies RNN weights to Q-network (except output layer)
- **Frozen Layers**: Preserves RNN knowledge in early layers
- **Fine-tuning**: Only trains final layers with Q-learning
- **RNN Guidance**: Uses RNN predictions to guide exploration
- **Faster Convergence**: Requires fewer episodes than pure DQN

**Training Features**:
- Starts with human-like behavior from RNN
- Conservative epsilon decay (maintains RNN knowledge)
- Action diversity tracking and bonuses
- Aggressive action bias during exploration
- Shorter episodes (30 seconds) for faster iteration

**Output**: Models saved to `rl_models/hybrid_rnn_dqn_model_[character_id].keras`

**⚠️ Known Issues**:
- RNN-to-action-space mapping complexity
- Still sensitive to FPS variations
- Requires both supervised (RNN) and RL training pipelines
- Architecture compatibility challenges between RNN and DQN

### 5. Neuroevolution Training ⚠️

**Requirements**:
- Pre-trained RNN model as base population
- Stable, long-running game environment
- Patience (evolution takes many generations)

**Process**:
1. Ensure RNN base model exists
2. Start evolution training:
   ```
   python controller.py "1" "train_neuroevolution"
   ```
3. Let evolution run for 30+ generations (each generation = 15 individuals × 3 minutes)

**How it Works**:
- **Genetic Algorithm**: Evolves neural network weights without gradients
- **Population**: 15 individuals per generation
- **Base Weights**: Starts from pre-trained RNN weights
- **Mutation**: Gaussian noise applied to subset of weights
- **Crossover**: Combines successful individuals
- **Tournament Selection**: Best performers become parents
- **Elitism**: Always keeps best individual

**Fitness Evaluation**:
- **Win/Loss Outcomes** (50%): Primary performance metric
- **Damage Efficiency** (25%): Damage dealt vs received
- **Action Diversity** (15%): Prevents button mashing
- **Combat Engagement** (10%): Optimal fighting distance
- **Longevity Bonus**: Surviving full evaluation period

**Advanced Features**:
- **State-Change Detection**: Only acts when game state changes significantly
- **Global Best Tracking**: Saves best individual across all generations
- **FPS Independence**: Time-based evaluation instead of frame-based
- **Move State Filtering**: Prevents attacks during move animations

**Output**: Best evolved weights saved to `neuroevolution_models/global_best_char_[character_id].pkl`

**⚠️ Known Issues**:
- Extremely time-consuming (hours per character)
- Sensitive to FPS variations despite improvements
- Complex fitness function tuning required
- May not find significantly better solutions than RNN baseline
- Round transition timing issues

## Project Structure

- **`PythonAPI/`** - Main code directory
  - **Working Bots**:
    - `bot.py` - Standard ANN implementation
    - `rnn_bot.py` - RNN/LSTM implementation
  - **Advanced Bots** (⚠️ Computational Issues):
    - `dqn_bot.py` - Deep Q-Network bot
    - `hybrid_bot.py` - Hybrid RNN-DQN bot  
    - `neuroevolution_bot.py` - Evolved neural network bot
  - **Training Systems**:
    - `dqn_trainer.py` - DQN training system
    - `rnn_dqn_trainer.py` - Hybrid RNN-DQN trainer
    - `neuroevolution_trainer.py` - Genetic algorithm trainer
    - `rl_environment.py` - Reinforcement learning environment wrapper
  - **Core Systems**:
    - `controller.py` - Main interface (supports all bot types)
    - `game_state.py` - Game state representation
    - `player.py` - Player state representation
    - `buttons.py` - Button state representation
    - `command.py` - Command objects to send to game
  - **Utilities**:
    - `listen_to_key.py` - Keyboard input detection
    - `make_dataset.py` - Dataset creation utilities

- **Data Directories**:
  - `normalized_character_datasets/` - Raw recorded gameplay
  - `flattened_window_datasets/` - Processed training data
  - `models/` - Trained standard ANN models
  - `RNN_models/` - Trained RNN/LSTM models
  - `rl_models/` - Trained DQN and Hybrid models
  - `neuroevolution_models/` - Evolved neural network weights

- **Training Scripts**:
  - `train_models/train_individual_character.py` - Standard ANN training
  - `train_models/train_individual_character_rnn.py` - RNN training
  - `train_models/train_dqn.py` - DQN architecture and training utilities

- **Game Files**:
  - `single-player/` - Single-player game setup
  - `two-players/` - Two-player game setup

## Comparison of AI Methods

| Method | Status | Training Time | Performance | Computational Cost | FPS Sensitivity |
|--------|--------|---------------|-------------|-------------------|-----------------|
| **Standard ANN** | ✅ Working | ~30 minutes | Good | Low | Low |
| **RNN/LSTM** | ✅ Working | ~1-2 hours | Very Good | Medium | Low |
| **DQN** | ⚠️ Issues | ~6-12 hours | Variable | Very High | High |
| **Hybrid RNN-DQN** | ⚠️ Issues | ~3-6 hours | Variable | High | High |
| **Neuroevolution** | ⚠️ Issues | ~8-20 hours | Variable | Very High | Medium |

### Detailed Method Analysis

#### ✅ **Working Methods (Recommended)**

**Standard ANN**:
- ✅ Fast training and execution
- ✅ Reliable and stable
- ✅ Good for beginners
- ❌ Limited temporal understanding

**RNN/LSTM**:
- ✅ Excellent temporal sequence learning
- ✅ Better action prediction
- ✅ Automated hyperparameter tuning
- ✅ Most balanced approach
- ❌ Larger model size

#### ⚠️ **Advanced Methods (Experimental)**

**DQN**:
- ✅ Learns optimal strategies through experience
- ✅ GPU acceleration support
- ✅ Comprehensive reward system
- ❌ Requires extensive computational resources
- ❌ Sensitive to FPS variations
- ❌ Training instability

**Hybrid RNN-DQN**:
- ✅ Combines supervised and reinforcement learning
- ✅ Faster convergence than pure DQN
- ✅ Starts with human-like behavior
- ❌ Complex architecture requirements
- ❌ Still faces DQN computational issues
- ❌ Difficult action space mapping

**Neuroevolution**:
- ✅ No gradient-based training required
- ✅ Can discover novel strategies
- ✅ Global optimization approach
- ✅ State-change based action system
- ❌ Extremely time-consuming
- ❌ No guarantee of improvement over RNN
- ❌ Complex fitness function design

## Known Issues and Limitations

### Variable FPS Problem
The most significant challenge affecting advanced methods is the variable FPS of the Street Fighter II Turbo emulator:

- **FPS Variability**: Game runs at 1-7 FPS instead of consistent 60 FPS
- **Timing Dependencies**: RL methods rely on consistent timing for proper learning
- **Action Frequency**: Bots may spam actions or become unresponsive
- **Round Transitions**: Inconsistent timing between rounds disrupts training

### Computational Overhead
Advanced methods require significant computational resources:

- **Memory Usage**: Large experience replay buffers (50K-100K experiences)
- **GPU Requirements**: CUDA-capable GPU strongly recommended
- **Training Time**: Hours to days of continuous training
- **Evaluation Time**: 3+ minutes per individual for neuroevolution

### Solutions Attempted
- ✅ Time-based evaluation instead of frame-based
- ✅ State-change detection for action timing
- ✅ FPS-adaptive action dampening
- ✅ Round transition management
- ❌ Still insufficient for stable RL training

## Recommendations

### For Learning and Development
- **Start with Standard ANN**: Simple, fast, and reliable
- **Upgrade to RNN**: Best balance of performance and complexity
- **Experiment with Advanced Methods**: Educational value, but expect challenges

### For Production Use
- **Use RNN/LSTM models**: Most reliable advanced method
- **Avoid RL methods**: Until FPS stability is resolved
- **Focus on data quality**: Better human gameplay data improves all methods

### For Research
- **Investigate FPS stabilization**: Core blocker for RL methods
- **Explore transformer architectures**: Potential next-generation approach
- **Develop better reward functions**: Critical for RL success

## Troubleshooting

### General Issues
- **Game not responding**: Ensure proper connection and model exists
- **Character not moving**: Check model training for that character
- **Connection errors**: Verify correct port (9999 for P1, 10000 for P2)

### Advanced Method Issues
- **DQN not learning**: Check GPU availability and FPS stability
- **Hybrid bot errors**: Ensure RNN model exists first
- **Neuroevolution timeout**: Verify sufficient evaluation time
- **Memory errors**: Reduce batch sizes or use CPU-only mode

### Performance Issues
- **Slow inference**: Use GPU acceleration when available
- **High memory usage**: Reduce replay buffer size or population size
- **Training instability**: Monitor FPS consistency and adjust timing parameters

## Future Work

### Short-term Improvements
- FPS stabilization techniques
- Better state representation for RL
- Improved reward function design
- Memory optimization for long training runs

### Long-term Research Directions
- Transformer-based architectures for fighting games
- Model-based reinforcement learning approaches
- Multi-agent training environments
- Real-time strategy adaptation systems

---

**Note**: While the advanced AI methods (DQN, Hybrid, Neuroevolution) demonstrate cutting-edge techniques in game AI, they currently face significant computational and timing challenges. The RNN/LSTM approach provides the best balance of performance, reliability, and advanced features for practical use.