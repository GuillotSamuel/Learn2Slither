# 🐍 Learn2Slither

*A Snake game with artificial intelligence based on reinforcement learning*

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Technical Solution](#-technical-solution)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Game Modes](#-game-modes)
- [Project Structure](#-project-structure)
- [Results and Performance](#-results-and-performance)
- [Bonus Features](#-bonus-features)
- [Contributors](#-contributors)

## 🎯 About the Project

Learn2Slither is an advanced implementation of the classic Snake game using reinforcement learning with Q-Learning and Deep Q-Networks. The project enables training an AI capable of playing Snake autonomously and achieving impressive performance.

The agent learns to navigate the environment, avoid obstacles, collect food, and maximize its length through a sophisticated reward system and an optimized neural network architecture.

## 🔬 Technical Solution

### Neural Network Architecture

- **Model**: Multi-Layer Perceptron (MLP) with 4 fully-connected layers
- **Input**: 16-dimensional vector representing the game state
- **Output**: 3 possible actions (turn left, go straight, turn right)
- **Architecture**:
  - Input layer: 16 → 256 neurons
  - Hidden layer 1: 256 → 128 neurons
  - Hidden layer 2: 128 → 64 neurons
  - Output layer: 64 → 3 neurons
  - Activation: ReLU
  - Dropout: 10% to prevent overfitting

### State Representation

The game state is encoded in the following dimensions:
- **Raycast vision**: 4 cardinal directions
- **Distances to obstacles**: walls and snake body
- **Relative food position**
- **Current snake direction**

### Learning Algorithm

- **Q-Learning** with deep neural networks
- **Exploration vs Exploitation**: ε-greedy strategy with adaptive decay
- **Optimizer**: Adam with learning rate of 0.001
- **Loss function**: Mean Squared Error (MSE)
- **Discount factor**: γ = 0.99

### Reward System

- **Positive reward**: +10 for collecting an apple
- **Negative reward**: -10 for collision (death)
- **Survival reward**: +1 for each step without collision
- **"Ultra Rewards" mode**: enriched reward system based on food proximity

## ✨ Features

- 🎮 **4 game modes**: training, evaluation, automatic play, manual play
- 🎨 **2 render modes**: basic (console) and classic (graphics)
- 📊 **Real-time metrics**: score, length, performance ratio
- 📈 **Performance visualization**: saved training graphs
- 🎯 **Variable board sizes**: from 5x5 to 20x20, or random
- 💾 **Automatic saving**: models saved at different stages
- 🚀 **Evaluation mode**: performance testing over multiple episodes

## 🛠 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Step-by-step Installation

1. **Clone the project**
   ```bash
   git clone https://github.com/your-username/Learn2Slither.git
   cd Learn2Slither
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install --no-cache-dir -r requirements.txt
   pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

### Main Dependencies

- `torch`: Deep learning framework
- `pygame`: Graphics rendering engine
- `numpy`: Numerical computations
- `matplotlib`: Data visualization
- `pyyaml`: Configuration
- `flake8`: Code linting

## 🚀 Usage

### Basic Command

```bash
python src/main.py --mode [MODE] [OPTIONS]
```

### Usage with Makefile

The project includes a Makefile to simplify common commands:

```bash
# Train the agent
make train

# Train the agent on randomly sized maps (5 to 20 squares)
make train_random_map

# Evaluate performance
make evaluate

# Evaluate persormance on randomly sized maps (5 to 20 squares)
make evaluate_random_map

# Play with trained AI
make play

# Play with trained AI on randomly sized maps (5 to 20 squares)
make play_random_map

# Play manually
make manual

# See all available commands
make help
```

## 🎮 Game Modes

### 1. Training Mode
```bash
python src/main.py --mode train --episodes 10000 --display --board_size 10
```
- Trains the agent over a specified number of episodes
- Automatic model saving at different stages
- Performance graph generation

### 2. Evaluation Mode
```bash
python src/main.py --mode evaluate --episodes 100 --model models/end_model.pkl
```
- Tests the performance of a trained model
- Calculates detailed statistics
- Silent mode or with display

### 3. Automatic Play Mode
```bash
python src/main.py --mode play --model models/end_model.pkl --render_mode classic
```
- AI plays automatically
- Real-time graphical display
- Adjustable speed parameters

### 4. Manual Mode
```bash
python src/main.py --mode manual --render_mode classic
```
- Human control with keyboard arrows
- Complete graphical interface
- Perfect for testing and comparing with AI

### Advanced Options

| Option | Description | Possible Values |
|--------|-------------|-----------------|
| `--episodes` | Number of episodes | Integer ≥ 100 |
| `--board_size` | Board size | 10-30, or 0 for random |
| `--display_speed` | Display speed | Float (0.1 recommended) |
| `--ultra_rewards` | Enhanced rewards | true/false |
| `--render_mode` | Render mode | basic/classic |
| `--model` | Model path | Path to .pkl file |

## 📁 Project Structure

```
Learn2Slither/
├── src/                          # Main source code
│   ├── main.py                   # Program entry point
│   ├── trainer.py                # Training and evaluation logic
│   ├── mlp.py                    # Neural network architecture
│   ├── board.py                  # Game board logic
│   ├── state.py                  # Game state representation
│   ├── utils.py                  # Utility functions
│   ├── manual.py                 # Manual game mode
│   └── assets/                   # Graphical resources
│       └── classic/              # Sprites for classic mode
├── models/                       # Training models
│   ├── 1_model.pkl              # Model after 1 episode
│   ├── 10_model.pkl             # Model after 10 episodes
│   ├── 100_model.pkl            # Model after 100 episodes
│   ├── decile_model.pkl         # Model at 10% of training
│   └── mid_model.pkl            # Model at mid-training
├── models_eval/                  # Final evaluation models
├── graphs_eval/                  # Performance graphs
├── requirements.txt              # Python dependencies
├── Makefile                      # Automated commands
└── README.md                     # Documentation
```

## 📊 Results and Performance

### Performance Metrics

- **Maximum length achieved**: 57 (absolute record)
- **Average length**: ~35 after complete training
- **Survival rate**: >85% on long games
- **Convergence**: ~5000-8000 episodes for a stable model

### Generated Graphs

The system automatically generates performance graphs:
- Average score evolution
- Snake length progression
- Exploration vs exploitation rate
- Neural network loss curves

## 🏆 Bonus Features

### Advanced Features Implemented

- ✅ **Performance graphs**: Complete visualization of training metrics
- ✅ **Length record**: Exceeded the goal of 35 with a record of 59
- ✅ **Random boards**: Training on different board sizes
- ✅ **Multiple render modes**: Basic interface and advanced graphics
- ✅ **Adaptive reward system**: "Ultra rewards" mode for optimized learning
- ✅ **Smart saving**: Models saved at different stages
- ✅ **Complete command-line interface**: Advanced parameterization via arguments
- ✅ **Integrated Makefile**: Automation of common tasks

### Technical Improvements

- Optimized MLP architecture with dropout
- Adaptive ε-greedy exploration strategy
- Input state normalization

## 👨‍💻 Contributors

Developed by Samuel Guillot.

*For any questions or suggestions, feel free to open an issue or contribute to the project!*

## 📜 License

This project is licensed under the **MIT License**.
