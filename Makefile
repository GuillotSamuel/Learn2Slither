# ============================================================================#
# Makefile for Snake Q-Learning Project                                       #
# ============================================================================#
# This Makefile provides convenient commands for training, evaluating, and    #
# playing the Snake game using different reinforcement learning algorithms    #
# including Q-Learning and Deep Q-Learning with various map configurations.   #
# ============================================================================#

# Default target
.DEFAULT_GOAL := help

# Make configuration
.PHONY: help clean install deps check train train_random_map train_multithreaded train_random_map_multithreaded train_deep_q_learning train_deep_q_learning_random_map evaluate evaluate_all evaluate_random_map evaluate_multithreaded evaluate_random_map_multithreaded evaluate_deep_q_learning evaluate_random_map_deep_q_learning play play_random_map play_multithreaded play_random_map_multithreaded play_deep_q_learning play_random_map_deep_q_learning manual

# Python interpreter and main script configuration
PYTHON ?= $(shell if [ -d venv ]; then echo "venv/bin/python"; else echo "python3"; fi)
MAIN = src/main.py
PIP ?= $(shell if [ -d venv ]; then echo "venv/bin/pip"; else echo "pip3"; fi)

# Model folder paths for storing trained models
MODEL_FOLDER=models
MODEL_Q_LEARNING_FOLDER=$(MODEL_FOLDER)/models_eval_q_learning
MODEL_Q_LEARNING_RANDOM_MAP_FOLDER=$(MODEL_FOLDER)/models_random_map_q_learning
MODEL_Q_LEARNING_FOLDER_MULTITHREADED=$(MODEL_FOLDER)/models_eval_q_learning_multithreaded
MODEL_Q_LEARNING_RANDOM_MAP_FOLDER_MULTITHREADED=$(MODEL_FOLDER)/models_random_map_q_learning_multithreaded
MODEL_DEEP_LEARNING_FOLDER=$(MODEL_FOLDER)/models_eval_deep_q
MODEL_DEEP_LEARNING_RANDOM_MAP_FOLDER=$(MODEL_FOLDER)/models_random_map_deep_q

# Model file paths for different training configurations
MODEL_Q_LEARNING=$(MODEL_Q_LEARNING_FOLDER)/end_model.pkl              		# Standard Q-Learning model
MODEL_Q_LEARNING_RANDOM_MAP=$(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER)/end_model.pkl  # Q-Learning with random maps
MODEL_Q_LEARNING_MULTITHREADED=$(MODEL_Q_LEARNING_FOLDER_MULTITHREADED)/end_model.pkl              		# Standard Q-Learning model multithreaded
MODEL_Q_LEARNING_RANDOM_MAP_MULTITHREADED=$(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER_MULTITHREADED)/end_model.pkl  # Q-Learning with random maps multithreaded
MODEL_DEEP_LEARNING=$(MODEL_DEEP_LEARNING_FOLDER)/end_model.pkl               		# Deep Q-Learning model
MODEL_DEEP_LEARNING_RANDOM_MAP=$(MODEL_DEEP_LEARNING_RANDOM_MAP_FOLDER)/end_model.pkl   # Deep Q-Learning with random maps

# Game configuration parameters
BOARD_SIZE ?= 10        # Default board size (10x10 grid)
DISPLAY_SPEED ?= 0.07   # Animation speed for visual gameplay (seconds between frames)
EPISODES_TRAIN ?= 1000000  # Default episodes for training
EPISODES_EVAL ?= 1000      # Default episodes for evaluation

# ============================================================================#
# UTILITY TARGETS                                                             #
# ============================================================================#

# Install Python dependencies
install: setup_venv deps check

# Create venv and activate it
setup_venv:
	@echo "Checking if virtual environment exists..."
	@if [ -d venv ]; then \
		echo "Virtual environment already exists. Skipping setup."; \
	else \
		echo "Virtual environment not found. Creating a new one."; \
		echo "Setting up Python virtual environment..."; \
		python3 -m venv venv; \
		echo "Virtual environment created successfully."; \
		echo "Virtual environment setup complete. Use 'source venv/bin/activate' to activate it."; \
	fi

# Activate the project's virtual environment (deactivates any current venv first)
activate_venv:
	@echo "Switching to project virtual environment..."
	@if [ -d venv ]; then \
		echo "Virtual environment found. To activate it, run:"; \
		echo "source venv/bin/activate"; \
		echo ""; \
		echo "Or run this command directly:"; \
		echo "deactivate 2>/dev/null || true && source venv/bin/activate"; \
	else \
		echo "Virtual environment not found. Creating it first..."; \
		$(MAKE) setup_venv; \
		echo "Now run: source venv/bin/activate"; \
	fi

# Install dependencies from requirements.txt
deps:
	@echo "Installing Python dependencies..."
	@if [ -d venv ]; then \
		echo "Using virtual environment..."; \
		venv/bin/pip install -r requirements.txt; \
		venv/bin/pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu; \
	else \
		echo "No virtual environment found, using system pip..."; \
		$(PIP) install -r requirements.txt; \
		pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu; \
	fi

# Check if Python and dependencies are available
check:
	@echo "Checking Python environment..."
	@$(PYTHON) --version
	@echo "Checking if main script exists..."
	@test -f $(MAIN) || (echo "Error: $(MAIN) not found" && exit 1)
	@echo "Environment check passed!"

# Clean generated files and cached data
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pkl~" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Cleanup completed!"

# ============================================================================#
# TRAINING TARGETS                                                            #
# ============================================================================#

# Train Q-Learning agent on fixed-size board with enhanced rewards
train:
	@echo "Starting Q-Learning training ($(EPISODES_TRAIN) episodes)..."
	$(PYTHON) $(MAIN) --mode train --episodes $(EPISODES_TRAIN) --board_size $(BOARD_SIZE) --ultra_rewards --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_FOLDER) --episode_logs

# Train Q-Learning agent on random board sizes (5-20) for better generalization
train_random_map:
	@echo "Starting Q-Learning training on random maps ($(EPISODES_TRAIN) episodes)..."
	$(PYTHON) $(MAIN) --mode train --episodes $(EPISODES_TRAIN) --board_size 0 --ultra_rewards --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER) --episode_logs

# Train Q-Learning agent on fixed-size board with enhanced rewards with multithreading (high CPU utilization)
train_multithreaded:
	@echo "Starting multithreaded Q-Learning training ($(EPISODES_TRAIN) episodes)..."
	$(PYTHON) $(MAIN) --mode train --episodes $(EPISODES_TRAIN) --board_size $(BOARD_SIZE) --ultra_rewards --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_FOLDER_MULTITHREADED) --episode_logs

# Train Q-Learning agent with maximum CPU utilization (fewer logs, more threads)
train_multithreaded_performance:
	@echo "Starting high-performance multithreaded Q-Learning training ($(EPISODES_TRAIN) episodes)..."
	@echo "This mode maximizes CPU utilization with minimal synchronization overhead"
	$(PYTHON) $(MAIN) --mode train --episodes $(EPISODES_TRAIN) --board_size $(BOARD_SIZE) --ultra_rewards --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_FOLDER_MULTITHREADED) --no_episode_logs

# Quick CPU performance test with optimized settings
test_cpu_performance:
	@echo "Quick CPU performance test (5000 episodes, optimized for maximum CPU usage)..."
	@echo "Monitor CPU usage with: htop, top, or watch 'ps -eLf | grep python | wc -l'"
	$(PYTHON) $(MAIN) --mode train --episodes 5000 --board_size 8 --ultra_rewards --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_FOLDER_MULTITHREADED) --no_episode_logs

# Train Q-Learning agent on random board sizes (5-20) for better generalization with multithreading
train_random_map_multithreaded:
	@echo "Starting multithreaded Q-Learning training on random maps ($(EPISODES_TRAIN) episodes)..."
	$(PYTHON) $(MAIN) --mode train --episodes $(EPISODES_TRAIN) --board_size 0 --ultra_rewards --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER_MULTITHREADED) --episode_logs

# Train Deep Q-Learning agent on fixed-size board using neural network
train_deep_q_learning:
	@echo "Starting Deep Q-Learning training (10000 episodes)..."
	$(PYTHON) $(MAIN) --mode train --episodes 10000 --board_size $(BOARD_SIZE) --ultra_rewards --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_FOLDER) --episode_logs

# Train Deep Q-Learning agent on random board sizes for improved adaptability
train_deep_q_learning_random_map:
	@echo "Starting Deep Q-Learning training on random maps (10000 episodes)..."
	$(PYTHON) $(MAIN) --mode train --episodes 10000 --board_size 0 --ultra_rewards --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_RANDOM_MAP_FOLDER) --episode_logs

# Train all models sequentially
train_all: train train_multithreaded train_deep_q_learning
	@echo "All training completed!"

# ============================================================================#
# EVALUATION TARGETS                                                          #
# ============================================================================#

# Evaluate all three models in 10x10 maps
evaluate_all: check
	@echo "Evaluating all models ($(EPISODES_EVAL) episodes each)..."
	$(PYTHON) $(MAIN) --mode evaluate --episodes $(EPISODES_EVAL) --model $(MODEL_Q_LEARNING) --board_size $(BOARD_SIZE) --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_FOLDER) --no_episode_logs
	$(PYTHON) $(MAIN) --mode evaluate --episodes $(EPISODES_EVAL) --model $(MODEL_Q_LEARNING_MULTITHREADED) --board_size $(BOARD_SIZE) --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_FOLDER_MULTITHREADED) --no_episode_logs
	$(PYTHON) $(MAIN) --mode evaluate --episodes $(EPISODES_EVAL) --model $(MODEL_DEEP_LEARNING) --board_size $(BOARD_SIZE) --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_FOLDER) --no_episode_logs

# Evaluate Q-Learning agent performance on fixed-size board (1000 episodes)
evaluate: check
	@echo "Evaluating Q-Learning agent ($(EPISODES_EVAL) episodes)..."
	$(PYTHON) $(MAIN) --mode evaluate --episodes $(EPISODES_EVAL) --model $(MODEL_Q_LEARNING) --board_size $(BOARD_SIZE) --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_FOLDER)

# Evaluate Q-Learning agent performance on random board sizes
evaluate_random_map: check
	@echo "Evaluating Q-Learning agent on random maps ($(EPISODES_EVAL) episodes)..."
	$(PYTHON) $(MAIN) --mode evaluate --episodes $(EPISODES_EVAL) --model $(MODEL_Q_LEARNING_RANDOM_MAP) --board_size 0 --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER)

# Evaluate Q-Learning multithreaded agent performance on fixed-size board (1000 episodes)
evaluate_multithreaded: check
	@echo "Evaluating multithreaded Q-Learning agent ($(EPISODES_EVAL) episodes)..."
	$(PYTHON) $(MAIN) --mode evaluate --episodes $(EPISODES_EVAL) --model $(MODEL_Q_LEARNING_MULTITHREADED) --board_size $(BOARD_SIZE) --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_FOLDER_MULTITHREADED)

# Evaluate Q-Learning multithreaded agent performance on random board sizes
evaluate_random_map_multithreaded: check
	@echo "Evaluating multithreaded Q-Learning agent on random maps ($(EPISODES_EVAL) episodes)..."
	$(PYTHON) $(MAIN) --mode evaluate --episodes $(EPISODES_EVAL) --model $(MODEL_Q_LEARNING_RANDOM_MAP_MULTITHREADED) --board_size 0 --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER_MULTITHREADED)

# Evaluate Deep Q-Learning agent performance (1000 episodes for thorough testing)
evaluate_deep_q_learning: check
	@echo "Evaluating Deep Q-Learning agent ($(EPISODES_EVAL) episodes)..."
	$(PYTHON) $(MAIN) --mode evaluate --episodes $(EPISODES_EVAL) --model $(MODEL_DEEP_LEARNING) --board_size $(BOARD_SIZE) --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_FOLDER)

# Evaluate Deep Q-Learning agent on random board sizes (100 episodes)
evaluate_random_map_deep_q_learning: check
	@echo "Evaluating Deep Q-Learning agent on random maps (100 episodes)..."
	$(PYTHON) $(MAIN) --mode evaluate --episodes 100 --model $(MODEL_DEEP_LEARNING_RANDOM_MAP) --board_size 0 --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_RANDOM_MAP_FOLDER)

# ============================================================================#
# GAMEPLAY TARGETS                                                            #
# ============================================================================#

# Watch the Q-Learning agent play with visual rendering on fixed-size board
play: check
	@echo "Starting Q-Learning agent gameplay..."
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_Q_LEARNING) --board_size $(BOARD_SIZE) --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_FOLDER)

# Watch the Q-Learning agent play on random board sizes (5-20)
play_random_map: check
	@echo "Starting Q-Learning agent gameplay on random maps..."
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_Q_LEARNING_RANDOM_MAP) --board_size 0 --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER)

# Watch the Q-Learning multithreaded agent play with visual rendering on fixed-size board
play_multithreaded: check
	@echo "Starting multithreaded Q-Learning agent gameplay..."
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_Q_LEARNING_MULTITHREADED) --board_size $(BOARD_SIZE) --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_FOLDER_MULTITHREADED)

# Watch the Q-Learning multithreaded agent play on random board sizes (5-20)
play_random_map_multithreaded: check
	@echo "Starting multithreaded Q-Learning agent gameplay on random maps..."
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_Q_LEARNING_RANDOM_MAP_MULTITHREADED) --board_size 0 --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER_MULTITHREADED)

# Watch the Deep Q-Learning agent play with visual rendering
play_deep_q_learning: check
	@echo "Starting Deep Q-Learning agent gameplay..."
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_DEEP_LEARNING) --board_size $(BOARD_SIZE) --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_FOLDER)

# Watch the Deep Q-Learning agent play on random board sizes
play_random_map_deep_q_learning: check
	@echo "Starting Deep Q-Learning agent gameplay on random maps..."
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_DEEP_LEARNING_RANDOM_MAP) --board_size 0 --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_RANDOM_MAP_FOLDER)

# ============================================================================#
# INTERACTIVE TARGETS                                                         #
# ============================================================================#

# Play Snake manually using keyboard controls (10x10 board, basic rendering)
manual: check
	@echo "Starting manual Snake game..."
	$(PYTHON) $(MAIN) --mode manual --render_mode basic --board_size 10

# ============================================================================#
# DEVELOPMENT TARGETS                                                         #
# ============================================================================#

# Check code style with flake8
lint:
	@echo "Checking code style with flake8..."
	flake8 src/ --max-line-length=120 --ignore=E501,W503

# Run quick test to ensure the main script works
test: check
	@echo "Running quick functionality test..."
	$(PYTHON) $(MAIN) --mode manual --board_size 5 --episodes 1 || echo "Test completed (manual mode)"

# Show project information
info:
	@echo "============================================================================"
	@echo "Snake Q-Learning Project Information"
	@echo "============================================================================"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Main script: $(MAIN)"
	@echo "Board size: $(BOARD_SIZE)"
	@echo "Display speed: $(DISPLAY_SPEED)"
	@echo "Training episodes: $(EPISODES_TRAIN)"
	@echo "Evaluation episodes: $(EPISODES_EVAL)"
	@echo ""
	@echo "Available models:"
	@test -f $(MODEL_Q_LEARNING) && echo "  ✓ Q-Learning model" || echo "  ✗ Q-Learning model (not trained)"
	@test -f $(MODEL_Q_LEARNING_MULTITHREADED) && echo "  ✓ Multithreaded Q-Learning model" || echo "  ✗ Multithreaded Q-Learning model (not trained)"
	@test -f $(MODEL_DEEP_LEARNING) && echo "  ✓ Deep Q-Learning model" || echo "  ✗ Deep Q-Learning model (not trained)"
	@echo "============================================================================"

# ============================================================================#
# HELP TARGET                                                                 #
# ============================================================================#

# Display usage information and available commands
help:
	@echo "============================================================================"
	@echo "Snake Q-Learning Project - Available Commands"
	@echo "============================================================================"
	@echo ""
	@echo "SETUP COMMANDS:"
	@echo "  make install                             - Install Python dependencies"
	@echo "  make deps                                - Install dependencies from requirements.txt"
	@echo "  make check                               - Verify Python environment and files"
	@echo "  make clean                               - Clean temporary files and cache"
	@echo "  make info                                - Show project information and model status"
	@echo ""
	@echo "TRAINING COMMANDS:"
	@echo "  make train                               - Train Q-Learning agent (fixed board)"
	@echo "  make train_random_map                    - Train Q-Learning agent (random board sizes)"
	@echo "  make train_multithreaded                 - Train Q-Learning agent (fixed board, multithreaded)"
	@echo "  make train_multithreaded_performance     - Train Q-Learning agent (maximum CPU utilization)"  
	@echo "  make train_random_map_multithreaded      - Train Q-Learning agent (random boards, multithreaded)"
	@echo "  make train_deep_q_learning               - Train Deep Q-Learning agent (neural network)"
	@echo "  make train_deep_q_learning_random_map    - Train Deep Q-Learning (random boards)"
	@echo "  make train_all                           - Train all main models sequentially"
	@echo ""
	@echo "EVALUATION COMMANDS:"
	@echo "  make evaluate                            - Evaluate Q-Learning agent ($(EPISODES_EVAL) episodes)"
	@echo "  make evaluate_random_map                 - Evaluate Q-Learning agent (random boards)"
	@echo "  make evaluate_multithreaded              - Evaluate Q-Learning agent (multithreaded)"
	@echo "  make evaluate_random_map_multithreaded   - Evaluate Q-Learning agent (random boards, multithreaded)"
	@echo "  make evaluate_deep_q_learning            - Evaluate Deep Q-Learning ($(EPISODES_EVAL) episodes)"
	@echo "  make evaluate_random_map_deep_q_learning - Evaluate Deep Q-Learning (random boards)"
	@echo "  make evaluate_all                        - Evaluate all models on fixed board"
	@echo ""
	@echo "GAMEPLAY COMMANDS:"
	@echo "  make play                                - Watch Q-Learning agent play ($(BOARD_SIZE)x$(BOARD_SIZE) board)"
	@echo "  make play_random_map                     - Watch Q-Learning agent (random boards 5-20)"
	@echo "  make play_multithreaded                  - Watch Q-Learning agent play (multithreaded)"
	@echo "  make play_random_map_multithreaded       - Watch Q-Learning agent (random boards, multithreaded)"
	@echo "  make play_deep_q_learning                - Watch Deep Q-Learning agent play"
	@echo "  make play_random_map_deep_q_learning     - Watch Deep Q-Learning (random boards)"
	@echo ""
	@echo "INTERACTIVE COMMANDS:"
	@echo "  make manual                              - Play Snake manually (keyboard controls)"
	@echo ""
	@echo "DEVELOPMENT COMMANDS:"
	@echo "  make lint                                - Check code style with flake8"
	@echo "  make test                                - Run quick functionality test"
	@echo "  make help                                - Show this help message"
	@echo ""
	@echo "ENVIRONMENT VARIABLES:"
	@echo "  PYTHON                                   - Python interpreter (default: python3)"
	@echo "  BOARD_SIZE                               - Board size (default: $(BOARD_SIZE))"
	@echo "  EPISODES_TRAIN                           - Training episodes (default: $(EPISODES_TRAIN))"
	@echo "  EPISODES_EVAL                            - Evaluation episodes (default: $(EPISODES_EVAL))"
	@echo "  DISPLAY_SPEED                            - Animation speed (default: $(DISPLAY_SPEED))"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make train EPISODES_TRAIN=500000         - Train with custom episode count"
	@echo "  make evaluate BOARD_SIZE=15              - Evaluate on 15x15 board"
	@echo "  make play DISPLAY_SPEED=0.1              - Play with slower animation"
	@echo ""
	@echo "============================================================================"
