# ============================================================================#
# Makefile for Snake Q-Learning Project                                       #
# ============================================================================#
# This Makefile provides convenient commands for training, evaluating, and    #
# playing the Snake game using different reinforcement learning algorithms    #
# including Q-Learning and Deep Q-Learning with various map configurations.   #
# ============================================================================#

# Python interpreter and main script configuration
PYTHON=python
MAIN=src/main.py

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
BOARD_SIZE=10        # Default board size (10x10 grid)
DISPLAY_SPEED=0.07   # Animation speed for visual gameplay (seconds between frames)

# ============================================================================#
# TRAINING TARGETS                                                            #
# ============================================================================#

# Train Q-Learning agent on fixed-size board with enhanced rewards
train:
	$(PYTHON) $(MAIN) --mode train --episodes 1000000 --board_size $(BOARD_SIZE) --ultra_rewards --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_FOLDER)

# Train Q-Learning agent on random board sizes (5-20) for better generalization
train_random_map:
	$(PYTHON) $(MAIN) --mode train --episodes 5000000 --board_size 0 --ultra_rewards --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER)

# Train Q-Learning agent on fixed-size board with enhanced rewards with multithreading
train_multithreaded:
	$(PYTHON) $(MAIN) --mode train --episodes 1000000 --board_size $(BOARD_SIZE) --ultra_rewards --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_FOLDER)

# Train Q-Learning agent on random board sizes (5-20) for better generalization with multithreading
train_random_map_multithreaded:
	$(PYTHON) $(MAIN) --mode train --episodes 5000000 --board_size 0 --ultra_rewards --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER)

# Train Deep Q-Learning agent on fixed-size board using neural network
train_deep_q_learning:
	$(PYTHON) $(MAIN) --mode train --episodes 10000 --board_size $(BOARD_SIZE) --ultra_rewards --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_FOLDER)

# Train Deep Q-Learning agent on random board sizes for improved adaptability
train_deep_q_learning_random_map:
	$(PYTHON) $(MAIN) --mode train --episodes 10000 --board_size 0 --ultra_rewards --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_RANDOM_MAP_FOLDER)

# ============================================================================#
# EVALUATION TARGETS                                                          #
# ============================================================================#

# Evaluate all three models in 10x10 maps
evaluate_all: 
	$(PYTHON) $(MAIN) --mode evaluate --episodes 1000 --model $(MODEL_Q_LEARNING) --board_size $(BOARD_SIZE) --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_FOLDER) --episode_logs False
	$(PYTHON) $(MAIN) --mode evaluate --episodes 1000 --model $(MODEL_Q_LEARNING_MULTITHREADED) --board_size $(BOARD_SIZE) --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_FOLDER) --episode_logs False
	$(PYTHON) $(MAIN) --mode evaluate --episodes 1000 --model $(MODEL_DEEP_LEARNING) --board_size $(BOARD_SIZE) --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_FOLDER) --episode_logs False

# Evaluate Q-Learning agent performance on fixed-size board (1000 episodes)
evaluate:
	$(PYTHON) $(MAIN) --mode evaluate --episodes 1000 --model $(MODEL_Q_LEARNING) --board_size $(BOARD_SIZE) --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_FOLDER)

# Evaluate Q-Learning agent performance on random board sizes
evaluate_random_map:
	$(PYTHON) $(MAIN) --mode evaluate --episodes 1000 --model $(MODEL_Q_LEARNING_RANDOM_MAP) --board_size 0 --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER)

# Evaluate Q-Learning multithreaded agent performance on fixed-size board (1000 episodes)
evaluate_multithreaded:
	$(PYTHON) $(MAIN) --mode evaluate --episodes 1000 --model $(MODEL_Q_LEARNING_MULTITHREADED) --board_size $(BOARD_SIZE) --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_FOLDER)

# Evaluate Q-Learning multithreaded agent performance on random board sizes
evaluate_random_map_multithreaded:
	$(PYTHON) $(MAIN) --mode evaluate --episodes 1000 --model $(MODEL_Q_LEARNING_RANDOM_MAP_MULTITHREADED) --board_size 0 --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER_MULTITHREADED)

# Evaluate Deep Q-Learning agent performance (1000 episodes for thorough testing)
evaluate_deep_q_learning:
	$(PYTHON) $(MAIN) --mode evaluate --episodes 1000 --model $(MODEL_DEEP_LEARNING) --board_size $(BOARD_SIZE) --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_FOLDER)

# Evaluate Deep Q-Learning agent on random board sizes (100 episodes)
evaluate_random_map_deep_q_learning:
	$(PYTHON) $(MAIN) --mode evaluate --episodes 100 --model $(MODEL_DEEP_LEARNING_RANDOM_MAP) --board_size 0 --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_RANDOM_MAP_FOLDER)

# ============================================================================#
# GAMEPLAY TARGETS                                                            #
# ============================================================================#

# Watch the Q-Learning agent play with visual rendering on fixed-size board
play:
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_Q_LEARNING) --board_size $(BOARD_SIZE) --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_FOLDER)

# Watch the Q-Learning agent play on random board sizes (5-20)
play_random_map:
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_Q_LEARNING_RANDOM_MAP) --board_size 0 --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method q_learning --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER)

# Watch the Q-Learning multithreaded agent play with visual rendering on fixed-size board
play_multithreaded:
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_Q_LEARNING_MULTITHREADED) --board_size $(BOARD_SIZE) --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_FOLDER_MULTITHREADED)

# Watch the Q-Learning multithreaded agent play on random board sizes (5-20)
play_random_map_multithreaded:
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_Q_LEARNING_RANDOM_MAP_MULTITHREADED) --board_size 0 --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method q_learning_multithreaded --model_folder_path $(MODEL_Q_LEARNING_RANDOM_MAP_FOLDER_MULTITHREADED)

# Watch the Deep Q-Learning agent play with visual rendering
play_deep_q_learning:
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_DEEP_LEARNING) --board_size $(BOARD_SIZE) --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_FOLDER)

# Watch the Deep Q-Learning agent play on random board sizes
play_random_map_deep_q_learning:
	$(PYTHON) $(MAIN) --mode play --model $(MODEL_DEEP_LEARNING_RANDOM_MAP) --board_size 0 --display_speed $(DISPLAY_SPEED) --render_mode classic --training_method deep_q_learning --model_folder_path $(MODEL_DEEP_LEARNING_RANDOM_MAP_FOLDER)

# ============================================================================#
# INTERACTIVE TARGETS                                                         #
# ============================================================================#

# Play Snake manually using keyboard controls (10x10 board, basic rendering)
manual:
	$(PYTHON) $(MAIN) --mode manual --render_mode basic --board_size 10

# ============================================================================#
# HELP TARGET                                                                 #
# ============================================================================#

# Display usage information and available commands
help:
	@echo "============================================================================"
	@echo "Snake Q-Learning Project - Available Commands"
	@echo "============================================================================"
	@echo ""
	@echo "TRAINING COMMANDS:"
	@echo "  make train                               - Train Q-Learning agent (fixed board)"
	@echo "  make train_random_map                    - Train Q-Learning agent (random board sizes)"
	@echo "  make train_multithreaded                 - Train Q-Learning agent (fixed board, multithreaded)"
	@echo "  make train_random_map_multithreaded      - Train Q-Learning agent (random boards, multithreaded)"
	@echo "  make train_deep_q_learning               - Train Deep Q-Learning agent (neural network)"
	@echo "  make train_deep_q_learning_random_map    - Train Deep Q-Learning (random boards)"
	@echo ""
	@echo "EVALUATION COMMANDS:"
	@echo "  make evaluate                            - Evaluate Q-Learning agent (1000 episodes)"
	@echo "  make evaluate_random_map                 - Evaluate Q-Learning agent (random boards)"
	@echo "  make evaluate_multithreaded              - Evaluate Q-Learning agent (multithreaded)"
	@echo "  make evaluate_random_map_multithreaded   - Evaluate Q-Learning agent (random boards, multithreaded)"
	@echo "  make evaluate_deep_q_learning            - Evaluate Deep Q-Learning (1000 episodes)"
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
	@echo "  make help                                - Show this help message"
	@echo ""
	@echo "============================================================================"
