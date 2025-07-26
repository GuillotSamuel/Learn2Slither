# src/trainer.py
import random
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import threading
import time
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

from board import Board
from state import get_state
from mlp import MLP

GAMMA = 0.99
LEARNING_RATE = 0.001
Q_TABLE_STOP_RATIO = 0.85  # Stop training when Q-table reaches 85% of estimated max size
MODEL_SAVE_FOLDER = "models"
GRAPH_SAVE_FOLDER = "graphs"


class Trainer:
    def __init__(self, display_training=False,
                 display_evaluation=False,
                 board_size=10,
                 display_speed=0.1,
                 ultra_rewards=False,
                 render_mode='classic',
                 num_episodes=10000,
                 training_method='q_learning',
                 model_folder_path=None,
                 episode_logs=True):
        """
        Initializes a reinforcement learning trainer with specified configuration parameters.

        Creates a new Trainer instance with customizable settings for training and evaluation.
        Sets up the neural network model, optimizer, epsilon parameters for exploration,
        and initializes containers for tracking performance metrics during training.

        Args:
            display_training (bool): Whether to visually display the training process.
                Defaults to False.
            display_evaluation (bool): Whether to visually display the evaluation process.
                Defaults to False.
            board_size (int): Size of the game board. If 0, uses random sizes between 5-20.
                Defaults to 10.
            display_speed (float): Speed factor controlling display update rate in seconds.
                Defaults to 0.1.
            ultra_rewards (bool): Whether to enable enhanced reward scheme. Defaults to False.
            render_mode (str): Rendering mode for visuals. Defaults to 'classic'.
            num_episodes (int): Number of episodes for training. Defaults to 10000.
            training_method (str): Training algorithm to use ('q_learning', 'deep_q_learning',
                or 'q_learning_multithreaded'). Defaults to 'q_learning'.
            model_folder_path (str, optional): Custom path for saving models. If None,
                generates path based on board size. Defaults to None.
            episode_logs (bool): Whether to display episode logs during training/evaluation.
                Defaults to True.

        Returns:
            None

        Example:
            >>> trainer = Trainer(display_training=True, board_size=15, num_episodes=5000)
            >>> trainer.train(1000)
        """
        self.training_method = training_method
        self.display_training = display_training
        self.display_evaluation = display_evaluation
        self.ultra_rewards = ultra_rewards
        self.board_size = board_size
        self.model = MLP()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.epsilon = self.calculate_epsilon()
        self.epsilon_min = 0.01
        self.epsilon_max = self.epsilon
        self.k = self.calculate_k(num_episodes)
        self.display_speed = display_speed
        self.render_mode = render_mode
        self.start_time = time.time()
        self.q_table_max_size = 0
        self.model_folder_path = self._generate_model_folder_path(model_folder_path)
        self.episode_logs = episode_logs

        self.scores = []
        self.length_ratios = []
        self.max_scores = []
        self.min_scores = []
        self.moves = []
        self.snake_lengths = []

    def _generate_model_folder_path(self, model_folder_path):
        """
        Generates the model folder path based on board size configuration.
        
        This method creates a descriptive folder name that indicates the board size
        configuration used for training. This helps organize models by their
        training configuration and avoid conflicts between different setups.
        
        Args:
            model_folder_path (str or None): Custom model folder path. If None,
                a path will be generated based on board size.
                
        Returns:
            str: The model folder path to use for saving models and metrics.
            
        Example:
            >>> trainer = Trainer(board_size=10)
            >>> trainer._generate_model_folder_path(None)
            'models_size_10'
            
            >>> trainer = Trainer(board_size=0)  # Random sizes
            >>> trainer._generate_model_folder_path(None)
            'models_random_5_to_20'
        """
        if model_folder_path is None:
            if self.board_size == 0:
                size_suffix = "random_5_to_20"
            else:
                size_suffix = f"size_{self.board_size}"
            return f"models_{size_suffix}"
        else:
            return model_folder_path

    def calculate_k(self, num_episodes):
        """
        Calculates the decay constant k for epsilon-greedy policy in
        reinforcement learning.

        This function computes the value of k such that the epsilon
        value decays from epsilon_max toward epsilon_min over the
        course of training. The calculation is based on the exponential
        decay formula: epsilon = epsilon_min + (epsilon_max -
        epsilon_min) * exp(-k) Here, k is derived to ensure that
        epsilon approaches epsilon_min by the end of the specified
        number of episodes.

        Args:
            num_episodes (int): The total number of training episodes.
            Although provided, it is not used in the current calculation.

        Returns:
            float: The computed decay constant k for epsilon decay.

        Example:
        >>> agent.epsilon_max = 1.0
        >>> agent.epsilon_min = 0.1
        >>> agent.calculate_k(num_episodes=1000)
        4.605170185988091
        """
        epsilon_target = self.epsilon_min + 0.01
        decay_ratio = ((epsilon_target - self.epsilon_min)
                       / (self.epsilon_max - self.epsilon_min))

        k = -np.log(decay_ratio)
        return k

    def calculate_epsilon(self):
        """
        Calculates the initial epsilon value based on the board size.

        This method determines the starting exploration rate (epsilon)
        for the agent, scaling it between predefined minimum and maximum
        epsilon values depending on the size of the game board. The board
        size is clamped within a specified range to ensure epsilon stays
        within the intended bounds.

        Args:
            None

        Returns:
            float: The initial epsilon value rounded to two decimal places.

        Example:
            >>> self.board_size = 10
            >>> self.calculate_epsilon()
            0.4
        """
        if self.board_size == 0:
            return 0.9

        min_size, max_size = 5, 20
        min_epsilon, max_epsilon = 0.1, 0.7

        clamped_size = max(min_size, min(max_size, self.board_size))
        size_ratio = (clamped_size - min_size) / (max_size - min_size)
        epsilon = min_epsilon + size_ratio * (max_epsilon - min_epsilon)

        return round(epsilon, 2)

    def update_epsilon(self, episode, num_episodes):
        """
        Updates the epsilon value for epsilon-greedy action selection
        using exponential decay.

        This method progressively reduces the epsilon value over the
        course of training episodes, starting from epsilon_max and
        decaying towards epsilon_min according to the formula:
        epsilon = epsilon_min + (epsilon_max - epsilon_min)
                  * exp(-k * episode / num_episodes).
        This encourages more exploration early in training
        (0% to 80% of episodes)
        and more exploitation later in the second phase of the training
        (80% to 100% of episodes).

        Args:
            episode (int): The current episode number in training.
            num_episodes (int): The total number of training episodes.

        Returns:
            None

        Example:
            >>> agent.update_epsilon(50, 1000)
        """
        # Define phases
        decay_phase = int(0.80 * num_episodes)  # first 80% episodes
        final_phase = num_episodes - decay_phase  # last 20%

        if episode < decay_phase:
            # Exponential decay in early phase
            decay = np.exp(-self.k * episode / decay_phase)
            self.epsilon = (self.epsilon_min
                            + (self.epsilon_max - self.epsilon_min) * decay)
        else:
            # Linear decay in final phase: from epsilon_min to almost zero
            progress = (episode - decay_phase) / final_phase
            final_target = 0.0001  # final epsilon value for pure exploitation
            self.epsilon = (self.epsilon_min
                            - (self.epsilon_min - final_target) * progress)

        # Safety check
        self.epsilon = max(self.epsilon, 0.0001)

    def save_model(self, path):
        """
        Saves the current model's state dictionary to the specified
        file path.

        Creates the directory path if it does not exist, then saves
        the model parameters using PyTorch's serialization functionality.

        Args:
            path (str): The file path where the model state dictionary
            will be saved.

        Returns:
            None

        Example:
            >>> agent.save_model('checkpoints/model_epoch_10.pth')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Loads the model state dictionary from the specified file
        path and sets the model to evaluation mode.

        This replaces the current model parameters with those stored
        at the given path, and switches the model to eval mode to
        disable training-specific layers like dropout.

        Args:
            path (str): The file path from which to load the model
            state dictionary.

        Returns:
            None

        Example:
            >>> agent.load_model('checkpoints/model_epoch_10.pth')
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def train(self, num_episodes):
        """
        Trains the reinforcement learning model using the specified training method.

        Determines which training algorithm to use based on the training_method
        parameter and delegates to the appropriate training function. Supports
        deep Q-learning, traditional Q-learning, and multithreaded Q-learning.

        Args:
            num_episodes (int): The total number of training episodes to run.

        Returns:
            None

        Example:
            >>> trainer = Trainer(training_method='q_learning')
            >>> trainer.train(1000)
        """
        if self.training_method == 'deep_q_learning':
            self.train_deep_q_learning(num_episodes)
        elif self.training_method == 'q_learning':
            self.train_q_learning(num_episodes)
        elif self.training_method == 'q_learning_multithreaded':
            self.train_q_learning_multithreaded(num_episodes)
        else:
            raise ValueError(f"Unknown training method: {self.training_method}")
        
    def define_q_table_max_size(self):
        """
        Defines the size of the Q-table based on the board size and state representation.

        This method calculates the theoretical maximum size of the Q-table based on
        the state representation used in the game. The state is represented as a 
        16-dimensional vector (4 directions × 4 cell types: Wall, Snake, Apple green, Apple Red).
        Each dimension contains normalized distances from 0 to 1 with discrete steps.
        
        The calculation considers:
        - 4 vision rays (front, left, right, back)
        - 4 cell types per ray (W=Wall, S=Snake, A=Apple green, R=Apple red)
        - Distance normalization with max_distance=10, giving discrete steps
        - Board size constraints that limit possible configurations
        
        Returns:
            int: The estimated maximum number of unique states in the Q-table

        Example:
            >>> trainer.define_q_table_size()
            >>> print(f"Estimated Q-table size: {trainer.q_table_max_size}")
        """
        # State representation analysis:
        # - 16 dimensions total (4 rays × 4 cell types)
        # - Each dimension: normalized distance from 0 to 1
        # - max_distance = 10 (from state.py), so we have discrete steps
        
        max_distance = 10  # From encode_vision function in state.py
        rays_count = 4     # front, left, right, back
        cell_types = 4     # W, S, A, R
        
        # Discrete distance values: 0, 0.1, 0.2, ..., 1.0 (11 possible values)
        # But in practice, distances are min(dist/max_distance, 1.0)
        # So we have max_distance + 1 possible discrete values per dimension
        discrete_values_per_dimension = max_distance + 1
        
        # Theoretical maximum if all combinations were possible
        theoretical_max = discrete_values_per_dimension ** (rays_count * cell_types)
        
        # Practical considerations:
        # 1. Board size limits the maximum distance
        max_board_distance = min(self.board_size - 1, max_distance)
        practical_values_per_dimension = max_board_distance + 1
        
        # 2. Not all state combinations are valid due to game constraints
        # For example, if there's a wall at distance 3, there can't be 
        # a snake/apple at distance 5 in the same direction
        
        # Estimate based on board size - larger boards have more possible states
        if self.board_size == 0:  # Random board sizes
            # Use average case for random boards (5-20 range)
            avg_board_size = 12.5
            size_factor = avg_board_size / 10
        else:
            size_factor = self.board_size / 10
        
        # Practical estimate considering game constraints
        # This is a heuristic based on the observation that not all
        # theoretical combinations are reachable in actual gameplay
        practical_max = int(
            (practical_values_per_dimension ** (rays_count * cell_types)) 
            * (size_factor ** 2)  # Quadratic scaling with board size
            * 0.01  # Constraint factor (only ~1% of theoretical states are reachable)
        )
        
        # Set bounds to reasonable values
        min_size = 1000    # Minimum reasonable size
        max_size = 1000000 # Maximum to prevent memory issues
        
        estimated_size = max(min_size, min(practical_max, max_size))
        
        # Store the estimated size
        self.q_table_max_size = estimated_size
        
        print(
            f"Q-table size estimation:\n"
            f"  Board size: {self.board_size}\n"
            f"  Theoretical max states: {theoretical_max:,}\n"
            f"  Practical max distance: {max_board_distance}\n"
            f"  Estimated reachable states: {estimated_size:,}\n"
            f"  Memory estimate: ~{estimated_size * 3 * 8 / 1024 / 1024:.1f} MB"
        )
        
    def train_q_learning_multithreaded(self, num_episodes):
        """
        Trains the agent using multithreaded Q-learning with local Q-tables that
        periodically merge into a central shared Q-table.

        This method implements a multithreaded Q-learning architecture where each
        thread maintains its own local Q-table and trains independently. Threads
        periodically synchronize by merging their local Q-tables with a central
        shared Q-table using weighted averaging. This approach improves training
        speed through parallelization while maintaining learning stability.

        Args:
            num_episodes (int): The total number of training episodes to run
                               across all threads.

        Returns:
            None

        Example:
            >>> trainer.train_q_learning_multithreaded(1000)
        """
        # Define Q-table size based on board size and state representation
        self.define_q_table_max_size()
        
        # Number of threads to use - optimize for available CPU cores
        num_threads = self.calculate_optimal_threads()
        episodes_per_thread = num_episodes // num_threads
        
        # Calculate optimal sync frequency based on training size
        sync_frequency = self.calculate_sync_frequency(num_episodes, num_threads)
        
        # Central Q-table and lock for thread-safe access
        self.central_q_table = self.make_q_table()
        self.central_lock = threading.Lock()
        
        # Metrics tracking
        self.thread_metrics = {
            'scores': [],
            'lengths': [],
            'moves': [],
            'ratios': []
        }
        self.metrics_lock = threading.Lock()
        
        print(f"Starting multithreaded Q-learning with {num_threads} threads")
        print(f"Episodes per thread: {episodes_per_thread}")
        print(f"Sync frequency: every {sync_frequency} episodes")
        print(f"Total synchronizations per thread: {episodes_per_thread // sync_frequency}")
        
        # Launch threads
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=self.q_learning_thread,
                args=(thread_id, episodes_per_thread, sync_frequency),
                daemon=False
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Final metrics compilation
        self._compile_final_metrics()
        
        # Save final model
        final_path = os.path.join(self.model_folder_path, "end_model.pkl")
        self._save_q_table(final_path)
        
        # Plot training metrics
        self.plot_training_metrics()
        
        print(f"\nMultithreaded Q-learning completed!")
        print(f"Total training time: {time.time() - self.start_time:.2f}s")
        print(f"Total unique states learned: {len(self.central_q_table):,}")
        print(f"Central Q-table memory usage: ~{len(self.central_q_table) * 3 * 8 / 1024 / 1024:.1f} MB")
    
    def make_q_table(self):
        """
        Creates a new Q-table using defaultdict for automatic state initialization.

        Constructs a Q-table as a defaultdict that automatically initializes
        new states with zero Q-values for all three possible actions when
        first accessed. This eliminates the need for explicit state checking.

        Args:
            None

        Returns:
            defaultdict: A Q-table mapping states to lists of 3 Q-values
                (one for each action: FORWARD, LEFT, RIGHT).

        Example:
            >>> trainer = Trainer()
            >>> q_table = trainer.make_q_table()
            >>> q_table['new_state']  # Automatically returns [0.0, 0.0, 0.0]
            [0.0, 0.0, 0.0]
        """
        return defaultdict(lambda: [0.0, 0.0, 0.0])
    
    def merge_q_tables(self, central_table, local_table, weight=0.1):
        """
        Merges a local Q-table into the central Q-table using weighted averaging.

        Combines the knowledge from a local Q-table into the central shared Q-table
        using weighted averaging. This allows gradual incorporation of new knowledge
        while maintaining stability by preventing abrupt changes to established values.

        Args:
            central_table (defaultdict): The central shared Q-table to update.
            local_table (defaultdict): The local Q-table containing new knowledge.
            weight (float): The weight for local table values. Defaults to 0.1.

        Returns:
            None

        Example:
            >>> central = trainer.make_q_table()
            >>> local = trainer.make_q_table()
            >>> local['state1'] = [1.0, 2.0, 3.0]
            >>> trainer.merge_q_tables(central, local, weight=0.2)
        """
        for state_key, local_q_values in local_table.items():
            central_q_values = central_table[state_key]
            for action_idx in range(3):
                # Weighted average: 90% old, 10% new
                central_q_values[action_idx] = (
                    (1 - weight) * central_q_values[action_idx] + 
                    weight * local_q_values[action_idx]
                )
    
    def q_learning_thread(self, thread_id, episodes, sync_every=100):
        """
        Q-learning training function for a single thread.
        
        Each thread maintains its own local Q-table and trains independently.
        Periodically, it synchronizes with the central Q-table by:
        1. Copying central Q-table values to local table
        2. Merging local improvements back to central table
        
        Args:
            thread_id (int): Unique identifier for this thread.
            episodes (int): Number of episodes this thread should run.
            sync_every (int): Frequency of synchronization with central table.
        """
        # Local Q-table for this thread
        local_q_table = self.make_q_table()
        learning_rate = 0.1
        local_epsilon = self.epsilon
        
        # Local metrics
        local_scores = []
        local_lengths = []
        local_moves = []
        local_ratios = []
        
        print(f"Thread {thread_id}: Starting training for {episodes} episodes")
        
        for episode in range(episodes):
            # Determine board size
            if self.board_size == 0:
                current_board_size = random.randint(5, 20)
            else:
                current_board_size = self.board_size
            
            # Initialize game
            board = Board(board_size=current_board_size,
                         ultra_rewards=self.ultra_rewards,
                         render_mode=self.render_mode,
                         episodes_logs=self.episode_logs)
            state = get_state(board.board, board.snake, board.direction)
            state_key = self._state_to_key(state)
            done = False
            total_reward = 0
            steps = 0
            
            # Update local epsilon
            total_episodes = episodes * 8  # Approximate total episodes across all threads
            local_epsilon = self._update_local_epsilon(local_epsilon, episode + thread_id * episodes, total_episodes)
            
            # Episode loop
            while not done:
                # Initialize Q-values for new state if not seen before
                if state_key not in local_q_table:
                    # Try to get values from central table first
                    with self.central_lock:
                        if state_key in self.central_q_table:
                            local_q_table[state_key] = self.central_q_table[state_key].copy()
                        else:
                            local_q_table[state_key] = [0.0, 0.0, 0.0]
                
                # Select action using epsilon-greedy policy
                if random.random() < local_epsilon:
                    action_idx = random.randint(0, 2)
                else:
                    action_idx = np.argmax(local_q_table[state_key])
                
                actions = ['FORWARD', 'LEFT', 'RIGHT']
                action_str = actions[action_idx]
                
                # Take action
                next_state_raw, reward, done = board.step(action_str)
                next_state = get_state(board.board, board.snake, board.direction)
                next_state_key = self._state_to_key(next_state)
                total_reward += reward
                steps += 1
                
                # Initialize Q-values for next state if not seen before
                if next_state_key not in local_q_table:
                    with self.central_lock:
                        if next_state_key in self.central_q_table:
                            local_q_table[next_state_key] = self.central_q_table[next_state_key].copy()
                        else:
                            local_q_table[next_state_key] = [0.0, 0.0, 0.0]
                
                # Q-learning update rule
                if done:
                    target = reward
                else:
                    target = reward + GAMMA * max(local_q_table[next_state_key])
                
                current_q = local_q_table[state_key][action_idx]
                local_q_table[state_key][action_idx] = current_q + learning_rate * (target - current_q)
                
                state_key = next_state_key
            
            # Store episode metrics
            ratio = len(board.snake) / (current_board_size ** 2)
            local_scores.append(total_reward)
            local_lengths.append(len(board.snake))
            local_moves.append(steps)
            local_ratios.append(ratio)
            
            # Periodic synchronization with central Q-table
            if (episode + 1) % sync_every == 0:
                with self.central_lock:
                    # Merge local improvements into central table
                    self.merge_q_tables(self.central_q_table, local_q_table, weight=0.1)
                    
                    # Update local table with central improvements
                    for state_key, central_q_values in self.central_q_table.items():
                        if state_key in local_q_table:
                            # Keep local table, but update with some central knowledge
                            for action_idx in range(3):
                                local_q_table[state_key][action_idx] = (
                                    0.8 * local_q_table[state_key][action_idx] +
                                    0.2 * central_q_values[action_idx]
                                )
                        else:
                            # Copy new states from central table
                            local_q_table[state_key] = central_q_values.copy()
                
                if self.episode_logs:
                    print(f"Thread {thread_id}: Episode {episode + 1}/{episodes}, "
                          f"Reward: {total_reward:.2f}, "
                          f"Length: {len(board.snake)}, "
                          f"Local Q-table: {len(local_q_table):,}, "
                          f"Central Q-table: {len(self.central_q_table):,}")
        
        # Final synchronization
        with self.central_lock:
            self.merge_q_tables(self.central_q_table, local_q_table, weight=0.15)
        
        # Store final metrics
        with self.metrics_lock:
            self.thread_metrics['scores'].extend(local_scores)
            self.thread_metrics['lengths'].extend(local_lengths)
            self.thread_metrics['moves'].extend(local_moves)
            self.thread_metrics['ratios'].extend(local_ratios)
        
        print(f"Thread {thread_id}: Completed training. Local Q-table size: {len(local_q_table):,}")
    
    def _update_local_epsilon(self, local_epsilon, episode, total_episodes):
        """
        Updates epsilon value for a local thread using the same decay strategy as single-threaded.

        Applies the same two-phase epsilon decay strategy used in single-threaded training
        to maintain consistency across different training modes. Uses exponential decay
        for the first 80% of episodes and linear decay for the final 20%.

        Args:
            local_epsilon (float): Current epsilon value for this thread.
            episode (int): Current episode number within this thread.
            total_episodes (int): Total episodes across all threads.

        Returns:
            float: Updated epsilon value for the thread.

        Example:
            >>> trainer = Trainer()
            >>> new_epsilon = trainer._update_local_epsilon(0.5, 100, 1000)
            >>> print(new_epsilon)
            0.3
        """
        # Use similar decay strategy as the main training
        decay_phase = int(0.80 * total_episodes)
        final_phase = total_episodes - decay_phase

        if episode < decay_phase:
            decay = np.exp(-self.k * episode / decay_phase)
            return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * decay
        else:
            progress = (episode - decay_phase) / final_phase
            final_target = 0.0001
            return self.epsilon_min - (self.epsilon_min - final_target) * progress
    
    def _compile_final_metrics(self):
        """
        Compiles training metrics from all threads into the main trainer metrics.

        Aggregates performance metrics collected from all training threads into
        the main trainer's metric containers. Calculates rolling statistics
        for scores and copies the central Q-table for compatibility with
        single-threaded evaluation methods.

        Args:
            None

        Returns:
            None

        Example:
            >>> trainer = Trainer()
            >>> # After multithreaded training completes
            >>> trainer._compile_final_metrics()
            >>> print(len(trainer.scores))  # Total episodes from all threads
        """
        # Sort all metrics by episode order (approximate)
        all_scores = self.thread_metrics['scores']
        all_lengths = self.thread_metrics['lengths']
        all_moves = self.thread_metrics['moves']
        all_ratios = self.thread_metrics['ratios']
        
        # Store in main trainer
        self.scores = all_scores
        self.snake_lengths = all_lengths
        self.moves = all_moves
        self.length_ratios = all_ratios
        
        # Calculate rolling statistics
        self.max_scores = []
        self.min_scores = []
        
        for i in range(len(self.scores)):
            window_start = max(0, i - 99)
            window_scores = self.scores[window_start:i + 1]
            self.max_scores.append(max(window_scores))
            self.min_scores.append(min(window_scores))
        
        # Copy central Q-table to main Q-table for compatibility
        self.q_table = dict(self.central_q_table)
    
    def train_q_learning(self, num_episodes):
        """
        Trains the agent using traditional Q-learning with a Q-table.

        This method implements the classic Q-learning algorithm using a 
        Q-table to store state-action values. For each episode, it initializes 
        a game board, runs the game until completion, and updates the Q-table 
        using the Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)].
        It uses epsilon-greedy exploration and tracks training metrics.

        Args:
            num_episodes (int): The total number of training episodes
            to run.

        Returns:
            None

        Example:
            >>> trainer.train_q_learning(100)
        """
        # Define Q-table size based on board size and state representation
        self.define_q_table_max_size()

        # Initialize Q-table as a dictionary
        self.q_table = {}
        learning_rate = 0.1
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            if self.board_size == 0:
                current_board_size = random.randint(5, 20)
            else:
                current_board_size = self.board_size

            board = Board(board_size=current_board_size,
                          ultra_rewards=self.ultra_rewards,
                          render_mode=self.render_mode,
                          episodes_logs=self.episode_logs)
            state = get_state(board.board,
                              board.snake,
                              board.direction)
            state_key = self._state_to_key(state)
            done = False
            total_reward = 0
            steps = 0

            if (
                self.display_training
                and (episode in self.get_display_episodes(num_episodes))
            ):
                window_thread = threading.Thread(target=board.render_in_window,
                                                 daemon=True)
                window_thread.start()

            # Epsilon decay
            self.update_epsilon(episode, num_episodes)

            while not done:
                # Initialize Q-values for new state if not seen before
                if state_key not in self.q_table:
                    self.q_table[state_key] = [0.0, 0.0, 0.0]
                
                # Select action using epsilon-greedy policy
                if random.random() < self.epsilon:
                    action_idx = random.randint(0, 2)
                else:
                    action_idx = np.argmax(self.q_table[state_key])
                
                actions = ['FORWARD', 'LEFT', 'RIGHT']
                action_str = actions[action_idx]
                
                next_state_raw, reward, done = board.step(action_str)
                next_state = get_state(board.board,
                                       board.snake,
                                       board.direction)
                next_state_key = self._state_to_key(next_state)
                total_reward += reward
                steps += 1

                # Initialize Q-values for next state if not seen before
                if next_state_key not in self.q_table:
                    self.q_table[next_state_key] = [0.0, 0.0, 0.0]

                # Q-learning update rule
                if done:
                    target = reward
                else:
                    target = reward + GAMMA * max(self.q_table[next_state_key])
                
                current_q = self.q_table[state_key][action_idx]
                self.q_table[state_key][action_idx] = current_q + learning_rate * (target - current_q)

                state_key = next_state_key

                if (
                    self.display_training
                    and (episode in self.get_display_episodes(num_episodes))
                ):
                    time.sleep(self.display_speed)

            self.scores.append(total_reward)
            self.max_scores.append(max(self.scores[-100:]))
            self.min_scores.append(min(self.scores[-100:]))
            self.moves.append(steps)
            self.snake_lengths.append(len(board.snake))
            ratio = len(board.snake) / (current_board_size ** 2)
            self.length_ratios.append(ratio)

            if (
                self.display_training
                and (episode in self.get_display_episodes(num_episodes))
            ):
                window_thread.join()

            model_save = self.get_model_saves(num_episodes)
            if episode in model_save:
                path = os.path.join(self.model_folder_path,
                                    f"{model_save[episode]}_model.pkl")
                self._save_q_table(path)

            # Check if we should stop training based on Q-table size
            current_q_table_ratio = len(self.q_table) / self.q_table_max_size
            # if current_q_table_ratio >= Q_TABLE_STOP_RATIO:
            #     print(f"\nEarly stopping triggered!")
            #     print(f"Q-table size reached {current_q_table_ratio:.1%} of estimated maximum")
            #     print(f"Current size: {len(self.q_table):,} / {self.q_table_max_size:,}")
            #     print(f"Stopping at episode {episode + 1}/{num_episodes}")
            #     break

            if self.episode_logs:
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Episode time: {time.time() - episode_start_time:.2f}s, "
                      f"Total Reward: {total_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, "
                      f"Snake length: {len(board.snake)}, "
                      f"Map size: {current_board_size}, "
                      f"Ratio: {ratio * 100:.2f}%, "
                      f"Q-table: {len(self.q_table):,} / {self.q_table_max_size:,} "
                      f"({current_q_table_ratio:.1%})")

        self.plot_training_metrics()
        print("Q-learning training completed."
              f" Total time: {time.time() - self.start_time:.2f}s")

    def train_deep_q_learning(self, num_episodes):
        """
        Trains the reinforcement learning model over a specified
        number of episodes.

        This method runs a training loop for the given number of
        episodes. For each episode, it initializes a game board
        (random size if not fixed), runs the game until completion,
        and updates the model using Q-learning with epsilon-greedy
        exploration. It progressively decays epsilon to reduce
        exploration over time. Training metrics such as rewards,
        steps, and snake lengths are tracked and optionally displayed.
        Models are periodically saved based on predefined checkpoints.
        Finally, training metrics are plotted and total
        training time is printed.

        Args:
        num_episodes (int): The total number of training episodes to run.

        Returns:
        None

        Example:
        >>> trainer = SnakeTrainer()
        >>> trainer.train(100)
        """
        for episode in range(num_episodes):
            episode_start_time = time.time()
            if self.board_size == 0:
                current_board_size = random.randint(5, 20)
            else:
                current_board_size = self.board_size

            board = Board(board_size=current_board_size,
                          ultra_rewards=self.ultra_rewards,
                          render_mode=self.render_mode,
                          episodes_logs=self.episode_logs)
            state = get_state(board.board,
                              board.snake,
                              board.direction)
            done = False
            total_reward = 0
            steps = 0

            if (
                self.display_training
                and (episode in self.get_display_episodes(num_episodes))
            ):
                window_thread = threading.Thread(target=board.render_in_window,
                                                 daemon=True)
                window_thread.start()

            # Epsilon decay
            self.update_epsilon(episode, num_episodes)

            while not done:
                current_q_values = self.model(state)
                action_idx = self.select_action(current_q_values)
                actions = ['FORWARD', 'LEFT', 'RIGHT']
                action_str = actions[action_idx]
                next_state_raw, reward, done = board.step(action_str)
                next_state = get_state(board.board,
                                       board.snake,
                                       board.direction)
                total_reward += reward
                steps += 1

                if done:
                    target = reward
                else:
                    max_q_next = torch.max(self.model(next_state))
                    target = reward + GAMMA * max_q_next

                self.optimizer.zero_grad()
                predicted_q_values = current_q_values[action_idx]
                loss = self.criterion(predicted_q_values.unsqueeze(0),
                                      torch.tensor([target],
                                      dtype=torch.float32))
                loss.backward()
                self.optimizer.step()

                state = next_state

                if (
                    self.display_training
                    and (episode in self.get_display_episodes(num_episodes))
                ):
                    time.sleep(self.display_speed)

            self.scores.append(total_reward)
            self.max_scores.append(max(self.scores[-100:]))
            self.min_scores.append(min(self.scores[-100:]))
            self.moves.append(steps)
            self.snake_lengths.append(len(board.snake))
            ratio = len(board.snake) / (current_board_size ** 2)
            self.length_ratios.append(ratio)

            if (
                self.display_training
                and (episode in self.get_display_episodes(num_episodes))
            ):
                window_thread.join()

            model_save = self.get_model_saves(num_episodes)
            if episode in model_save:
                path = os.path.join(self.model_folder_path,
                                    f"{model_save[episode]}_model.pkl")
                self.save_model(path)

            if self.episode_logs:
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Episode time: {time.time() - episode_start_time:.2f}s, "
                      f"Total Reward: {total_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, "
                      f"Snake length: {len(board.snake)}, "
                      f"Map size: {current_board_size}, "
                      f"Ratio: {ratio * 100:.2f}%")

        self.plot_training_metrics()
        print("Training completed."
              f" Total time: {time.time() - self.start_time:.2f}s")

    def get_display_episodes(self, num_episodes):
        """
        Calculate a set of episode indices to display based on
        deciles and recent episodes.

        This function computes a set of episode indices that
        includes the decile-based breakpoints
        (i.e., 10%, 20%, ..., 90% through the total number of episodes)
        as well as the last 10 episodes. This is useful for selectively
        displaying representative episodes from a series, emphasizing
        both the distribution across the whole set and recent activity.

        Args:
            num_episodes (int): The total number of episodes.

        Returns:
            set[int]: A set of episode indices representing decile
            cutoffs and the
            last 10 episodes.

        Example:
            >>> get_display_episodes(100)
            {10, 20, 30, 40, 50, 60, 70, 80, 90, 90,
            91, 92, 93, 94, 95, 96, 97, 98, 99}
        """
        deciles = {int(num_episodes * i / 10) for i in range(1, 10)}
        last_episodes = set(range(num_episodes - 10, num_episodes))
        return deciles.union(last_episodes)

    def get_model_saves(self, num_episodes):
        """
        Generate a dictionary mapping specific episode indices
        to save labels.

        This function creates a dictionary where keys are selected
        episode indices, and values are string labels describing the
        save points. It includes fixed indices (0, 9, 99), as well as
        indices corresponding to the first decile, the midpoint, and
        the last episode. This can be used to mark important checkpoints
        for saving models during training.

        Args:
            num_episodes (int): The total number of episodes.

        Returns:
            dict[int, str]: A dictionary mapping episode indices to
            descriptive save labels.

        Example:
            >>> get_model_saves(100)
            {0: '1', 9: '10', 99: '100', 10: 'decile', 50: 'mid', 99: 'end'}
        """
        saves = {
            0: f"{1}",
            9: f"{10}",
            99: f"{100}",
            num_episodes // 10: "decile",
            num_episodes // 2: "mid",
            num_episodes - 1: "end"
        }
        return saves

    def select_action(self, q_values):
        """
        Selects an action using epsilon-greedy policy based on Q-values.

        Implements epsilon-greedy action selection where a random action is
        chosen with probability epsilon (exploration), otherwise the action
        with the highest Q-value is selected (exploitation). Assumes three
        possible actions indexed 0, 1, and 2.

        Args:
            q_values (torch.Tensor): A tensor containing Q-values for each
                possible action in the current state.

        Returns:
            int: The index of the selected action (0, 1, or 2).

        Example:
            >>> import torch
            >>> trainer = Trainer()
            >>> q_vals = torch.tensor([1.0, 2.5, 0.3])
            >>> action = trainer.select_action(q_vals)
            >>> print(action)  # Most likely 1 (highest Q-value)
            1
        """
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        else:
            return torch.argmax(q_values).item()

    def plot_training_metrics(self):
        """
        Plots and saves training metrics visualization to a PNG file.

        Generates a comprehensive multi-panel plot showing the evolution of key
        training metrics: scores (with rolling max/min), steps per episode,
        snake lengths, and length-to-board-size ratios. The plot is saved as
        'training_metrics.png' in the model folder.

        Args:
            None

        Returns:
            None

        Example:
            >>> trainer = Trainer()
            >>> # After training completes
            >>> trainer.plot_training_metrics()
            # Creates 'models/training_metrics.png' with 4-panel visualization
        """
        episodes = range(1, len(self.scores) + 1)

        plt.figure(figsize=(12, 12))

        # Scores
        plt.subplot(4, 1, 1)
        plt.plot(episodes,
                 self.scores,
                 label='Score')
        plt.plot(episodes,
                 self.max_scores,
                 label='Max Score (100 eps)')
        plt.plot(episodes,
                 self.min_scores,
                 label='Min Score (100 eps)')
        plt.title('Scores over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend()

        # Steps
        plt.subplot(4, 1, 2)
        plt.plot(episodes,
                 self.moves,
                 color='orange',
                 label='Steps per Episode')
        plt.title('Snake Movements per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()

        # Snake Length
        plt.subplot(4, 1, 3)
        plt.plot(episodes,
                 self.snake_lengths,
                 color='green',
                 label='Snake Length')
        plt.title('Snake Length over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.legend()

        # Ratio Length / Board Size²
        plt.subplot(4, 1, 4)
        plt.plot(episodes,
                 self.length_ratios,
                 color='purple',
                 label='Length/Map Ratio')
        plt.title('Snake Length Ratio (Length / Board Cells)')
        plt.xlabel('Episode')
        plt.ylabel('Ratio')
        plt.legend()

        plt.tight_layout()
        os.makedirs(self.model_folder_path, exist_ok=True)
        plt.savefig(f"{self.model_folder_path}/training_metrics.png")

    def evaluate(self, model_path, num_episodes=10):
        """
        Evaluates the performance of a trained model using the appropriate method.

        Determines which evaluation method to use based on the training_method
        and delegates to the corresponding evaluation function. Supports evaluation
        of both deep Q-learning and traditional Q-learning models.

        Args:
            model_path (str): The path to the trained model file to evaluate.
            num_episodes (int): The number of episodes to run for evaluation.
                Defaults to 10.

        Returns:
            None

        Example:
            >>> trainer = Trainer(training_method='q_learning')
            >>> trainer.evaluate('models/end_model.pkl', num_episodes=5)
        """
        if self.training_method == 'deep_q_learning':
            self.evaluate_deep_q_learning(model_path, num_episodes)
        elif self.training_method == 'q_learning':
            self.evaluate_q_learning(model_path, num_episodes)
        else:
            raise ValueError(f"Unknown evaluation method: {self.training_method}")
        
    def evaluate_q_learning(self, model_path, num_episodes=10):
        """
        Evaluates the Q-learning agent's performance using the saved Q-table.

        This method loads a saved Q-table, runs it in evaluation
        mode on the environment, and collects metrics including
        total score, snake length, steps taken, and the length-to-board-size
        ratio for each episode. Optionally, it renders the game
        visually at a controlled speed. After each episode, metrics
        are printed to the console.

        Args:
            model_path (str): The path to the saved Q-table file.
            num_episodes (int): The number of episodes to run for evaluation.

        Returns:
            None

        Example:
            >>> trainer.evaluate_q_learning('models/end_model.pkl', 10)
        """
        self._load_q_table(model_path)

        total_scores = []
        total_lengths = []
        total_steps = []
        total_ratios = []

        for episode in range(num_episodes):
            if self.board_size == 0:
                current_board_size = random.randint(8, 11)
            else:
                current_board_size = self.board_size
            board = Board(board_size=current_board_size,
                          render_mode=self.render_mode,
                          episodes_logs=self.episode_logs)
            state = get_state(board.board,
                              board.snake,
                              board.direction)
            state_key = self._state_to_key(state)
            done = False
            total_reward = 0
            steps = 0

            if self.display_evaluation:
                window_thread = threading.Thread(target=board.render_in_window,
                                                 daemon=True)
                window_thread.start()

            while not done:
                # Use greedy policy (no exploration) for evaluation
                if state_key in self.q_table:
                    action_idx = np.argmax(self.q_table[state_key])
                else:
                    # If state not in Q-table, choose random action
                    action_idx = random.randint(0, 2)
                
                actions = ['FORWARD', 'LEFT', 'RIGHT']
                action_str = actions[action_idx]

                next_state_raw, reward, done = board.step(action_str)
                state = get_state(board.board,
                                  board.snake,
                                  board.direction)
                state_key = self._state_to_key(state)
                total_reward += reward
                steps += 1

                if self.display_evaluation:
                    time.sleep(self.display_speed)

            if self.display_evaluation:
                window_thread.join()

            ratio = len(board.snake) / (current_board_size ** 2)
            total_scores.append(total_reward)
            total_lengths.append(len(board.snake))
            total_steps.append(steps)
            total_ratios.append(ratio)

            if self.episode_logs:
                print(f"[Evaluation] Episode {episode + 1}/{num_episodes} - "
                      f"Score: {total_reward}, "
                      f"Length: {len(board.snake)}, "
                      f"Steps: {steps}, "
                      f"Ratio: {ratio:.4f}, "
                      f"Board Size: {current_board_size}")

        print("\n--- Q-learning Evaluation Summary ---"
              f"Max Length: {np.max(total_lengths)}"
              f"Average Length: {np.mean(total_lengths):.2f}"
              f"Average Steps: {np.mean(total_steps):.2f}"
              f"Average Ratio: {np.mean(total_ratios) * 100:.2f}%")

    def evaluate_deep_q_learning(self, model_path, num_episodes=10):
        """
        Evaluate the trained model's performance over a
        specified number of episodes.

        This method loads a saved model, runs it in evaluation
        mode on the environment, and collects metrics including
        total score, snake length, steps taken, and the length-to-board-size
        ratio for each episode. Optionally, it renders the game
        visually at a controlled speed. After each episode, metrics
        are printed to the console.

        Args:
            model_path (str): Path to the saved model file to load.
            num_episodes (int, optional): Number of episodes to run
            for evaluation. Defaults to 10.

        Returns:
            None

        Example:
            >>> agent.evaluate('models/snake_model.pth', num_episodes=5)
            [Evaluation] Episode 1/5 - Score: 12,
                                       Length: 5,
                                       Steps: 50,
                                       Ratio: 0.0313
            ...
        """
        self.load_model(model_path)
        self.model.eval()

        total_scores = []
        total_lengths = []
        total_steps = []
        total_ratios = []

        for episode in range(num_episodes):
            if self.board_size == 0:
                current_board_size = random.randint(8, 11)
            else:
                current_board_size = self.board_size
            board = Board(board_size=current_board_size,
                          render_mode=self.render_mode,
                          episodes_logs=self.episode_logs)
            state = get_state(board.board,
                              board.snake,
                              board.direction)
            done = False
            total_reward = 0
            steps = 0

            if self.display_evaluation:
                window_thread = threading.Thread(target=board.render_in_window,
                                                 daemon=True)
                window_thread.start()

            while not done:
                with torch.no_grad():
                    q_values = self.model(state)
                    action_idx = torch.argmax(q_values).item()
                actions = ['FORWARD', 'LEFT', 'RIGHT']
                action_str = actions[action_idx]

                next_state_raw, reward, done = board.step(action_str)
                state = get_state(board.board,
                                  board.snake,
                                  board.direction)
                total_reward += reward
                steps += 1

                if self.display_evaluation:
                    time.sleep(self.display_speed)

            if self.display_evaluation:
                window_thread.join()

            ratio = len(board.snake) / (current_board_size ** 2)
            total_scores.append(total_reward)
            total_lengths.append(len(board.snake))
            total_steps.append(steps)
            total_ratios.append(ratio)

            if self.episode_logs:
                print(f"[Evaluation] Episode {episode + 1}/{num_episodes} - "
                      f"Score: {total_reward}, "
                      f"Length: {len(board.snake)}, "
                      f"Steps: {steps}, "
                      f"Ratio: {ratio:.4f}, "
                      f"Board Size: {current_board_size}")

        print("\n--- Evaluation Summary ---"
              f"Max Length: {np.max(total_lengths)}"
              f"Average Length: {np.mean(total_lengths):.2f}"
              f"Average Steps: {np.mean(total_steps):.2f}"
              f"Average Ratio: {np.mean(total_ratios) * 100:.2f}%")

    def _state_to_key(self, state):
        """
        Converts a state tensor or array to a hashable key for Q-table storage.

        Transforms state representations (torch.Tensor or numpy.ndarray) into
        tuple format that can be used as dictionary keys in the Q-table.
        Handles tensor detachment and flattening automatically.

        Args:
            state (torch.Tensor or numpy.ndarray): The game state to convert
                into a hashable format.

        Returns:
            tuple: A flattened tuple representation of the state suitable
                for use as a Q-table dictionary key.

        Example:
            >>> import torch
            >>> trainer = Trainer()
            >>> state = torch.tensor([1.0, 2.0, 3.0])
            >>> key = trainer._state_to_key(state)
            >>> print(key)
            (1.0, 2.0, 3.0)
        """
        if isinstance(state, torch.Tensor):
            return tuple(state.detach().numpy().flatten())
        else:
            return tuple(state.flatten())
    
    def _save_q_table(self, path):
        """
        Saves the Q-table to a file using pickle serialization.

        Creates the necessary directory structure if it doesn't exist and
        serializes the Q-table dictionary to disk using Python's pickle module.
        This allows the trained Q-table to be loaded later for evaluation or
        continued training.

        Args:
            path (str): The file path where the Q-table will be saved.
                Directory will be created if it doesn't exist.

        Returns:
            None

        Example:
            >>> trainer = Trainer()
            >>> # After training
            >>> trainer._save_q_table('models/my_model.pkl')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def _load_q_table(self, path):
        """
        Loads a Q-table from a file using pickle deserialization.

        Deserializes a previously saved Q-table from disk and stores it
        in the trainer's q_table attribute. This allows loading of
        pre-trained models for evaluation or continued training.

        Args:
            path (str): The file path from which to load the Q-table.
                File must exist and contain a valid pickled Q-table.

        Returns:
            None

        Example:
            >>> trainer = Trainer()
            >>> trainer._load_q_table('models/trained_model.pkl')
            >>> # Q-table is now loaded and ready for evaluation
        """
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    def calculate_sync_frequency(self, num_episodes, num_threads):
        """
        Calculates the optimal synchronization frequency based on the total number of episodes.
        
        This function determines how often threads should synchronize with the central Q-table
        based on the total training volume. For larger trainings, less frequent synchronization
        is more efficient while maintaining learning quality.
        
        Args:
            num_episodes (int): Total number of episodes across all threads.
            num_threads (int): Number of threads that will be used.
            
        Returns:
            int: The number of episodes between synchronizations for each thread.
            
        Examples:
            >>> trainer.calculate_sync_frequency(100, 4)    # Demo
            10  # Sync every 10 episodes
            
            >>> trainer.calculate_sync_frequency(1000, 4)   # Small training
            50  # Sync every 50 episodes
            
            >>> trainer.calculate_sync_frequency(100000, 8) # Medium training
            500  # Sync every 500 episodes
            
            >>> trainer.calculate_sync_frequency(5000000, 8) # Large training
            2500  # Sync every 2500 episodes
        """
        episodes_per_thread = num_episodes // num_threads
        
        # Define sync frequency rules based on total episodes
        if num_episodes <= 200:
            # Very small training (demo) - sync frequently for quick feedback
            sync_freq = max(5, episodes_per_thread // 10)
        elif num_episodes <= 1000:
            # Small training - moderate sync frequency
            sync_freq = max(10, episodes_per_thread // 8)
        elif num_episodes <= 10000:
            # Medium training - less frequent sync
            sync_freq = max(50, episodes_per_thread // 6)
        elif num_episodes <= 100000:
            # Large training - infrequent sync
            sync_freq = max(200, episodes_per_thread // 5)
        elif num_episodes <= 1000000:
            # Very large training - very infrequent sync
            sync_freq = max(1000, episodes_per_thread // 4)
        else:
            # Massive training (5M+) - extremely infrequent sync
            sync_freq = max(2000, episodes_per_thread // 3)
        
        # Ensure sync frequency doesn't exceed episodes per thread
        sync_freq = min(sync_freq, episodes_per_thread)
        
        # Ensure minimum sync of at least once per thread
        sync_freq = max(sync_freq, 1)
        
        return sync_freq
    
    def calculate_optimal_threads(self):
        """
        Calculates the optimal number of threads based on system resources.
        
        This function determines the ideal number of threads to use for multithreaded
        Q-learning training. It considers CPU cores and applies heuristics to maximize 
        performance without overwhelming the system.
        
        Returns:
            int: The optimal number of threads to use.
            
        Examples:
            >>> trainer.calculate_optimal_threads()  # 4-core system
            6  # Uses 1.5x cores for I/O bound tasks
            
            >>> trainer.calculate_optimal_threads()  # 8-core system
            12 # Uses 1.5x cores for I/O bound tasks
            
            >>> trainer.calculate_optimal_threads()  # 16-core system
            20 # Uses 1.25x cores (more conservative for high-end systems)
        """
        # Get system information
        cpu_count = os.cpu_count() or 4
        
        try:
            # Try to get system load average (Linux/Mac)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()[0]  # 1-minute load average
                system_busy = load_avg > cpu_count * 0.8
            else:
                # Windows or no load average available
                system_busy = False
        except:
            system_busy = False
        
        # Base calculation: Use more threads than cores since Q-learning has I/O waits
        if cpu_count <= 2:
            # Very small systems: Use 1.5x cores
            base_threads = max(2, int(cpu_count * 1.5))
        elif cpu_count <= 4:
            # Small systems: Use 1.5x cores
            base_threads = int(cpu_count * 1.5)
        elif cpu_count <= 8:
            # Medium systems: Use 1.5x cores
            base_threads = int(cpu_count * 1.5)
        elif cpu_count <= 16:
            # Large systems: Use 1.25x cores (more conservative)
            base_threads = int(cpu_count * 1.25)
        else:
            # Very large systems: Use 1.2x cores (even more conservative)
            base_threads = int(cpu_count * 1.2)
        
        # Adjust based on detected system load
        if system_busy:
            # System seems busy, reduce threads
            base_threads = max(2, int(base_threads * 0.75))
            load_status = "busy, reducing threads"
        else:
            load_status = "normal"
        
        # Apply reasonable bounds
        optimal_threads = max(2, min(base_threads, cpu_count * 2))
        
        # Ensure we don't exceed practical limits
        optimal_threads = min(optimal_threads, 32)  # Hard cap at 32 threads
        
        print(f"System resources detected:")
        print(f"  CPU cores: {cpu_count}")
        print(f"  System load: {load_status}")
        print(f"  Optimal threads calculated: {optimal_threads} ({optimal_threads/cpu_count:.2f}x cores)")
        
        return optimal_threads
