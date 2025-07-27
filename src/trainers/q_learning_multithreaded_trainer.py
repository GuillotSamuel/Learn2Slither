import random
import numpy as np
import os
import threading
import time
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

from board import Board
from state import get_state

GAMMA = 0.99
Q_TABLE_STOP_RATIO = 0.85


class QLearningMultiThreadedTrainer:
    def __init__(self, display_training=False,
                 display_evaluation=False,
                 board_size=10,
                 display_speed=0.1,
                 ultra_rewards=False,
                 render_mode='classic',
                 num_episodes=10000,
                 model_folder_path=None,
                 episode_logs=True):
        """
        Initializes a Multithreaded Q-Learning trainer for reinforcement
        learning.

        Args:
            display_training (bool): Whether to visually display the
            training process.
            display_evaluation (bool): Whether to visually display the
            evaluation process.
            board_size (int): Size of the game board. If 0, uses random
            sizes between 5-20.
            display_speed (float): Speed factor controlling display update
            rate in seconds.
            ultra_rewards (bool): Whether to enable enhanced reward
            scheme.
            render_mode (str): Rendering mode for visuals.
            num_episodes (int): Number of episodes for training.
            model_folder_path (str, optional): Custom path for saving models.
            episode_logs (bool): Whether to display episode logs during
            training/evaluation.
        """
        self.display_training = display_training
        self.display_evaluation = display_evaluation
        self.ultra_rewards = ultra_rewards
        self.board_size = board_size
        self.epsilon = self.calculate_epsilon()
        self.epsilon_min = 0.01
        self.epsilon_max = self.epsilon
        self.k = self.calculate_k(num_episodes)
        self.display_speed = display_speed
        self.render_mode = render_mode
        self.start_time = time.time()
        self.q_table_max_size = 0
        self.model_folder_path = self._gen_model_folder_path(model_folder_path)
        self.episode_logs = episode_logs

        self.scores = []
        self.length_ratios = []
        self.max_scores = []
        self.min_scores = []
        self.moves = []
        self.snake_lengths = []

    def _gen_model_folder_path(self, model_folder_path):
        """
        Generates a model folder path based on board size configuration.

        This method creates a default folder path for saving trained Q-tables
        when no custom path is provided. The path includes descriptive suffixes
        based on the board size configuration to distinguish between different
        model types, specifically for multithreaded training variants.

        Args:
            model_folder_path (str or None): Custom path for model storage.
                If None, generates a default path.

        Returns:
            str: The generated or provided model folder path.

        Example:
            >>> trainer.board_size = 10
            >>> trainer._generate_model_folder_path(None)
            'models_eval_q_learning_multithreaded'
        """
        if model_folder_path is None:
            if self.board_size == 0:
                size_suffix = "random_multithreaded"
            else:
                size_suffix = "eval_q_learning_multithreaded"
            return f"models_{size_suffix}"
        else:
            return model_folder_path

    def calculate_k(self, num_episodes):
        """
        Calculates the decay constant k for epsilon-greedy policy.

        This method computes the decay rate parameter used in the exponential
        decay formula for epsilon in the epsilon-greedy exploration strategy.
        The decay constant is determined based on the target epsilon value
        and the total number of training episodes.

        Args:
            num_episodes (int): Total number of training episodes.

        Returns:
            float: The calculated decay constant k.

        Example:
            >>> trainer.calculate_k(1000)
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

        This method determines the starting exploration rate for the
        epsilon-greedy policy. Larger board sizes get higher epsilon values
        to encourage more exploration, while smaller boards get lower values.
        Random board sizes (board_size=0) use a fixed high epsilon value.

        Args:
            None

        Returns:
            float: The calculated initial epsilon value between 0.1 and 0.7.

        Example:
            >>> trainer.board_size = 15
            >>> trainer.calculate_epsilon()
            0.43
        """
        if self.board_size == 0:
            return 0.9

        min_size, max_size = 5, 20
        min_epsilon, max_epsilon = 0.1, 0.7

        clamped_size = max(min_size, min(max_size, self.board_size))
        size_ratio = (clamped_size - min_size) / (max_size - min_size)
        epsilon = min_epsilon + size_ratio * (max_epsilon - min_epsilon)

        return round(epsilon, 2)

    def define_q_table_max_size(self):
        """
        Calculates the theoretical maximum size of the Q-table.

        This method computes the upper bound on the number of possible states
        based on the state representation dimensions. It considers ray-based
        vision with four directions and four cell types, but applies a
        reduction factor based on board size since not all theoretical states
        are reachable in practice.

        Args:
            None

        Returns:
            None: Updates self.q_table_max_size attribute.

        Example:
            >>> trainer.board_size = 10
            >>> trainer.define_q_table_max_size()
            >>> trainer.q_table_max_size
            1073741824  # Theoretical maximum with board size factor
        """
        max_distance = 10  # From encode_vision function in state.py
        rays_count = 4     # front, left, right, back
        cell_types = 4     # W, S, A, R

        discrete_values_per_dimension = max_distance + 1
        theoretical_max = (discrete_values_per_dimension
                           ** (rays_count * cell_types))

        max_board_distance = min(self.board_size - 1, max_distance)
        practical_values_per_dimension = max_board_distance + 1

        if self.board_size == 0:  # Random board sizes
            avg_board_size = 12.5
            size_factor = avg_board_size / 10
        else:
            size_factor = self.board_size / 10

        practical_max = int(
            (practical_values_per_dimension ** (rays_count * cell_types))
            * (size_factor ** 2)
            * 0.01
        )

        # Set bounds to reasonable values
        min_size = 1000  # Minimum reasonable size
        max_size = 1000000  # Maximum to prevent memory issues

        estimated_size = max(min_size, min(practical_max, max_size))
        self.q_table_max_size = estimated_size

        print(
            f"Q-table size estimation:\n"
            f"  Board size: {self.board_size}\n"
            f"  Theoretical max states: {theoretical_max:,}\n"
            f"  Practical max distance: {max_board_distance}\n"
            f"  Estimated reachable states: {estimated_size:,}\n"
            f"  Memory estimate: ~"
            f"{estimated_size * 3 * 8 / 1024 / 1024:.1f} MB"
        )

    def train(self, num_episodes):
        """
        Trains the agent using multithreaded Q-learning with local Q-tables.

        This method implements a parallelized version of the Q-learning
        algorithm where multiple worker threads train on separate episodes
        with local Q-tables. The threads periodically synchronize with a
        central Q-table to share learned knowledge. This approach
        significantly speeds up training on multi-core systems while
        maintaining learning quality.

        Args:
            num_episodes (int): Total number of training episodes to distribute
                across worker threads.

        Returns:
            None: Updates internal Q-table and saves training metrics, but
            doesn't return values.

        Example:
            >>> trainer = QLearningMultiThreadedTrainer(board_size=10)
            >>> trainer.train(10000)
            Using 8 threads for multithreaded Q-learning...
            Multithreaded Q-learning completed! Total training time: 125.34s
        """
        # Define Q-table size based on board size and state representation
        self.define_q_table_max_size()

        # Number of threads to use - optimize for available CPU cores
        num_threads = self.calculate_optimal_threads()
        print(f"Using {num_threads} threads for multithreaded "
              f"Q-learning (high CPU utilization mode)")

        # Calculate optimal sync frequency based on training size
        sync_frequency = self.calculate_sync_frequency(num_episodes,
                                                       num_threads)
        print(f"Sync frequency set to every {sync_frequency} "
              f"episodes for minimal lock contention")

        # Ultra-optimized locks for maximum performance
        self.central_q_table = self.make_q_table()
        self.central_lock = threading.Lock()  # Use fastest lock type

        # Central episode counter shared by all threads
        self.central_episode_counter = num_episodes
        self.episode_counter_lock = threading.Lock()

        # Metrics tracking with minimal locking
        self.thread_metrics = {
            'scores': [],
            'lengths': [],
            'moves': [],
            'ratios': []
        }
        self.metrics_lock = threading.Lock()

        if self.episode_logs:
            print(f"Starting Q-learning with {num_threads} threads")
            print(f"Total episodes in central pool: {num_episodes}")
            print(f"Sync frequency: every {sync_frequency} local episodes")

        # Launch threads
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=self.q_learning_thread,
                args=(thread_id, sync_frequency),
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

        if self.episode_logs:
            print("\nMultithreaded Q-learning completed!")
            print(f"Total training time: {time.time() - self.start_time:.2f}s")
            print(f"Total unique states learned: "
                  f"{len(self.central_q_table):,}")
            print(f"Central Q-table memory usage: "
                  f"~{len(self.central_q_table) * 3 * 8 / 1024 / 1024:.1f} MB")

    def make_q_table(self):
        """
        Creates a new Q-table using defaultdict for automatic initialization.

        This method creates a Q-table where new states are automatically
        initialized with default Q-values of [0.0, 0.0, 0.0] for the three
        possible actions (FORWARD, LEFT, RIGHT). This prevents KeyError
        exceptions when encountering new states during training.

        Args:
            None

        Returns:
            defaultdict: A Q-table with automatic state initialization.

        Example:
            >>> q_table = trainer.make_q_table()
            >>> q_table['new_state']  # Automatically creates [0.0, 0.0, 0.0]
            [0.0, 0.0, 0.0]
        """
        return defaultdict(lambda: [0.0, 0.0, 0.0])

    def merge_q_tables(self, central_table, local_table, weight=0.1):
        """
        Merges a local Q-table into the central Q-table using weighted
        averaging.

        This method performs asynchronous Q-table updates by merging learned
        Q-values from worker threads into the central Q-table. It uses weighted
        averaging to gradually incorporate new knowledge while preserving
        accumulated learning from other threads.

        Args:
            central_table (defaultdict): The main Q-table shared across
            threads.
            local_table (defaultdict): A thread's local Q-table to merge.
            weight (float): The weight for new values (default 0.1 = 10% new,
                90% old).

        Returns:
            None: Updates central_table in place.

        Example:
            >>> trainer.merge_q_tables(central_q, local_q, weight=0.2)
            # Merges local_q into central_q with 20% weight
        """
        for state_key, local_q_values in local_table.items():
            central_q_values = central_table[state_key]
            for action_idx in range(3):
                # Weighted average: 90% old, 10% new
                central_q_values[action_idx] = (
                    (1 - weight) * central_q_values[action_idx] +
                    weight * local_q_values[action_idx]
                )

    def q_learning_thread(self, thread_id, sync_every=100):
        """
        Q-learning training function executed by individual worker threads.

        This method implements the core training loop for a single worker
        thread in the multithreaded Q-learning system. Each thread maintains
        a local Q-table, processes batches of episodes, and periodically
        synchronizes with the central Q-table to share learned knowledge with
        other threads.

        Args:
            thread_id (int): Unique identifier for this worker thread.
            sync_every (int, optional): Number of episodes between Q-table
                synchronizations. Defaults to 100.

        Returns:
            None: Updates local metrics and contributes to central Q-table.

        Example:
            >>> # Called internally by train() method
            >>> trainer.q_learning_thread(thread_id=0, sync_every=1000)
            Thread 0: Starting training from central episode pool
            Thread 0: Completed 2500 episodes
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

        # Local episode counter for sync frequency
        local_episode_count = 0

        print(f"Thread {thread_id}"
              f": Starting training from central episode pool")

        while True:
            # Get very large batches to minimize lock contention completely
            episodes_batch = []
            with self.episode_counter_lock:
                # Get huge batch of episodes to minimize lock contention
                batch_size = min(100, self.central_episode_counter)
                for _ in range(batch_size):
                    if self.central_episode_counter <= 0:
                        break
                    self.central_episode_counter -= 1
                    episodes_batch.append(self.central_episode_counter)

                if not episodes_batch:
                    break  # No more episodes to process

                episodes_remaining = self.central_episode_counter

            # Process batch of episodes without locks
            for current_global_episode in episodes_batch:
                local_episode_count += 1

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
                state = get_state(board.board,
                                  board.snake,
                                  board.direction)
                state_key = self._state_to_key(state)
                done = False
                total_reward = 0
                steps = 0

                # Update local epsilon based on progress
                total_episodes_initial = (local_episode_count
                                          + episodes_remaining)
                progress = local_episode_count / max(total_episodes_initial, 1)
                local_epsilon = self._update_local_epsilon(progress)

                # Episode loop
                while not done:
                    # Initialize Q-values for new state if not seen before
                    if state_key not in local_q_table:
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
                    next_state = get_state(board.board,
                                           board.snake,
                                           board.direction)
                    next_state_key = self._state_to_key(next_state)
                    total_reward += reward
                    steps += 1

                    # Initialize Q-values for next state if not seen before
                    if next_state_key not in local_q_table:
                        local_q_table[next_state_key] = [0.0, 0.0, 0.0]

                    # Q-learning update rule
                    if done:
                        target = reward
                    else:
                        target = (reward + GAMMA
                                  * max(local_q_table[next_state_key]))

                    current_q = local_q_table[state_key][action_idx]
                    local_q_table[state_key][action_idx] = (current_q
                                                            + learning_rate
                                                            * (target
                                                               - current_q))

                    state_key = next_state_key

                # Store episode metrics
                ratio = len(board.snake) / (current_board_size ** 2)
                local_scores.append(total_reward)
                local_lengths.append(len(board.snake))
                local_moves.append(steps)
                local_ratios.append(ratio)

                # Display episode logs if enabled
                if (
                    self.episode_logs
                    and local_episode_count % 1000 == 0
                ):
                    print(
                        f"[Thread {thread_id}]"
                        f" Episode {local_episode_count}, "
                        f"Q-table: {len(local_q_table):,}, "
                        f"Remaining: {episodes_remaining}"
                    )

                # Extremely rare synchronization with central Q-table
                if local_episode_count % sync_every == 0:
                    # Absolute minimal lock time for merge
                    with self.central_lock:
                        # Very light merge with minimal weight
                        self.merge_q_tables(self.central_q_table,
                                            local_q_table,
                                            weight=0.01)

                    # Almost never print sync info to avoid I/O blocking
                    if local_episode_count % (sync_every * 50) == 0:
                        print(f"Thread {thread_id}"
                              f": Sync at episode {local_episode_count}")

        # Final synchronization - ultra minimal, no I/O blocking
        with self.central_lock:
            # Ultra light final merge
            self.merge_q_tables(self.central_q_table,
                                local_q_table,
                                weight=0.5)

        # Minimal final message
        print(f"Thread {thread_id}: Completed {local_episode_count} episodes")

        # Store final metrics in batch to reduce lock contention
        if local_scores:  # Only lock if we have data to store
            with self.metrics_lock:
                self.thread_metrics['scores'].extend(local_scores)
                self.thread_metrics['lengths'].extend(local_lengths)
                self.thread_metrics['moves'].extend(local_moves)
                self.thread_metrics['ratios'].extend(local_ratios)

        print(f"Thread {thread_id}: Completed training. "
              f"Processed {local_episode_count} episodes. "
              f"Local Q-table size: {len(local_q_table):,}")

    def _update_local_epsilon(self, progress):
        """
        Updates the epsilon value based on training progress for local threads.

        This method adapts the epsilon-greedy exploration rate for individual
        worker threads based on their training progress. It uses a two-phase
        decay strategy similar to the main training loop to balance exploration
        and exploitation over time.

        Args:
            progress (float): Training progress as a value between 0.0 and 1.0.

        Returns:
            float: The updated epsilon value for the current thread.

        Example:
            >>> new_epsilon = trainer._update_local_epsilon_by_progress(0.5)
            >>> new_epsilon
            0.25  # Reduced from initial value
        """
        # Use similar decay strategy as the main training
        decay_phase = 0.80  # first 80% of training

        if progress < decay_phase:
            decay = np.exp(-self.k * progress / decay_phase)
            return (self.epsilon_min
                    + (self.epsilon_max - self.epsilon_min)
                    * decay)
        else:
            final_progress = (progress - decay_phase) / (1.0 - decay_phase)
            final_target = 0.0001
            return (self.epsilon_min
                    - (self.epsilon_min - final_target)
                    * final_progress)

    def _compile_final_metrics(self):
        """
        Compiles training metrics from all worker threads.

        This method aggregates the performance metrics collected by individual
        worker threads during multithreaded training into the main trainer's
        metric storage. The metrics are sorted and organized for consistent
        plotting and analysis.

        Args:
            None

        Returns:
            None: Updates trainer's score, length, and ratio attributes.

        Example:
            >>> trainer._compile_final_metrics()
            # Aggregates metrics from all threads into trainer.scores, etc.
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

    def plot_training_metrics(self):
        """
        Creates and saves a comprehensive training metrics visualization.

        This method generates a multi-panel plot showing four key training
        metrics over episodes: scores (with rolling max/min), steps per
        episode, snake lengths, and length-to-board-size ratios. The
        visualization is saved as a PNG file in the model folder path for
        analysis and monitoring of multithreaded training progress.

        Args:
            None

        Returns:
            None: Saves the plot to "{model_folder_path}/training_metrics.png".

        Example:
            >>> trainer = QLearningMultiThreadedTrainer()
            >>> # After training...
            >>> trainer.plot_training_metrics()
            # Creates training_metrics.png with 4 subplots
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
        Evaluates the trained multithreaded Q-learning agent's performance.

        This method loads a previously trained Q-table from disk and runs the
        agent in evaluation mode (greedy policy with no exploration) for a
        specified number of episodes. It collects performance metrics including
        scores, snake lengths, steps taken, and board utilization ratios. The
        evaluation can optionally display the game visually for each episode.

        Args:
            model_path (str): Path to the saved Q-table pickle file to load for
                evaluation.
            num_episodes (int, optional): Number of evaluation episodes to run.
                Defaults to 10.

        Returns:
            None: Prints evaluation results to console but doesn't return
            values.

        Example:
            >>> trainer = QLearningMultiThreadedTrainer()
            >>> trainer.evaluate("models/end_model.pkl", num_episodes=5)
            [Evaluation] Episode 1/5 - Score: 45.0, Length: 8, Steps: 127...
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

            print(f"[Evaluation] Episode {episode + 1}"
                  f"/{num_episodes} - "
                  f"Score: {total_reward}, "
                  f"Length: {len(board.snake)}, "
                  f"Steps: {steps}, "
                  f"Ratio: {ratio:.4f}, "
                  f"Board Size: {current_board_size}")

        print("\n--- Multithreaded Q-learning Evaluation Summary ---"
              f"Max Length: {np.max(total_lengths)}"
              f"Average Length: {np.mean(total_lengths):.2f}"
              f"Average Steps: {np.mean(total_steps):.2f}"
              f"Average Ratio: {np.mean(total_ratios) * 100:.2f}%")

    def _state_to_key(self, state):
        """
        Converts a state representation to a hashable key for Q-table indexing.

        This method transforms the state vector (either numpy array or PyTorch
        tensor) into a tuple that can be used as a dictionary key for the
        Q-table. The conversion ensures the state can be efficiently stored
        and retrieved from the Q-table data structure in multithreaded
        environments.

        Args:
            state (torch.Tensor or numpy.ndarray): The state representation
            vector.

        Returns:
            tuple: A hashable tuple representation of the state.

        Example:
            >>> state = np.array([[0.1], [0.2], [0.3]])
            >>> key = trainer._state_to_key(state)
            >>> key
            (0.1, 0.2, 0.3)
        """
        import torch
        if isinstance(state, torch.Tensor):
            return tuple(state.detach().numpy().flatten())
        else:
            return tuple(state.flatten())

    def _save_q_table(self, path):
        """
        Saves the Q-table to a file using pickle serialization.

        This method creates the necessary directory structure if it doesn't
        exist and serializes the Q-table dictionary to a binary file for
        later loading and evaluation. The saved Q-table contains all learned
        state-action values from multithreaded training.

        Args:
            path (str): The file path where the Q-table should be saved.

        Returns:
            None

        Example:
            >>> trainer._save_q_table('models/multithreaded_q_table.pkl')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def _load_q_table(self, path):
        """
        Loads a Q-table from a file using pickle deserialization.

        This method reads a previously saved Q-table from a binary file
        and assigns it to the trainer instance for use in evaluation or
        continued training. The loaded Q-table contains all state-action
        values learned during multithreaded training.

        Args:
            path (str): The file path from which to load the Q-table.

        Returns:
            None: Updates self.q_table attribute.

        Example:
            >>> trainer._load_q_table('models/multithreaded_q_table.pkl')
        """
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    def _get_physical_cores(self):
        """
        Attempts to detect the number of physical CPU cores on the system.

        This method tries multiple platform-specific approaches to determine
        the actual number of physical CPU cores, as opposed to logical cores
        which may include hyperthreading. It tries Linux, macOS, and Windows
        detection methods before falling back to logical core count.

        Args:
            None

        Returns:
            int: The number of physical CPU cores detected.

        Example:
            >>> trainer._get_physical_cores()
            4  # On a 4-core CPU without hyperthreading
        """
        try:
            # Method 1: Linux - read from /proc/cpuinfo (most reliable)
            if os.path.exists('/proc/cpuinfo'):
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo = f.read()

                    # Count unique physical IDs
                    physical_ids = set()
                    core_ids_per_phys = {}

                    for line in cpuinfo.split('\n'):
                        if line.startswith('physical id'):
                            phys_id = line.split(':')[1].strip()
                            physical_ids.add(phys_id)
                            if phys_id not in core_ids_per_phys:
                                core_ids_per_phys[phys_id] = set()
                        elif line.startswith('core id'):
                            core_id = line.split(':')[1].strip()
                            # Associate with the last seen physical_id
                            if physical_ids:
                                last_phys_id = list(physical_ids)[-1]
                                core_ids_per_phys[last_phys_id].add(core_id)

                    if physical_ids and core_ids_per_phys:
                        # Calculate total physical cores across all sockets
                        total_physical_cores = sum(
                            len(cores)
                            for cores in core_ids_per_phys.values()
                        )
                        if total_physical_cores > 0:
                            return total_physical_cores

                except (IOError, ValueError, IndexError):
                    pass

            # Method 2: macOS - use sysctl
            if (
                os.name == 'posix'
                and hasattr(os, 'uname')
                and os.uname().sysname == 'Darwin'
            ):
                try:
                    import subprocess
                    result = subprocess.run(['sysctl',
                                             '-n',
                                             'hw.physicalcpu'],
                                            capture_output=True,
                                            text=True,
                                            timeout=5)
                    if result.returncode == 0:
                        return int(result.stdout.strip())
                except (subprocess.SubprocessError,
                        ValueError,
                        FileNotFoundError):
                    pass

            # Method 3: Windows - use WMI if available
            if os.name == 'nt':
                try:
                    import subprocess
                    result = subprocess.run([
                        'wmic', 'cpu', 'get', 'NumberOfCores', '/value'
                    ], capture_output=True, text=True, timeout=10)

                    if result.returncode == 0:
                        total_cores = 0
                        for line in result.stdout.split('\n'):
                            if 'NumberOfCores=' in line:
                                cores = int(line.split('=')[1].strip())
                                total_cores += cores
                        if total_cores > 0:
                            return total_cores
                except (subprocess.SubprocessError,
                        ValueError,
                        FileNotFoundError):
                    pass

        except Exception:
            pass

        # Fallback: use logical core count
        return os.cpu_count() or 4

    def calculate_optimal_threads(self):
        """
        Calculates the optimal number of threads for multithreaded Q-learning.

        This method determines the best number of worker threads based on the
        system's CPU architecture, considering physical cores, logical cores,
        and hyperthreading. It uses an aggressive approach to maximize CPU
        utilization for computationally intensive Q-learning tasks.

        Args:
            None

        Returns:
            int: The optimal number of threads to use for training.

        Example:
            >>> trainer.calculate_optimal_threads()
            8  # On an 8-core system
        """
        physical_cores = self._get_physical_cores()
        logical_cores = os.cpu_count() or 4

        print(f"Physical cores detected: {physical_cores}")
        print(f"Logical cores (os.cpu_count()): {logical_cores}")

        if physical_cores < logical_cores * 0.6:
            print(f"Physical core detection seems inaccurate (ratio:"
                  f" {physical_cores/logical_cores:.2f})")
            print(f"Using logical cores as fallback: {logical_cores}")
            effective_cores = logical_cores
        elif logical_cores > physical_cores * 1.3:
            # Hyperthreading detected
            print(f"Hyperthreading detected (logical/physical ratio:"
                  f" {logical_cores/physical_cores:.2f})")
            print(f"Using logical cores to maximize thread utilization:"
                  f" {logical_cores}")
            effective_cores = logical_cores
        else:
            effective_cores = physical_cores

        # Maximum aggressive approach
        if effective_cores >= 16:
            optimal_threads = effective_cores
        elif effective_cores >= 8:
            optimal_threads = int(effective_cores * 0.98)
        else:
            optimal_threads = max(2, int(effective_cores * 0.95))

        # Set very high upper bound to use all available resources
        optimal_threads = min(optimal_threads, 128)

        # Minor adjustments based on board size and memory
        if self.board_size == 0:  # Random board sizes
            optimal_threads = max(2, int(optimal_threads * 0.9))
        elif self.board_size > 20:
            optimal_threads = max(2, int(optimal_threads * 0.85))

        print(f"Optimal threads calculated: {optimal_threads}"
              f" (from {effective_cores} effective cores)")
        return optimal_threads

    def calculate_sync_frequency(self, num_episodes, num_threads):
        """
        Calculates the optimal synchronization frequency for multithreaded
        learning.

        This method determines how often worker threads should synchronize
        their local Q-tables with the central Q-table. The frequency is
        optimized based on the total number of episodes and number of threads
        to balance learning efficiency with computational overhead.

        Args:
            num_episodes (int): Total number of training episodes.
            num_threads (int): Number of worker threads.

        Returns:
            int: The number of episodes between synchronizations.

        Example:
            >>> trainer.calculate_sync_frequency(10000, 4)
            1000  # Sync every 1000 episodes
        """
        avg_episodes_per_thread = num_episodes // num_threads

        # Base frequency depends on training size
        if num_episodes <= 500:              # Very small training
            base_frequency = 200
        elif num_episodes <= 5000:           # Small training
            base_frequency = 1000
        elif num_episodes <= 50000:          # Medium training
            base_frequency = 10000
        elif num_episodes <= 500000:         # Large training
            base_frequency = 25000
        elif num_episodes <= 2000000:        # Very large training
            base_frequency = 50000
        else:                                # Massive training
            base_frequency = 100000

        # Adjust based on thread count
        if num_threads >= 8:
            base_frequency = int(base_frequency * 1.2)
        elif num_threads <= 2:
            base_frequency = int(base_frequency * 0.7)

        # Ensure reasonable bounds
        min_frequency = 500
        max_frequency = max(avg_episodes_per_thread // 2, 50)

        sync_frequency = max(min_frequency, min(base_frequency, max_frequency))

        return sync_frequency
