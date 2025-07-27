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
                 episode_logs=True,
                 optimize_states=True):
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
            optimize_states (bool): Whether to use optimized state representation
            for reduced state space and better generalization.
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
        self._use_optimized_states = optimize_states

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
            float: The calculated initial epsilon value between 0.7 and 0.95.

        Example:
            >>> trainer.board_size = 15
            >>> trainer.calculate_epsilon()
            0.83
        """
        if self.board_size == 0:
            return 0.95  # Increased from 0.9 to 0.95

        min_size, max_size = 5, 20
        min_epsilon, max_epsilon = 0.7, 0.95  # Increased from 0.1-0.7 to 0.7-0.95

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
                args=(thread_id, sync_frequency, num_threads),
                daemon=False
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Clean central Q-table before final compilation
        if len(self.central_q_table) > 8000:
            states_removed = self._clean_central_q_table(max_states=8000)
            if self.episode_logs and states_removed > 0:
                print(f"Central Q-table cleaned: removed {states_removed:,} states")

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
        averaging with state importance consideration.

        This method performs asynchronous Q-table updates by merging learned
        Q-values from worker threads into the central Q-table. It uses weighted
        averaging to gradually incorporate new knowledge while preserving
        accumulated learning from other threads. States with higher learning
        activity get prioritized.

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
        states_merged = 0
        for state_key, local_q_values in local_table.items():
            central_q_values = central_table[state_key]
            
            # Calculate adaptive weight based on Q-value confidence
            local_variance = np.var(local_q_values)
            central_variance = np.var(central_q_values)
            
            # Higher weight for states with more learning (higher variance)
            if local_variance > central_variance * 1.5:
                adaptive_weight = min(weight * 1.5, 0.3)  # Max 30% for high-confidence states
            elif local_variance < central_variance * 0.5:
                adaptive_weight = weight * 0.5  # Lower weight for low-confidence states
            else:
                adaptive_weight = weight
            
            for action_idx in range(3):
                # Weighted average with adaptive weight
                central_q_values[action_idx] = (
                    (1 - adaptive_weight) * central_q_values[action_idx] +
                    adaptive_weight * local_q_values[action_idx]
                )
            states_merged += 1
            
        return states_merged

    def q_learning_thread(self, thread_id, sync_every=100, total_threads=1):
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
        initial_learning_rate = 0.1
        min_learning_rate = 0.01
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
                # Adaptive batch size: larger early, smaller later for better load balancing
                base_batch_size = min(200, max(50, self.central_episode_counter // total_threads))
                adaptive_batch_size = min(base_batch_size, self.central_episode_counter)
                
                for _ in range(adaptive_batch_size):
                    if self.central_episode_counter <= 0:
                        break
                    episodes_batch.append(self.central_episode_counter)
                    self.central_episode_counter -= 1

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
                state_key = self._enhanced_state_to_key(state)
                done = False
                total_reward = 0
                steps = 0

                # Advanced adaptive epsilon with multiple factors
                total_episodes_initial = (local_episode_count + episodes_remaining)
                progress = local_episode_count / max(total_episodes_initial, 1)
                local_epsilon = self._calculate_adaptive_epsilon_decay(
                    local_episode_count, 
                    total_episodes_initial, 
                    len(local_q_table)
                )
                
                # Adaptive learning rate: decreases with progress
                learning_rate = (min_learning_rate + 
                               (initial_learning_rate - min_learning_rate) * 
                               (1.0 - progress))

                # Adaptive gamma for better learning dynamics
                adaptive_gamma = self._calculate_adaptive_gamma(progress, current_board_size)
                
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
                    next_state_key = self._enhanced_state_to_key(next_state)
                    total_reward += reward
                    steps += 1

                    # Initialize Q-values for next state if not seen before
                    if next_state_key not in local_q_table:
                        local_q_table[next_state_key] = [0.0, 0.0, 0.0]

                    # Q-learning update rule with adaptive gamma
                    if done:
                        target = reward
                    else:
                        target = (reward + adaptive_gamma
                                  * max(local_q_table[next_state_key]))

                    current_q = local_q_table[state_key][action_idx]
                    td_error = target - current_q
                    local_q_table[state_key][action_idx] = (current_q
                                                            + learning_rate
                                                            * td_error)

                    # Prioritized learning update
                    learning_rate = self._prioritize_learning_update(local_q_table, state_key, action_idx, td_error, learning_rate)

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

                # Smart synchronization: more frequent early, less frequent later
                adjusted_sync_frequency = max(
                    sync_every // 4,  # Never less than 1/4 of original
                    int(sync_every * (0.5 + 0.5 * progress))  # Increase frequency as training progresses
                )
                
                if local_episode_count % adjusted_sync_frequency == 0:
                    # Merge weight also adapts: higher early, lower later
                    adaptive_weight = 0.15 * (1.0 - progress * 0.5)  # 0.15 -> 0.075
                    
                    with self.central_lock:
                        states_merged = self.merge_q_tables(self.central_q_table,
                                                           local_q_table,
                                                           weight=adaptive_weight)

                    # Reduced I/O for performance
                    if local_episode_count % (adjusted_sync_frequency * 25) == 0:
                        print(f"Thread {thread_id}: Sync at ep {local_episode_count}, "
                              f"LR: {learning_rate:.4f}, ε: {local_epsilon:.4f}, "
                              f"States: {len(local_q_table):,}, Merged: {states_merged}")
                        
                # Clean local Q-table periodically to prevent memory explosion
                if local_episode_count % (adjusted_sync_frequency * 5) == 0:
                    initial_size = len(local_q_table)
                    self._clean_local_q_table(local_q_table, max_states=3000)
                    if len(local_q_table) < initial_size:
                        print(f"Thread {thread_id}: Cleaned Q-table: "
                              f"{initial_size:,} -> {len(local_q_table):,} states")

        # Final synchronization with quality-based weight
        final_weight = min(0.2, len(local_q_table) / 1000.0)  # Weight based on Q-table size
        with self.central_lock:
            self.merge_q_tables(self.central_q_table,
                                local_q_table,
                                weight=final_weight)

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

    def _calculate_adaptive_epsilon_decay(self, episode_count, total_episodes, local_q_table_size):
        """
        Calculates adaptive epsilon decay rate based on multiple factors.
        
        This method dynamically adjusts epsilon decay based on training progress,
        Q-table growth rate, and learning stability to optimize exploration
        vs exploitation balance throughout training.
        
        Args:
            episode_count (int): Current episode number
            total_episodes (int): Total planned episodes
            local_q_table_size (int): Current size of local Q-table
            
        Returns:
            float: Adjusted epsilon value
        """
        # Base progress calculation
        progress = episode_count / max(total_episodes, 1)
        
        # Standard exponential decay
        standard_epsilon = (self.epsilon_min + 
                           (self.epsilon_max - self.epsilon_min) * 
                           np.exp(-self.k * progress))
        
        # Adaptive factors
        
        # Factor 1: Q-table growth rate (slower growth = more exploration needed)
        expected_size = min(1000, episode_count * 2)  # Rough estimate
        if local_q_table_size < expected_size * 0.7:
            # Q-table growing slowly, need more exploration
            exploration_bonus = 0.15 * (1.0 - progress)
        elif local_q_table_size > expected_size * 1.5:
            # Q-table growing rapidly, can reduce exploration
            exploration_bonus = -0.1 * progress
        else:
            exploration_bonus = 0.0
            
        # Factor 2: Training phase adaptation
        if progress < 0.2:  # Early phase - high exploration
            phase_modifier = 1.2
        elif progress < 0.5:  # Mid phase - balanced
            phase_modifier = 1.0
        elif progress < 0.8:  # Late phase - reduce exploration
            phase_modifier = 0.9
        else:  # Final phase - minimal exploration
            phase_modifier = 0.8
            
        # Factor 3: Board size consideration
        if self.board_size == 0:  # Random boards need consistent exploration
            size_modifier = 1.1
        elif self.board_size > 15:  # Large boards need more exploration
            size_modifier = 1.05
        else:
            size_modifier = 1.0
        
        # Combine all factors
        adaptive_epsilon = (standard_epsilon * phase_modifier * size_modifier + 
                           exploration_bonus)
        
        # Ensure bounds
        adaptive_epsilon = max(self.epsilon_min, 
                              min(self.epsilon_max, adaptive_epsilon))
        
        return adaptive_epsilon

    def _update_local_epsilon(self, progress):
        """
        Enhanced epsilon update with adaptive decay.
        
        Args:
            progress (float): Training progress (0.0 to 1.0)
            
        Returns:
            float: Updated epsilon value
        """
        # Use the base calculation as fallback
        base_epsilon = (self.epsilon_min + 
                       (self.epsilon_max - self.epsilon_min) * 
                       np.exp(-self.k * progress))
        
        # Apply adaptive considerations for better exploration balance
        if progress < 0.3:  # Early training - maintain high exploration
            return max(base_epsilon, self.epsilon_max * 0.7)
        elif progress < 0.7:  # Mid training - gradual reduction
            return base_epsilon
        else:  # Late training - rapid convergence to minimum
            return max(self.epsilon_min, base_epsilon * 0.8)

    def calculate_optimal_threads(self):
        """
        Calculates the optimal number of threads for multithreaded Q-learning.

        This method determines the best thread count by considering CPU
        characteristics, board size complexity, and memory constraints.
        It aims to maximize CPU utilization while preventing resource
        contention and memory pressure.

        Returns:
            int: Optimal number of worker threads for training.

        Example:
            >>> trainer.calculate_optimal_threads()
            6  # On an 8-core system with medium board size
        """
        # Get system capabilities
        physical_cores = self._get_physical_cores()
        logical_cores = os.cpu_count() or 4
        
        if self.episode_logs:
            print(f"Physical cores detected: {physical_cores}")
            print(f"Logical cores (os.cpu_count()): {logical_cores}")
        
        # Validate physical core detection
        if physical_cores < logical_cores * 0.6:
            if self.episode_logs:
                print(f"Physical core detection seems inaccurate (ratio:"
                      f" {physical_cores/logical_cores:.2f})")
                print(f"Using logical cores as fallback: {logical_cores}")
            effective_cores = logical_cores
        elif logical_cores > physical_cores * 1.3:
            # Hyperthreading detected
            if self.episode_logs:
                print(f"Hyperthreading detected (logical/physical ratio:"
                      f" {logical_cores/physical_cores:.2f})")
            # Use physical cores + 30% of hyperthreads for optimal performance
            effective_cores = physical_cores + max(1, (logical_cores - physical_cores) // 3)
        else:
            effective_cores = physical_cores

        # Base calculation with conservative approach
        if effective_cores >= 16:
            base_threads = int(effective_cores * 0.85)  # Leave some cores for system
        elif effective_cores >= 8:
            base_threads = int(effective_cores * 0.90)
        else:
            base_threads = max(2, int(effective_cores * 0.95))

        # Adjust based on problem complexity (board size)
        if self.board_size == 0:  # Random sizes - higher complexity
            complexity_factor = 0.9  # Slightly reduce for memory management
        elif self.board_size <= 8:
            complexity_factor = 0.8  # Lower complexity, fewer threads needed
        elif self.board_size <= 12:
            complexity_factor = 1.0  # Standard complexity
        elif self.board_size <= 16:
            complexity_factor = 0.95  # Reduce slightly for memory
        else:
            complexity_factor = 0.85  # Very high complexity, control memory
            
        adjusted_threads = int(base_threads * complexity_factor)
        
        # Memory-based constraints (prevent Q-table explosion)
        if self.q_table_max_size > 100000:  # Large state space
            memory_factor = 0.8  # Reduce threads to control memory
        elif self.q_table_max_size > 50000:
            memory_factor = 0.9
        else:
            memory_factor = 1.0
            
        final_threads = int(adjusted_threads * memory_factor)
        
        # Apply reasonable bounds
        min_threads = 2
        max_threads = min(32, logical_cores * 2)  # Never exceed 2x logical cores
        
        optimal = max(min_threads, min(max_threads, final_threads))
        
        if self.episode_logs:
            print(f"Thread optimization:")
            print(f"  Base threads from {effective_cores} effective cores: {base_threads}")
            print(f"  Board complexity factor: {complexity_factor}")
            print(f"  Memory constraint factor: {memory_factor}")
            print(f"  Final optimal threads: {optimal}")
        
        return optimal

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

    def _clean_local_q_table(self, local_q_table, max_states=5000):
        """
        Cleans a local Q-table by removing least valuable states when it becomes too large.
        
        This method prevents memory explosion by keeping only the most important states
        based on their Q-value variance (states with higher variance are more informative).
        
        Args:
            local_q_table (defaultdict): The local Q-table to clean
            max_states (int): Maximum number of states to keep
            
        Returns:
            None: Modifies local_q_table in place
        """
        if len(local_q_table) <= max_states:
            return
            
        # Calculate importance score for each state (variance of Q-values)
        state_importance = {}
        for state_key, q_values in local_q_table.items():
            # Higher variance = more important state (more learning potential)
            variance = np.var(q_values)
            max_q = max(q_values)
            # Combine variance and max Q-value for importance
            importance = variance + abs(max_q) * 0.1
            state_importance[state_key] = importance
            
        # Keep only the top max_states most important states
        sorted_states = sorted(state_importance.items(), key=lambda x: x[1], reverse=True)
        states_to_keep = set([state for state, _ in sorted_states[:max_states]])
        
        # Create new cleaned Q-table
        cleaned_q_table = defaultdict(lambda: [0.0, 0.0, 0.0])
        for state_key in states_to_keep:
            cleaned_q_table[state_key] = local_q_table[state_key].copy()
            
        # Replace the old Q-table
        local_q_table.clear()
        local_q_table.update(cleaned_q_table)

    def _clean_central_q_table(self, max_states=10000):
        """
        Cleans the central Q-table by removing least important states.
        
        This method prevents the central Q-table from growing too large by
        keeping only states with high learning importance (high variance or
        frequently accessed states).
        
        Args:
            max_states (int): Maximum number of states to keep in central Q-table
            
        Returns:
            int: Number of states removed
        """
        if len(self.central_q_table) <= max_states:
            return 0
            
        # Calculate importance for central states
        state_importance = {}
        for state_key, q_values in self.central_q_table.items():
            # Importance = variance + max absolute Q-value
            variance = np.var(q_values)
            max_abs_q = max(abs(q) for q in q_values)
            importance = variance + max_abs_q * 0.2
            state_importance[state_key] = importance
            
        # Keep top states
        sorted_states = sorted(state_importance.items(), key=lambda x: x[1], reverse=True)
        states_to_keep = set([state for state, _ in sorted_states[:max_states]])
        
        # Count states that will be removed
        states_removed = len(self.central_q_table) - len(states_to_keep)
        
        # Create new cleaned central Q-table
        new_central = self.make_q_table()
        for state_key in states_to_keep:
            new_central[state_key] = self.central_q_table[state_key].copy()
            
        self.central_q_table = new_central
        return states_removed

    def _should_early_stop(self, recent_scores, episode_count, total_episodes):
        """
        Determines if training should stop early based on performance metrics.
        
        This method implements sophisticated early stopping logic that considers
        multiple performance indicators including score improvement, variance
        reduction, and convergence patterns to prevent overtraining.
        
        Args:
            recent_scores (list): Recent episode scores
            episode_count (int): Current episode count
            total_episodes (int): Total planned episodes
            
        Returns:
            bool: True if training should stop early
        """
        # Only consider early stopping after 25% of training (more conservative)
        if episode_count < total_episodes * 0.25:
            return False
            
        # Need sufficient data for reliable statistics
        min_samples = max(150, total_episodes // 100)
        if len(recent_scores) < min_samples:
            return False
            
        # Multi-window analysis for robust detection
        window_size = min(100, len(recent_scores) // 3)
        if len(recent_scores) < window_size * 2:
            return False
            
        recent_window = recent_scores[-window_size:]
        older_window = recent_scores[-window_size*2:-window_size]
        
        recent_avg = np.mean(recent_window)
        older_avg = np.mean(older_window)
        
        # Adaptive threshold based on board size and training progress
        base_threshold = 3.0 if self.board_size <= 10 else 5.0
        progress_factor = episode_count / total_episodes
        adaptive_threshold = base_threshold * (1.0 + progress_factor)
        
        # Check multiple convergence indicators
        improvement = recent_avg - older_avg
        recent_std = np.std(recent_window)
        older_std = np.std(older_window)
        
        # Convergence indicators
        score_stagnation = improvement < adaptive_threshold
        variance_reduction = recent_std < older_std * 0.8  # Variance decreased significantly
        
        # Additional check: are we consistently underperforming?
        performance_decline = recent_avg < older_avg - adaptive_threshold * 2
        
        # Only stop if multiple indicators suggest convergence
        convergence_score = 0
        if score_stagnation:
            convergence_score += 1
        if variance_reduction:
            convergence_score += 1
        if not performance_decline:  # Not declining is good
            convergence_score += 1
            
        # Require strong evidence (2/3 indicators) for early stopping
        should_stop = convergence_score >= 2 and progress_factor > 0.4
        
        if should_stop and self.episode_logs:
            print(f"Early stopping triggered at {progress_factor*100:.1f}% progress:")
            print(f"  Recent avg: {recent_avg:.2f}, Older avg: {older_avg:.2f}")
            print(f"  Improvement: {improvement:.2f} (threshold: {adaptive_threshold:.2f})")
            print(f"  Convergence indicators: {convergence_score}/3")
            
        return should_stop

    def _prioritize_learning_update(self, local_q_table, state_key, action_idx, td_error, learning_rate):
        """
        Applies sophisticated prioritized learning based on multiple factors.
        
        This method implements advanced prioritization that considers TD error
        magnitude, state visitation frequency, and Q-value uncertainty to
        optimize learning efficiency.
        
        Args:
            local_q_table (dict): Local Q-table
            state_key (str): State identifier
            action_idx (int): Action index
            td_error (float): Temporal difference error
            learning_rate (float): Base learning rate
            
        Returns:
            float: Dynamically adjusted learning rate
        """
        error_magnitude = abs(td_error)
        
        # Get current Q-values for variance calculation
        q_values = local_q_table[state_key]
        q_variance = np.var(q_values)
        
        # Multi-factor prioritization
        priority_multiplier = 1.0
        
        # Factor 1: TD Error Magnitude (primary factor)
        if error_magnitude > 100:      # Critical learning opportunity
            priority_multiplier *= 2.0
        elif error_magnitude > 50:     # High importance
            priority_multiplier *= 1.6
        elif error_magnitude > 20:     # Medium importance
            priority_multiplier *= 1.3
        elif error_magnitude > 10:     # Moderate importance
            priority_multiplier *= 1.1
        elif error_magnitude < 1:      # Very low importance
            priority_multiplier *= 0.7
        elif error_magnitude < 3:      # Low importance
            priority_multiplier *= 0.85
        
        # Factor 2: Q-value Uncertainty (high variance = more learning potential)
        if q_variance > 1000:          # High uncertainty
            priority_multiplier *= 1.4
        elif q_variance > 500:         # Medium uncertainty
            priority_multiplier *= 1.2
        elif q_variance > 100:         # Some uncertainty
            priority_multiplier *= 1.1
        elif q_variance < 10:          # Very low uncertainty (converged)
            priority_multiplier *= 0.8
            
        # Factor 3: Action Value Range (wider range = more learning needed)
        q_range = max(q_values) - min(q_values)
        if q_range > 100:              # Large action value differences
            priority_multiplier *= 1.3
        elif q_range > 50:             # Moderate differences
            priority_multiplier *= 1.1
        elif q_range < 5:              # Small differences (near convergence)
            priority_multiplier *= 0.9
            
        # Ensure reasonable bounds
        priority_multiplier = max(0.5, min(2.5, priority_multiplier))
        
        return learning_rate * priority_multiplier

    def _optimize_state_representation(self, state):
        """
        Optimizes state representation to reduce dimensionality and improve
        generalization while maintaining important information for learning.
        
        This method applies intelligent quantization and feature selection
        to the raw state vector, reducing the state space size while
        preserving critical information for decision making.
        
        Args:
            state (numpy.ndarray): Raw state vector from get_state()
            
        Returns:
            tuple: Optimized state representation for Q-table indexing
        """
        state_flat = state.flatten()
        
        # Quantization levels for different distance ranges
        # This reduces continuous values to discrete bins for better Q-table efficiency
        optimized_features = []
        
        for i, value in enumerate(state_flat):
            # Intelligent quantization based on distance importance
            if value >= 0.9:  # Very far/no obstacle
                quantized = 1.0
            elif value >= 0.7:  # Far
                quantized = 0.8
            elif value >= 0.5:  # Medium distance
                quantized = 0.6
            elif value >= 0.3:  # Close
                quantized = 0.4
            elif value >= 0.1:  # Very close
                quantized = 0.2
            else:  # Immediate danger
                quantized = 0.0
                
            optimized_features.append(quantized)
        
        return tuple(optimized_features)

    def _enhanced_state_to_key(self, state):
        """
        Enhanced state-to-key conversion with optimized representation.
        
        This method combines the original state conversion with optimized
        representation to balance between state space reduction and
        information preservation.
        
        Args:
            state: State vector to convert
            
        Returns:
            tuple: Optimized hashable state key
        """
        # Try optimized representation first for better generalization
        if hasattr(self, '_use_optimized_states') and self._use_optimized_states:
            return self._optimize_state_representation(state)
        else:
            # Fallback to original method
            import torch
            if isinstance(state, torch.Tensor):
                return tuple(state.detach().numpy().flatten())
            else:
                return tuple(state.flatten())
