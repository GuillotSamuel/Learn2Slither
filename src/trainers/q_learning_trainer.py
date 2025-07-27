import random
import numpy as np
import os
import torch
import threading
import time
import matplotlib.pyplot as plt
import pickle

from board import Board
from state import get_state

GAMMA = 0.99
Q_TABLE_STOP_RATIO = 0.85


class QLearningTrainer:
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
        Initializes a Q-Learning trainer for reinforcement learning.

        Args:
            display_training (bool): Whether to visually display the training
            process.
            display_evaluation (bool): Whether to visually display the
            evaluation process.
            board_size (int): Size of the game board. If 0, uses random sizes
            between 5-20.
            display_speed (float): Speed factor controlling display update rate
            in seconds.
            ultra_rewards (bool): Whether to enable enhanced reward scheme.
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
        self.q_table = {}

    def _gen_model_folder_path(self, model_folder_path):
        """
        Generates a model folder path based on board size configuration.

        This method creates a default folder path for saving trained Q-tables
        when no custom path is provided. The path includes descriptive suffixes
        based on the board size configuration to distinguish between different
        model types.

        Args:
            model_folder_path (str or None): Custom path for model storage.
                If None, generates a default path.

        Returns:
            str: The generated or provided model folder path.

        Example:
            >>> trainer.board_size = 10
            >>> trainer._generate_model_folder_path(None)
            'models_eval_q_learning'
        """
        if model_folder_path is None:
            if self.board_size == 0:
                size_suffix = "random_size"
            else:
                size_suffix = "eval_q_learning"
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
            float: The calculated initial epsilon value between 0.1 and 0.9.

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

    def update_epsilon(self, episode, num_episodes):
        """
        Updates the epsilon value using a two-phase decay strategy.

        This method implements a sophisticated epsilon decay with two phases:
        exponential decay for the first 80% of episodes followed by linear
        decay to near-zero for pure exploitation in the final 20% of episodes.
        This approach balances exploration and exploitation effectively.

        Args:
            episode (int): Current episode number (0-indexed).
            num_episodes (int): Total number of training episodes.

        Returns:
            None: Updates self.epsilon in place.

        Example:
            >>> trainer.update_epsilon(500, 1000)
            >>> trainer.epsilon  # Updated epsilon value
            0.05
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

        # Estimate based on board size
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
        Trains the Q-learning agent using epsilon-greedy policy
        and Q-table updates.

        This method implements the core Q-learning algorithm with a tabular
        representation. It initializes the Q-table, runs training episodes with
        decaying epsilon exploration, updates Q-values using the Bellman
        equation, and periodically saves model checkpoints. Training metrics
        are collected and optionally displayed visually for selected episodes.

        Args:
            num_episodes (int): Number of training episodes to run the
                Q-learning algorithm for.

        Returns:
            None: Updates internal Q-table and saves training metrics, but
                doesn't return values.

        Example:
            >>> trainer = QLearningTrainer(board_size=10, num_episodes=1000)
            >>> trainer.train(1000)
            Episode 1/1000, Episode time: 0.45s, Total Reward: -12.50...
            Q-learning training completed. Total time: 450.23s
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
                self.q_table[state_key][action_idx] = (current_q
                                                       + learning_rate
                                                       * (target - current_q))

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

            if self.episode_logs:
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Episode time: "
                      f"{time.time() - episode_start_time:.2f}s, "
                      f"Total Reward: {total_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, "
                      f"Snake length: {len(board.snake)}, "
                      f"Map size: {current_board_size}, "
                      f"Ratio: {ratio * 100:.2f}%, "
                      f"Q-table: {len(self.q_table):,}"
                      f" / {self.q_table_max_size:,} "
                      f"({current_q_table_ratio:.1%})")

        self.plot_training_metrics()
        print("Q-learning training completed."
              f" Total time: {time.time() - self.start_time:.2f}s")

    def get_display_episodes(self, num_episodes):
        """
        Calculates episode indices for visual display during training.

        This method determines which episodes should have visual rendering
        enabled by combining decile milestones (10%, 20%, ..., 90%) with
        the final 10 episodes. This provides a good balance between monitoring
        training progress and performance efficiency by avoiding excessive
        rendering.

        Args:
            num_episodes (int): Total number of training episodes to calculate
                display points from.

        Returns:
            set[int]: Set of episode indices (0-based) where visual display
                should be enabled.

        Example:
            >>> trainer = QLearningTrainer()
            >>> episodes = trainer.get_display_episodes(100)
            >>> episodes
            {10, 20, 30, 40, 50, 60, 70, 80, 90, 91, 92, 93, 94,
            95, 96, 97, 98, 99}
        """
        deciles = {int(num_episodes * i / 10) for i in range(1, 10)}
        last_episodes = set(range(num_episodes - 10, num_episodes))
        return deciles.union(last_episodes)

    def get_model_saves(self, num_episodes):
        """
        Generates episode-to-label mapping for model checkpoint saving.

        This method creates a dictionary that maps specific episode indices to
        descriptive labels for saving model checkpoints at key training
        milestones. Includes early episodes (1st, 10th, 100th), fractional
        milestones (decile, mid-point), and the final episode for comprehensive
        evaluation coverage.

        Args:
            num_episodes (int): Total number of training episodes to determine
                milestone positions.

        Returns:
            dict[int, str]: Dictionary mapping episode indices (0-based) to
                string labels for saved model files.

        Example:
            >>> trainer = QLearningTrainer()
            >>> saves = trainer.get_model_saves(1000)
            >>> saves
            {0: '1', 9: '10', 99: '100', 100: 'decile', 500: 'mid', 999: 'end'}
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

    def plot_training_metrics(self):
        """
        Creates and saves a comprehensive training metrics visualization.

        This method generates a multi-panel plot showing four key training
        metrics over episodes: scores (with rolling max/min), steps per
        episode, snake lengths, and length-to-board-size ratios. The
        visualization is saved as a PNG file in the model folder path for
        analysis and monitoring of training progress.

        Args:
            None

        Returns:
            None: Saves the plot to "{model_folder_path}/training_metrics.png".

        Example:
            >>> trainer = QLearningTrainer()
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
        Evaluates the trained Q-learning agent's performance using
        a saved model.

        This method loads a previously trained Q-table from disk and
        runs the agent in evaluation mode (greedy policy with no
        exploration) for a specified number of episodes. It collects
        performance metrics including scores, snake lengths, steps taken,
        and board utilization ratios. The evaluation can optionally display
        the game visually for each episode.

        Args:
            model_path (str): Path to the saved Q-table pickle file to load for
                evaluation.
            num_episodes (int, optional): Number of evaluation episodes to run.
                Defaults to 10.

        Returns:
            None: Prints evaluation results to console but doesn't return
                values.

        Example:
            >>> trainer = QLearningTrainer()
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

        print("\n--- Q-learning Evaluation Summary ---"
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
        and retrieved from the Q-table data structure.

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
        if isinstance(state, torch.Tensor):
            return tuple(state.detach().numpy().flatten())
        else:
            return tuple(state.flatten())

    def _save_q_table(self, path):
        """
        Saves the Q-table to a file using pickle serialization.

        This method creates the necessary directory structure if it doesn't
        exist and serializes the Q-table dictionary to a binary file for
        later loading and evaluation.

        Args:
            path (str): The file path where the Q-table should be saved.

        Returns:
            None

        Example:
            >>> trainer._save_q_table('models/q_table.pkl')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def _load_q_table(self, path):
        """
        Loads a Q-table from a file using pickle deserialization.

        This method reads a previously saved Q-table from a binary file
        and assigns it to the trainer instance for use in evaluation or
        continued training.

        Args:
            path (str): The file path from which to load the Q-table.

        Returns:
            None: Updates self.q_table attribute.

        Example:
            >>> trainer._load_q_table('models/q_table.pkl')
        """
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)
