import random
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import threading
import time
import matplotlib.pyplot as plt

from board import Board
from state import get_state_full_map
from trainers.mlp_full_map import MLP
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
import threading

GAMMA = 0.99
LEARNING_RATE = 0.001


class DeepQLearningTrainerFullMap:
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
        Initializes a Deep Q-Learning trainer for reinforcement learning.

        Args:
            display_training (bool): Whether to visually display the
            training process.
            display_evaluation (bool): Whether to visually display the
            evaluation process.
            board_size (int): Size of the game board. If 0, uses random
            sizes between 5-20.
            display_speed (float): Speed factor controlling display update
            rate in seconds.
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
        
        # Calculate input size for the neural network
        # For full map: board_size² + 4 (direction encoding)
        if board_size == 0:
            # For variable board sizes, use a reasonable default
            input_size = 15 * 15 + 4  # 229 features
        else:
            input_size = board_size * board_size + 4
            
        self.model = MLP(input_size=input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.epsilon = self.calculate_epsilon()
        self.epsilon_min = 0.01
        self.epsilon_max = self.epsilon
        self.k = self.calculate_k(num_episodes)
        self.display_speed = display_speed
        self.render_mode = render_mode
        self.start_time = time.time()
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

        This method creates a default folder path for saving trained
        models when no custom path is provided. The path includes
        descriptive suffixes based on the board size configuration to
        distinguish between different model types.

        Args:
            model_folder_path (str or None): Custom path for model storage.
                If None, generates a default path.

        Returns:
            str: The generated or provided model folder path.

        Example:
            >>> trainer.board_size = 10
            >>> trainer._generate_model_folder_path(None)
            'models_eval_deep_q'
        """
        if model_folder_path is None:
            if self.board_size == 0:
                size_suffix = "random_map_deep_q_full_map"
            else:
                size_suffix = "eval_deep_q_full_map"
            return f"models_{size_suffix}"
        else:
            return model_folder_path

    def calculate_k(self, num_episodes):
        """
        Calculates the decay constant k for epsilon-greedy policy.

        This method computes the decay rate parameter used in the
        exponential decay formula for epsilon in the epsilon-greedy
        exploration strategy. The decay constant is determined based on
        the target epsilon value and the total number of training episodes.

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

    def save_model(self, path):
        """
        Saves the current model's state dictionary to a file.

        This method creates the necessary directory structure if it doesn't
        exist and saves the neural network's learned parameters to the
        specified path using PyTorch's serialization format.

        Args:
            path (str): The file path where the model should be saved.

        Returns:
            None

        Example:
            >>> trainer.save_model('models/trained_model.pth')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Loads a saved model's state dictionary from a file.

        This method loads the neural network's learned parameters from the
        specified path and sets the model to evaluation mode for inference.
        The model must have the same architecture as when it was saved.

        Args:
            path (str): The file path from which to load the model.

        Returns:
            None

        Example:
            >>> trainer.load_model('models/trained_model.pth')
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def train(self, num_episodes):
        """
        Trains the reinforcement learning model using Deep Q-Learning.

        This method implements the Deep Q-Learning algorithm using a neural
        network to approximate Q-values. It trains the model through episodes
        of gameplay, using epsilon-greedy exploration and experience replay
        through immediate Q-value updates. The training includes epsilon decay,
        model checkpointing, and comprehensive metrics collection.

        Args:
            num_episodes (int): Total number of training episodes to run.

        Returns:
            None: Updates the neural network model and saves training metrics.

        Example:
            >>> trainer = DeepQLearningTrainer(board_size=10)
            >>> trainer.train(5000)
            Episode 1/5000, Episode time: 0.25s, Total Reward: -15.50...
            Training completed. Total time: 892.45s
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
            state = get_state_full_map(board.board,
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
                next_state = get_state_full_map(board.board,
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
                      f"Episode time: "
                      f"{time.time() - episode_start_time:.2f}s, "
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
        Calculates episode indices for visual display during training.

        This method determines which episodes should be visually displayed
        by combining decile milestones (10%, 20%, etc.) with the last 10
        episodes of training. This provides good coverage of the training
        progress without overwhelming performance.

        Args:
            num_episodes (int): Total number of training episodes.

        Returns:
            set[int]: Set of episode indices to display visually.

        Example:
            >>> trainer.get_display_episodes(1000)
            {100, 200, 300, 400, 500, 600, 700, 800, 900, 990, 991, ..., 999}
        """
        deciles = {int(num_episodes * i / 10) for i in range(1, 10)}
        last_episodes = set(range(num_episodes - 10, num_episodes))
        return deciles.union(last_episodes)

    def get_model_saves(self, num_episodes):
        """
        Generates a mapping of episode indices to model save labels.

        This method creates a dictionary that specifies which episodes should
        trigger model saves and what labels to use for the saved files.
        It includes early episodes, milestone episodes, and the final model
        for comprehensive progress tracking.

        Args:
            num_episodes (int): Total number of training episodes.

        Returns:
            dict[int, str]: Dictionary mapping episode indices to save
            file labels.

        Example:
            >>> trainer.get_model_saves(1000)
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

    def select_action(self, q_values):
        """
        Selects an action using epsilon-greedy exploration policy.

        This method implements the epsilon-greedy action selection strategy
        where with probability epsilon a random action is chosen (exploration),
        and with probability (1-epsilon) the action with the highest Q-value
        is selected (exploitation).

        Args:
            q_values (torch.Tensor): Q-values for all possible actions.

        Returns:
            int: The selected action index (0, 1, or 2).

        Example:
            >>> q_values = torch.tensor([0.1, 0.8, 0.3])
            >>> action = trainer.select_action(q_values)
            >>> action in [0, 1, 2]
            True
        """
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        else:
            return torch.argmax(q_values).item()

    def plot_training_metrics(self):
        """
        Creates and saves comprehensive training metrics visualization.

        This method generates a multi-panel plot showing training progress
        across four key metrics: scores (rewards), snake lengths, length
        ratios, and number of moves per episode. The visualization is saved
        as a PNG file in the model folder for later analysis.

        Args:
            None

        Returns:
            None: Saves plot to '{model_folder_path}/training_metrics.png'.

        Example:
            >>> trainer.plot_training_metrics()
            # Creates and saves training_metrics.png with 4 subplots
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
        Evaluates the performance of a trained deep Q-learning model.

        This method loads a previously trained neural network model and runs
        it in evaluation mode (no exploration, greedy policy only) for a
        specified number of episodes. It collects comprehensive performance
        metrics including scores, snake lengths, steps taken, and board
        utilization ratios. Visual display is optional during evaluation.

        Args:
            model_path (str): Path to the saved PyTorch model file (.pth)
                to load for evaluation.
            num_episodes (int, optional): Number of evaluation episodes to run.
                Defaults to 10.

        Returns:
            None: Prints evaluation results to console but doesn't return
            values.

        Example:
            >>> trainer = DeepQLearningTrainer()
            >>> trainer.evaluate("models/end_model.pkl", num_episodes=5)
            [Evaluation] Episode 1/5 - Score: 45.0, Length: 8, Steps: 127...
            --- Deep Q-Learning Evaluation Summary ---
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
            state = get_state_full_map(board.board,
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
                state = get_state_full_map(board.board,
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

            print(f"[Evaluation] Episode {episode + 1}/{num_episodes} - "
                  f"Score: {total_reward}, "
                  f"Length: {len(board.snake)}, "
                  f"Steps: {steps}, "
                  f"Ratio: {ratio:.4f}, "
                  f"Board Size: {current_board_size}")

        print("\n--- Deep Q-Learning Evaluation Summary ---"
              f"Max Length: {np.max(total_lengths)} "
              f"Average Length: {np.mean(total_lengths):.2f} "
              f"Average Steps: {np.mean(total_steps):.2f} "
              f"Average Ratio: {np.mean(total_ratios) * 100:.2f}%")
