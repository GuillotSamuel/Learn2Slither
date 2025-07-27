# src/trainer.py
from .deep_learning_trainer import DeepQLearningTrainer
from .q_learning_trainer import QLearningTrainer
from .q_learning_multithreaded_trainer import QLearningMultiThreadedTrainer


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
        Initializes a reinforcement learning trainer that delegates to specialized trainers.

        Creates a new Trainer instance that acts as a factory and delegates training
        to the appropriate specialized trainer based on the training method.

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
        self.trainer_params = {
            'display_training': display_training,
            'display_evaluation': display_evaluation,
            'board_size': board_size,
            'display_speed': display_speed,
            'ultra_rewards': ultra_rewards,
            'render_mode': render_mode,
            'num_episodes': num_episodes,
            'model_folder_path': model_folder_path,
            'episode_logs': episode_logs
        }
        
        # Initialize the appropriate specialized trainer
        self._create_specialized_trainer()

    def _create_specialized_trainer(self):
        """
        Creates and initializes the appropriate specialized trainer.

        This method instantiates a trainer object based on the training_method
        attribute. It supports three types: deep Q-learning using neural networks,
        traditional Q-learning with tabular methods, and multithreaded Q-learning
        for improved performance on multi-core systems.

        Args:
            None

        Returns:
            None: Sets self.specialized_trainer attribute.

        Raises:
            ValueError: If the training_method is not recognized.

        Example:
            >>> trainer.training_method = 'deep_q_learning'
            >>> trainer._create_specialized_trainer()
        """
        if self.training_method == 'deep_q_learning':
            self.specialized_trainer = DeepQLearningTrainer(**self.trainer_params)
        elif self.training_method == 'q_learning':
            self.specialized_trainer = QLearningTrainer(**self.trainer_params)
        elif self.training_method == 'q_learning_multithreaded':
            self.specialized_trainer = QLearningMultiThreadedTrainer(**self.trainer_params)
        else:
            raise ValueError(f"Unknown training method: {self.training_method}")

    def train(self, num_episodes):
        """
        Trains the reinforcement learning model using the specified training method.

        Delegates to the appropriate specialized trainer based on the training_method
        parameter. Supports deep Q-learning, traditional Q-learning, and multithreaded Q-learning.

        Args:
            num_episodes (int): The total number of training episodes to run.

        Returns:
            None

        Example:
            >>> trainer = Trainer(training_method='q_learning')
            >>> trainer.train(1000)
        """
        return self.specialized_trainer.train(num_episodes)
        
    def evaluate(self, model_path, num_episodes=10):
        """
        Evaluates the performance of a trained model using the appropriate method.

        Delegates to the appropriate specialized trainer for evaluation.

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
        return self.specialized_trainer.evaluate(model_path, num_episodes)

    def __getattr__(self, name):
        """
        Delegates attribute access to the specialized trainer instance.

        This method enables transparent access to methods and attributes of the
        specialized trainer through the main Trainer instance. It provides
        backward compatibility and allows the Trainer to act as a proxy for
        the underlying specialized trainer functionality.

        Args:
            name (str): The name of the attribute or method being accessed.

        Returns:
            object: The corresponding attribute or method from the specialized
            trainer instance.

        Raises:
            AttributeError: If the attribute doesn't exist on the specialized
            trainer.

        Example:
            >>> trainer = Trainer(training_method='q_learning')
            >>> trainer.some_method()  # Calls specialized_trainer.some_method()
        """
        if hasattr(self, 'specialized_trainer'):
            return getattr(self.specialized_trainer, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
