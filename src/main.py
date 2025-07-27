# src/main.py
import argparse
from trainers.trainer import Trainer
from manual import manual_mode
from utils import min_100


DEFAULT_MODEL_PATH = 'models'
DEFAULT_TRAINING_EPISODES = 1000
DEFAULT_EVALUATING_EPISODES = 10


def main():
    """
    Parses command-line arguments and runs the Snake Q-Learning
    program in the specified mode.

    This project implements a Q-Learning based AI to play the classic
    Snake game. The program supports multiple modes: training the AI,
    playing using a trained model, evaluating the model's performance,
    and manual gameplay. The main function handles user input from the
    command line to configure the game environment, training parameters,
    display settings, and model usage.

    Depending on the selected mode, it initializes the appropriate
    components and starts the corresponding process, such as training
    the agent over multiple episodes, evaluating its performance,
    playing games automatically, or allowing manual control.

    Args:
        None

    Returns:
        None

    Example:
        >>> python main.py --mode train --episodes 5000 --display
        # Starts training for 5000 episodes with display enabled
    """
    parser = argparse.ArgumentParser(description='Snake Q-Learning')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train',
                                 'play',
                                 'evaluate',
                                 'manual'],
                        help='Modes: train, play, evaluate, manual'
                        ' (default: train)')
    parser.add_argument('--episodes',
                        type=min_100,
                        default=1000,
                        help='Number of episodes for training'
                        ' or evaluation (default: 1000)')
    parser.add_argument('--model',
                        type=str,
                        default='models/end_model.pkl',
                        help='Path to the model file (used in play and'
                        'evaluate modes) (default: models/end_model.pkl)')
    parser.add_argument('--display',
                        action='store_true',
                        help='Display the game during training'
                        'or evaluation (default: true)')
    parser.add_argument('--board_size',
                        type=int, default=10,
                        help='Size of the game board, if =0 it will chose'
                        ' random board size between 10 and 30 at each '
                        'episode  (default: 10)')
    parser.add_argument('--display_speed',
                        type=float,
                        default=0.1,
                        help='Speed of display in seconds (default: 0.1)')
    parser.add_argument('--load',
                        action='store_true',
                        help='Load the model from the'
                        'specified path (default: true)')
    parser.add_argument('--ultra_rewards',
                        action='store_true',
                        default=True,
                        help='Use ultra rewards (default: true)')
    parser.add_argument('--render_mode',
                        type=str,
                        default='basic',
                        choices=['basic',
                                 'classic'],
                        help='Render modes: basic or classic (default: basic)')
    parser.add_argument('--num_game',
                        type=int,
                        default=3,
                        help="Number of games for play: int (default: 3)")
    parser.add_argument('--training_method',
                        type=str,
                        default='q_learning',
                        choices=['q_learning',
                                 'deep_q_learning',
                                 'q_learning_multithreaded'],
                        help='Training method: q_learning or deep_q_learning'
                        ' (default: q_learning)')
    parser.add_argument('--model_folder_path',
                        type=str,
                        default='models',
                        help='Path to the model folder (default: models)')
    parser.add_argument('--episode_logs',
                        action='store_true',
                        default=False,
                        help='Activate episode logs when training or evaluating (default: False)')
    parser.add_argument('--no_episode_logs',
                        action='store_true',
                        default=False,
                        help='Deactivate episode logs when training or evaluating')

    args = parser.parse_args()

    # Handle episode_logs logic: default False, but can be enabled with --episode_logs
    episode_logs_enabled = args.episode_logs
    if args.no_episode_logs:
        episode_logs_enabled = False

    # python main.py --mode train --episodes 10000 --display
    if args.mode == 'train':
        episodes = args.episodes if args.episodes is not None\
            else DEFAULT_TRAINING_EPISODES
        print(f"episode logs: {episode_logs_enabled}")
        trainer = Trainer(display_training=args.display,
                          board_size=args.board_size,
                          display_speed=args.display_speed,
                          ultra_rewards=args.ultra_rewards,
                          render_mode=args.render_mode,
                          num_episodes=episodes,
                          training_method=args.training_method,
                          model_folder_path=args.model_folder_path,
                          episode_logs=episode_logs_enabled)
        trainer.train(episodes)
    # python main.py --mode evaluate --episodes 10 --model models/end_model.pkl
    elif args.mode == 'evaluate':
        model_path = args.model if args.model else DEFAULT_MODEL_PATH
        episodes = args.episodes if args.episodes is not None\
            else DEFAULT_EVALUATING_EPISODES
        trainer = Trainer(display_evaluation=args.display,
                          board_size=args.board_size,
                          display_speed=args.display_speed,
                          render_mode=args.render_mode,
                          training_method=args.training_method,
                          model_folder_path=args.model_folder_path,
                          episode_logs=episode_logs_enabled)
        trainer.evaluate(model_path, episodes)
    # python main.py --mode play --model models/end_model.pkl
    elif args.mode == 'play':
        model_path = args.model if args.model else DEFAULT_MODEL_PATH
        trainer = Trainer(display_evaluation=True,
                          board_size=args.board_size,
                          display_speed=args.display_speed,
                          render_mode=args.render_mode,
                          training_method=args.training_method,
                          model_folder_path=args.model_folder_path,
                          episode_logs=episode_logs_enabled)
        trainer.evaluate(model_path, num_episodes=args.num_game)
    # python main.py --mode manual
    elif args.mode == 'manual':
        manual_mode(render_mode=args.render_mode,
                    board_size=args.board_size)

    else:
        print("Invalid mode. Use instead 'train', 'play', 'evaluate', 'manual'.")


if __name__ == "__main__":
    main()
