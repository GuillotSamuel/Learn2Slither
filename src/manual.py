# src/manual.py
import sys
import tty
import termios
import threading

from board import Board
from state import get_state_with_display


arrow_keys = {
    '\x1b[A': 'UP',
    '\x1b[B': 'DOWN',
    '\x1b[C': 'RIGHT',
    '\x1b[D': 'LEFT'
}


def get_key():
    """
    Reads a single keypress from the user, including special keys.

    This function reads raw input from the terminal without requiring the
    Enter key. It captures single characters and interprets escape sequences
    for special keys such as arrow keys. The terminal settings are temporarily
    modified to raw mode for this purpose and restored afterward.

    Args:
        None

    Returns:
        str: The key pressed by the user. This may be a single character or an
        escape sequence representing special keys (e.g., arrow keys).

    Example:
        >>> key = get_key()
        >>> if key == '\x1b[A':  # Up arrow key
        ...     print("Up arrow pressed")
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch1 = sys.stdin.read(1)
        if ch1 == '\x1b':
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                return ch1 + ch2 + ch3
            return ch1 + ch2
        return ch1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def manual_mode(render_mode,
                board_size=10):
    """
    Runs the game in manual mode, allowing the user to control
    the snake via arrow keys.

    This function initializes the game board with the specified
    render mode and starts a separate thread to render the game
    window. It captures user key presses to control the snake's
    movement in real-time, updating the game state and rendering
    output in the shell. The game continues until the user quits
    by pressing 'q' or the game ends.

    Args:
        render_mode (str): The mode used to render the game board,
        affecting the visual output.

    Returns:
        None

    Example:
        >>> manual_mode("ascii")
        Use arrow keys to play...
        (Displays game state and updates as user presses arrow keys)
    """
    game = Board(render_mode=render_mode,
                 board_size=board_size)
    game.render_in_shell()
    window_thread = threading.Thread(target=game.render_in_window, daemon=True)
    window_thread.start()
    print("Use arrow keys to play...\n")
    _, visual_representation = get_state_with_display(game.board,
                                                      game.snake,
                                                      game.direction)
    print(visual_representation)
    try:
        while not game.done:
            key = get_key()

            if key == 'q':
                print("Exiting the game...")
                game.done = True
                break

            action = arrow_keys.get(key)
            if action:
                state, reward, done = game.step(action)
                game.render_in_shell()
                print(f"Reward: {reward}, Done: {done}\n")

                _, visual_representation = get_state_with_display(state,
                                                                  game.snake,
                                                                  action)
                print(visual_representation)
            else:
                print("Invalid key! Use arrow keys only.")

    finally:
        game.done = True
        window_thread.join()
        print("Game Over! Your snake length was:", len(game.snake))
