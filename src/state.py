# src/state.py
import numpy as np


def ray(board, start_x, start_y, dx, dy):
    """
    Traces a ray on the board from a starting position in a
    specified direction until a wall is encountered.

    This function moves step-by-step from the starting coordinates
    (start_x, start_y) along the direction vector (dx, dy),
    collecting information about each cell encountered. It records
    the content of each cell and the distance from the start. The
    ray stops when it hits a wall or moves outside the board
    boundaries.

    Args:
        board (list[list[str]]): 2D grid representing the game board.
        start_x (int): The starting x-coordinate on the board.
        start_y (int): The starting y-coordinate on the board.
        dx (int): The step direction in the x-axis (-1, 0, or 1).
        dy (int): The step direction in the y-axis (-1, 0, or 1).

    Returns:
        list[tuple[str, int]]: A list of tuples, each containing the
        cell content ('W' for wall or other) and the distance from the
        start position along the ray.

    Example:
        >>> vision = ray(board, 5, 5, 0, 1)  # Ray cast downwards from (5,5)
        >>> print(vision)
        [(' ', 1), (' ', 2), ('W', 3)]
    """
    x, y = start_x, start_y
    vision = []

    while True:
        x += dx
        y += dy

        if x < 0 or x >= len(board) or y < 0 or y >= len(board[0]):
            vision.append(('W', (x - dx - start_x)
                           * dx + (y - dy - start_y) * dy))
            break

        cell = board[x][y]
        vision.append((cell, (x - start_x)
                       * dx + (y - start_y) * dy))

        if cell == 'W':
            break

    return vision


def encode_vision(vision_ray):
    """
    Encodes a single vision ray by recording the normalized distance to
    the first occurrence of each cell type.

    This function processes a list of tuples representing a vision ray,
    where each tuple contains a cell type and its distance from the origin.
    It retains only the first occurrence of each specified cell type
    ('W', 'S', 'A', 'R'), normalizing the distance by a maximum distance
    threshold to produce a value between 0 and 1.

    Args:
        vision_ray (list[tuple[str, int]]): A list of (cell_type, distance)
        tuples representing the contents and distances observed along a vision
        ray.

    Returns:
        list[float]: A list of normalized distances for each cell type
        ['W', 'S', 'A', 'R'], where 1.0 indicates absence within the
        max distance.

    Example:
        >>> vision_ray = [(' ', 1), ('S', 3), ('W', 5), ('A', 7)]
        >>> encode_vision(vision_ray)
        [0.5, 0.3, 0.7, 1.0]
    """
    types = ['W', 'S', 'A', 'R']
    distances = {t: 1.0 for t in types}
    max_distance = 10

    for cell_type, dist in vision_ray:
        if cell_type in types and distances[cell_type] == 1.0:
            distances[cell_type] = min(dist / max_distance, 1.0)

    ray_encoded = [distances[t] for t in types]

    return ray_encoded


def get_ray(board, head_x, head_y, direction):
    """
    Generates vision rays from the snake's head in four
    relative directions based on its current facing.

    Given the current direction of the snake, this function computes
    the ray traces for the front, left, back, and right directions
    relative to the snake’s orientation. It uses the `ray` function
    to obtain the contents and distances along each direction.

    Args:
        board (list[list[str]]): 2D grid representing the game board.
        head_x (int): The x-coordinate of the snake's head.
        head_y (int): The y-coordinate of the snake's head.
        direction (str): The current facing direction of the snake.
        One of 'UP', 'LEFT', 'DOWN', 'RIGHT'.

    Returns:
        dict[str, list[tuple[str, int]]]: A dictionary mapping
        'front', 'left', 'back', and 'right' to their corresponding
        vision rays (lists of (cell_type, distance) tuples).

    Example:
        >>> rays = get_ray(board, 5, 5, 'UP')
        >>> print(rays['front'])
        [(' ', 1), ('S', 2), ('W', 3)]
    """
    directions = {
        'UP':    [(-1, 0), (0, -1), (1, 0), (0, 1)],
        'LEFT':  [(0, -1), (1, 0), (0, 1), (-1, 0)],
        'DOWN':  [(1, 0), (0, 1), (-1, 0), (0, -1)],
        'RIGHT': [(0, 1), (-1, 0), (0, -1), (1, 0)],
    }

    front, left, back, right = directions[direction]

    return {
        'front': ray(board, head_x, head_y, *front),
        'left': ray(board, head_x, head_y, *left),
        'back': ray(board, head_x, head_y, *back),
        'right': ray(board, head_x, head_y, *right),
    }


def get_state(board, snake, direction):
    """
    Generates a hashable state representation of the game based
    on the snake's position, direction, and board vision.

    This function computes vision rays from the snake's head in the
    four relative directions, encodes each ray to normalized distance
    vectors, and concatenates them into a single state vector suitable
    for machine learning or game state tracking.

    Args:
        board (list[list[str]]): 2D grid representing the game board.
        snake (list[tuple[int, int]]): List of (x, y) coordinates
        representing the snake's body, head first.
        direction (str): Current direction the snake is facing
        ('UP', 'LEFT', 'DOWN', 'RIGHT').

    Returns:
        numpy.ndarray: A column vector (shape: [N, 1]) representing
        the encoded state.

    Example:
        >>> state = get_state(board, snake, 'RIGHT')
        >>> print(state.shape)
        (16, 1)
    """
    head_x, head_y = snake[0]

    rays = get_ray(board, head_x, head_y, direction)

    state_arrays = tuple(encode_vision(rays[key])
                         for key in ['front', 'left', 'right', 'back'])
    state_vector = np.concatenate(state_arrays).reshape(-1, 1)
    return state_vector


def colorize(cell):
    """
    Returns a colorized string representation of a board cell for
    terminal display.

    Maps specific cell types to ANSI color codes to enhance visual
    distinction when printing the game board in the terminal. Unknown
    cell types are returned without colorization.

    Args:
        cell (str): A string representing the cell content; typically
        a single character.

    Returns:
        str: The colorized string with ANSI escape codes for terminal
        output.

    Example:
        >>> print(colorize('S'))
        '\033[92mS\033[0m'
    """
    color_map = {
        '0': '\033[90m.',
        'W': '\033[90mW',
        'S': '\033[92mS',
        'A': '\033[94mA',
        'R': '\033[91mR',
    }
    reset = '\033[0m'
    return color_map.get(cell[0], cell[0]) + reset


def generate_visual(board, snake, direction):
    """
    Generates a formatted string visualizing the snake’s ray-based vision
    of its surroundings.

    This function creates a multi-line string representing what the snake
    "sees" along its front, left, right, and back directions using colored
    symbols. The snake's head is marked in the center, with rays displayed
    relative to it, providing an intuitive visualization for debugging or
    display in a terminal.

    Args:
        board (list[list[str]]): 2D grid representing the game board.
        snake (list[tuple[int, int]]): List of (x, y) coordinates
        representing the snake’s body, head first.
        direction (str): Current direction the snake is facing
        ('UP', 'LEFT', 'DOWN', 'RIGHT').

    Returns:
        str: A multi-line string visualizing the snake'ssurroundings with
        colored cells.

    Example:
        >>> print(generate_visual(board, snake, 'UP'))
        W
        S
        LHSR
        A
        W
    """
    head_x, head_y = snake[0]

    rays = get_ray(board, head_x, head_y, direction)

    # max_height = max(len(rays['front']), len(rays['back']))
    output = []

    for i in range(len(rays['front']) - 1, -1, -1):
        line = ' ' * len(rays['left']) + colorize(rays['front'][i])
        output.append(line)

    middle_line = (
        ''.join(colorize(c) for c in reversed(rays['left']))
        + '\033[93mH'
        + '\033[0m'
        + ''.join(colorize(c) for c in rays['right'])
    )
    output.append(middle_line)

    for i in range(len(rays['back'])):
        line = ' ' * len(rays['left']) + colorize(rays['back'][i])
        output.append(line)

    return '\n'.join(output)


def get_state_with_display(board, snake, direction):
    """
    Computes the Q-learning state vector and generates a visual representation
    of the snake’s environment.

    This function combines the numeric state encoding used for learning with
    a human-readable colored visual display of the snake’s surroundings,
    facilitating both algorithmic processing and debugging or visualization.

    Args:
        board (list[list[str]]): 2D grid representing the game board.
        snake (list[tuple[int, int]]): List of (x, y) coordinates representing
        the snake’s body, head first.
        direction (str): Current direction the snake is facing
        ('UP', 'LEFT', 'DOWN', 'RIGHT').

    Returns:
        tuple:
            - numpy.ndarray: The encoded state vector for Q-learning.
            - str: The colored multi-line string visualizing the snake’s
                   surroundings.

    Example:
        >>> state, visual = get_state_with_display(board, snake, 'RIGHT')
        >>> print(visual)
    """
    state = get_state(board, snake, direction)
    visual_representation = generate_visual(board, snake, direction)

    return state, visual_representation
