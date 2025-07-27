# src/board.py
import os
import random
import pygame

DEAD_REWARD = -150
VICTORY_REWARD = 100
GREEN_APPLE_REWARD = 20
RED_APPLE_REWARD = -20
SNAKE_MOVE_REWARD = -0.01
SNAKE_STAGNATION_REWARD = -1
SNAKE_WRAPPPING_REWARD = -5


class Board:
    def __init__(self, board_size=10,
                 ultra_rewards=False,
                 render_mode='basic',
                 episodes_logs=True):
        """
        Initializes the game board and its elements for the snake game.

        This constructor sets up the initial state of the game,
        including the snake's position, direction, reward system,
        apple placements, and board size. It also configures
        rendering options and maximum allowed steps. The snake
        and apples are spawned, and the game board is initialized
        accordingly.

        Args:
            board_size (int, optional): The size of the square game
            board. Defaults to 10.
            ultra_rewards (bool, optional): Whether to enable enhanced
            reward rules. Defaults to False.
            render_mode (str, optional): The mode used for rendering
            the game visuals. Defaults to 'basic'.

        Returns:
            None

        Example:
            >>> game = Board(board_size=15,
                             ultra_rewards=True,
                             render_mode='fancy')
        """
        self.snake = []
        self.direction = None
        self.done = False
        self.reward = 0
        self.ultra_rewards = ultra_rewards
        self.green_apples = []
        self.red_apple = []
        self.board_size = board_size
        self.steps = 0
        self.steps_since_last_green_apple = 0
        self.render_mode = render_mode
        self.max_steps = self.calculate_max_steps()
        self.episodes_logs = episodes_logs

        self.spawn_snake()
        self.spawn_apples()
        self.board = self.init_board()

    def calculate_max_steps(self):
        """
        Calculates the maximum number of steps allowed based on
        the board size.

        This function determines an upper limit on the number
        of steps for a game or simulation, scaling non-linearly
        with the board size. The calculation uses a base step
        count for a 10x10 board and applies quadratic-like
        growth controlled by an exponent alpha.

        Args:
        None

        Returns:
        int: The calculated maximum number of steps allowed for
        the current board size.

        Example:
        >>> game.board_size = 20
        >>> game.calculate_max_steps()
        92830
        """
        base_steps = 10000
        size_ratio = self.board_size / 10
        alpha = 2.2
        max_steps = int(base_steps * (size_ratio ** alpha))
        return max_steps

    def get_free_cells(self):
        """
        Retrieves all unoccupied cells on the game board.

        This function generates a list of coordinates representing
        cells that are not currently occupied by the snake, green apples,
        or the red apple. It iterates over the entire board and excludes
        positions that are in the set of occupied cells.

        Args:
        None

        Returns:
        list[tuple[int, int]]: A list of (x, y) coordinates for all
        free cells on the board.

        Example:
        >>> game.board_size = 5
        >>> game.snake = [(0, 0), (0, 1)]
        >>> game.green_apples = [(2, 2)]
        >>> game.red_apple = [(4, 4)]
        >>> game.get_free_cells()
        [(0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), ...]
        """
        occupied_cells = set(self.snake + self.green_apples + self.red_apple)
        return [(x, y) for x in range(self.board_size)
                for y in range(self.board_size)
                if (x, y) not in occupied_cells]

    def try_place_segment(self, x, y):
        """
        Attempts to place the next snake segment at the
        specified position.

        This function checks whether the given coordinates (x, y)
        are within the board boundaries and not already occupied
        by the snake. If the position is valid, it returns the
        coordinates; otherwise, it returns None.

        Args:
        x (int): The x-coordinate of the desired position.
        y (int): The y-coordinate of the desired position.

        Returns:
        tuple[int, int] | None: The valid position as a tuple
        (x, y) if placement is possible; otherwise, None.

        Example:
        >>> game.board_size = 5
        >>> game.snake = [(2, 2)]
        >>> game.try_place_segment(1, 1)
        (1, 1)
        >>> game.try_place_segment(2, 2)
        None
        """
        if 0 <= x < self.board_size \
            and 0 <= y < self.board_size \
                and (x, y) not in self.snake:
            return (x, y)
        return None

    def spawn_snake(self):
        """
        Spawns a snake of length 3 at a random valid position on the board.

        This function attempts to place a 3-segment snake by trying different
        starting positions and configurations. It supports straight snakes as
        well as snakes with left or right turns. The snake is positioned with
        its head first and the direction is automatically determined based on
        the segment arrangement. If no valid configuration is found, a fallback
        position is used near the board center.

        Args:
            None

        Returns:
            None: Updates self.snake and self.direction attributes in place.

        Example:
            >>> game.board_size = 10
            >>> game.spawn_snake()
            >>> len(game.snake)
            3
            >>> game.direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']
            True
        """
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        dir_vectors = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1),
        }

        def get_direction(snake):
            """Return direction from head to next segment"""
            if len(snake) < 2:
                return 'RIGHT'  # default
            head, neck = snake[0], snake[1]
            dx = head[0] - neck[0]  # difference in row (x)
            dy = head[1] - neck[1]  # difference in column (y)
            
            # The snake moves in the direction FROM neck TO head
            if dx == -1: return 'UP'      # head is above neck
            if dx == 1: return 'DOWN'     # head is below neck  
            if dy == -1: return 'LEFT'    # head is left of neck
            if dy == 1: return 'RIGHT'    # head is right of neck
            return 'RIGHT'

        def add_pos(pos, delta):
            return (pos[0] + delta[0], pos[1] + delta[1])

        positions = [(x, y) for x in range(self.board_size) for y in range(self.board_size)]
        random.shuffle(positions)

        for start in positions:
            for dir1 in directions:
                dx1, dy1 = dir_vectors[dir1]
                second = add_pos(start, (dx1, dy1))

                if not (0 <= second[0] < self.board_size and 0 <= second[1] < self.board_size):
                    continue

                # Try straight snake
                third = add_pos(second, (dx1, dy1))
                if 0 <= third[0] < self.board_size and 0 <= third[1] < self.board_size:
                    segments = [start, second, third]
                    snake = list(reversed(segments))  # head first
                    self.snake = snake
                    self.direction = get_direction(snake)
                    return

                # Right turn
                right_turn = {
                    'UP': 'RIGHT', 'RIGHT': 'DOWN',
                    'DOWN': 'LEFT', 'LEFT': 'UP'
                }
                dir2 = right_turn[dir1]
                dx2, dy2 = dir_vectors[dir2]
                third = add_pos(second, (dx2, dy2))
                if 0 <= third[0] < self.board_size and 0 <= third[1] < self.board_size:
                    segments = [start, second, third]
                    snake = list(reversed(segments))  # head first
                    self.snake = snake
                    self.direction = get_direction(snake)
                    return

                # Left turn
                left_turn = {
                    'UP': 'LEFT', 'LEFT': 'DOWN',
                    'DOWN': 'RIGHT', 'RIGHT': 'UP'
                }
                dir2 = left_turn[dir1]
                dx2, dy2 = dir_vectors[dir2]
                third = add_pos(second, (dx2, dy2))
                if 0 <= third[0] < self.board_size and 0 <= third[1] < self.board_size:
                    segments = [start, second, third]
                    snake = list(reversed(segments))  # head first
                    self.snake = snake
                    self.direction = get_direction(snake)
                    return

        # fallback
        center = self.board_size // 2
        self.snake = [(center, center + 2), (center, center + 1), (center, center)]
        self.direction = 'LEFT'

    def spawn_apples(self):
        """
        Randomly places green and red apples on free cells of the board.

        This function retrieves all free cells (not occupied by the
        snake or other apples), shuffles them, and places up to two
        green apples and one red apple on available positions. If
        there are fewer than three free cells, it places as many
        apples as possible.

        Args:
        None

        Returns:
        None: Updates the positions of green_apples and red_apple
        attributes in place.

        Example:
        >>> game.get_free_cells = lambda: [(1,1), (2,2), (3,3), (4,4)]
        >>> game.spawn_apples()
        >>> len(game.green_apples)
        2
        >>> len(game.red_apple)
        1
        """
        free_cells = self.get_free_cells()
        random.shuffle(free_cells)

        if len(free_cells) >= 3:
            self.green_apples = free_cells[:2]
            self.red_apple = [free_cells[2]]
        elif len(free_cells) >= 2:
            self.green_apples = free_cells[:2]
            self.red_apple = []
        elif len(free_cells) >= 1:
            self.green_apples = free_cells[:1]
            self.red_apple = []
        else:
            self.green_apples = []
            self.red_apple = []

    def init_board(self):
        """
        Initializes the game board with the current snake and apple positions.

        Creates a 2D list representing the board grid, where each cell
        is initialized to '0'. Snake segments are marked with 'S', green
        apples with 'A', and the red apple with 'R'. The function ensures
        all positions are within board boundaries before placing them.

        Args:
        None

        Returns:
        list[list[str]]: A 2D list representing the board state with snake
        and apples.

        Example:
        >>> game.snake = [(1, 1), (1, 2), (1, 3)]
        >>> game.green_apples = [(0, 0), (2, 2)]
        >>> game.red_apple = [(3, 3)]
        >>> board = game.init_board()
        >>> board[1][1]
        'S'
        >>> board[0][0]
        'A'
        >>> board[3][3]
        'R'
        """
        board = [['0' for _ in range(self.board_size)]
                 for _ in range(self.board_size)]

        for x, y in self.snake:
            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                board[x][y] = 'S'

        for x, y in self.green_apples:
            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                board[x][y] = 'A'

        for x, y in self.red_apple:
            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                board[x][y] = 'R'

        return board

    def distance(self, p1, p2):
        """
        Calculates the Euclidean distance between two points on a 2D plane.

        Given two points p1 and p2, each represented as a tuple of (x, y)
        coordinates, this function computes the straight-line distance between
        them using the Pythagorean theorem.

        Args:
        p1 (tuple[float, float]): The first point as (x, y).
        p2 (tuple[float, float]): The second point as (x, y).

        Returns:
        float: The Euclidean distance between the two points.

        Example:
        >>> distance((0, 0), (3, 4))
        5.0
        """
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        """ return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) """

    def is_wrapping_on_itself(self, snake):
        """
        Determines if the snake is wrapping on itself based on proximity
        of segments.

        This function checks how many snake body segments (excluding the
        head) are within a distance less than 2 units from the head. If
        more than three segments are that close, it indicates the snake
        is wrapping onto itself.

        Args:
        snake (list[tuple[int, int]]): List of (x, y) coordinates
        representing thesnake's body, with the head as the first element.

        Returns:
        bool: True if the snake is wrapping on itself, False otherwise.

        Example:
        >>> snake = [(5, 5), (5, 6), (6, 5), (6, 6), (4, 5)]
        >>> is_wrapping_on_itself(snake)
        True
        """
        head = snake[0]
        nearby_segments = sum(1 for segment in snake[1:]
                              if self.distance(segment, head) < 2)
        return nearby_segments > 3

    def step(self, action):
        """
        Advance the game by moving the snake in the specified direction and
        updating the game state.

        This method updates the snake's direction based on the given action
        relative to its current heading. It calculates the new head position,
        checks for collisions with walls or itself, and handles interactions
        with green and red apples. Rewards are assigned accordingly, and the
        game can end due to death, victory, or reaching the maximum allowed
        steps. The game board is updated after each step.

        Args:
            action (str): The action to take relative to the current direction.
            Expected values are 'FORWARD', 'LEFT', or 'RIGHT'.

        Returns:
            tuple: A tuple containing:
                - board (list[list[int]]): The updated game board state.
                - reward (int): The reward earned from the current step.
                - done (bool): Whether the game has ended.

        Example:
            >>> board, reward, done = game.step('LEFT')
            >>> print(done)
            False
        """
        if self.done:
            return self.board, self.reward, self.done

        sa = {'UP': {'FORWARD': 'UP',
                     'LEFT': 'LEFT',
                     'RIGHT': 'RIGHT',
                     'DOWN': 'DOWN'},
              'DOWN': {'FORWARD': 'DOWN',
                       'LEFT': 'RIGHT',
                       'RIGHT': 'LEFT',
                       'DOWN': 'UP'},
              'LEFT': {'FORWARD': 'LEFT',
                       'LEFT': 'DOWN',
                       'RIGHT': 'UP',
                       'DOWN': 'RIGHT'},
              'RIGHT': {'FORWARD': 'RIGHT',
                        'LEFT': 'UP',
                        'RIGHT': 'DOWN',
                        'DOWN': 'LEFT'}}

        self.direction = sa[self.direction].get(action, self.direction)

        head_x, head_y = self.snake[0]
        if self.direction == 'UP':
            new_head = (head_x - 1, head_y)
        elif self.direction == 'DOWN':
            new_head = (head_x + 1, head_y)
        elif self.direction == 'LEFT':
            new_head = (head_x, head_y - 1)
        elif self.direction == 'RIGHT':
            new_head = (head_x, head_y + 1)

        x, y = new_head
        if x < 0 \
            or x >= self.board_size \
                or y < 0 or y >= self.board_size \
                or (new_head in self.snake and new_head != self.snake[-1]):
            self.done = True
            self.reward = DEAD_REWARD
            return self.board, self.reward, self.done

        if new_head in self.green_apples:
            self.snake.insert(0, new_head)
            self.green_apples.remove(new_head)
            self.steps_since_last_green_apple = 0
            self.reward = GREEN_APPLE_REWARD
            free_cells = self.get_free_cells()
            if free_cells:
                self.green_apples.append(random.choice(free_cells))
        elif new_head in self.red_apple:
            if len(self.snake) == 1:
                self.done = True
                self.reward = DEAD_REWARD
                self.snake.insert(0, new_head)
                self.snake.pop()
                return self.board, self.reward, self.done

            self.snake.insert(0, new_head)
            self.snake.pop()

            if len(self.snake) > 1:
                self.snake.pop()

            self.red_apple.clear()
            free_cells = self.get_free_cells()
            if free_cells:
                self.red_apple.append(random.choice(free_cells))
            self.reward = RED_APPLE_REWARD
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            self.reward = SNAKE_MOVE_REWARD

        self.steps += 1

        if self.steps >= self.max_steps:
            self.done = True
            self.reward = DEAD_REWARD
            if self.episodes_logs == True:
                print(f"Game Over! Max steps reached. (maximum: {self.max_steps})")

        if self.ultra_rewards:
            if self.steps_since_last_green_apple >= 20:
                self.reward += SNAKE_STAGNATION_REWARD
            if self.is_wrapping_on_itself(self.snake):
                self.reward += SNAKE_WRAPPPING_REWARD

        if len(self.snake) == self.board_size * self.board_size:
            self.done = True
            self.reward = VICTORY_REWARD
            print("Game Won!")

        self.board = self.init_board()
        
        return self.board, self.reward, self.done

    def render_in_shell(self):
        """
        Display the current game board in the terminal with color-coded cells.

        This method clears the terminal screen and prints the game board,
        using ANSI escape codes to color different cell types for better
        visual distinction:
        - '.' for empty cells (gray)
        - 'S' for the snake (green)
        - 'A' for green apples (blue)
        - 'R' for red apples (red)

        Returns:
            None

        Example:
            >>> game.render_in_shell()
            (Displays the colored game board in the terminal)
        """
        os.system("clear")
        color_map = {
            '0': '\033[90m.',
            'S': '\033[92mS',
            'A': '\033[94mA',
            'R': '\033[91mR'
        }
        reset = '\033[0m'

        for row in self.board:
            line = ' '.join(color_map[cell] + reset for cell in row)
            print(line)
        print()

    def render_in_window(self):
        """
        Render the snake game in a Pygame window.

        Supports two render modes:
        - "basic": Simple colored rectangles for snake and apples.
        - "classic": Uses detailed sprite assets for snake parts and apples,
        with a grass-textured background.

        The method initializes Pygame, creates a window sized to the board,
        draws the game state each frame with the appropriate visuals,
        and handles the main event loop until the game ends or window is
        closed.

        Snake rendering in "classic" mode handles directional sprites for
        head, tail, body segments, and corners based on snake position.

        No return value; runs until self.done is True or window is closed.
        """
        render_mode = self.render_mode
        print(f"Rendering in {render_mode} mode...")
        pygame.init()
        cell_size = 30
        w, h = self.board_size * cell_size, self.board_size * cell_size + 40
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Snake Game")

        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 30)

        # Background colors for grass effect
        grass_colors = [
            [
                (0, random.randint(100, 120), 0)
                for _ in range(self.board_size)
            ]
            for _ in range(self.board_size)
        ]

        # Load assets if in classic mode
        assets_path = os.path.join("src", "assets", "classic")
        if render_mode == "classic":
            # Load all images
            def load_img(name):
                return pygame.image.load(os.path.join(assets_path, name))

            def resize(img):
                return pygame.transform.scale(img, (cell_size, cell_size))

            img_head_up = resize(load_img("head_up.png"))
            img_head_down = resize(load_img("head_down.png"))
            img_head_left = resize(load_img("head_left.png"))
            img_head_right = resize(load_img("head_right.png"))

            img_tail_up = resize(load_img("tail_down.png"))
            img_tail_down = resize(load_img("tail_up.png"))
            img_tail_left = resize(load_img("tail_right.png"))
            img_tail_right = resize(load_img("tail_left.png"))

            img_body_horizontal = resize(load_img("body_horizontal.png"))
            img_body_vertical = resize(load_img("body_vertical.png"))
            img_body_topleft = resize(load_img("body_topleft.png"))
            img_body_topright = resize(load_img("body_topright.png"))
            img_body_bottomleft = resize(load_img("body_bottomleft.png"))
            img_body_bottomright = resize(load_img("body_bottomright.png"))

            img_red_apple = resize(load_img("red_apple.png"))
            img_green_apple = resize(load_img("green_apple.png"))

        def get_snake_part_image(snake_positions, index):
            """
            Determines the appropriate snake part image based on the
            segment's position and its neighbors.

            This function selects the correct image for a snake segment
            depending on whether it is the head, tail, or a body part. For
            the head and tail, the direction is inferred from the adjacent
            segment's position. For body segments, the function determines
            if the segment is part of a straight line (horizontal or vertical)
            or a corner, choosing the corresponding image accordingly.

            Args:
                snake_positions (list[tuple[int, int]]): A list of
                (row, column) tuples representing the positions of the
                    snake segments in order from head to tail.
                index (int): The index of the current snake segment
                to get the image for.

            Returns:
                pygame.Surface: The image corresponding to the snake
                segment's correct orientation.

            Example:
                >>> snake = [(5, 5), (5, 4), (5, 3)]
                >>> img = get_snake_part_image(snake, 0)
                # Returns head image facing left
            """
            if len(snake_positions) == 1:
                return img_head_up  # Single segment snake

            current = snake_positions[index]

            # HEAD - positions are (row, col)
            if index == 0:
                next_pos = snake_positions[1]
                dx = current[1] - next_pos[1]
                dy = current[0] - next_pos[0]

                if dx == 1:
                    return img_head_right
                if dx == -1:
                    return img_head_left
                if dy == 1:
                    return img_head_down
                if dy == -1:
                    return img_head_up
                return img_head_up

            # TAIL
            if index == len(snake_positions) - 1:
                prev_pos = snake_positions[index - 1]
                dx = prev_pos[1] - current[1]
                dy = prev_pos[0] - current[0]

                if dx == 1:
                    return img_tail_right
                if dx == -1:
                    return img_tail_left
                if dy == 1:
                    return img_tail_down
                if dy == -1:
                    return img_tail_up
                return img_tail_up

            prev_pos = snake_positions[index - 1]
            next_pos = snake_positions[index + 1]

            prev_dx = current[1] - prev_pos[1]
            prev_dy = current[0] - prev_pos[0]
            next_dx = next_pos[1] - current[1]
            next_dy = next_pos[0] - current[0]

            if prev_dx != 0 and next_dx != 0:
                return img_body_horizontal
            if prev_dy != 0 and next_dy != 0:
                return img_body_vertical

            if (
                (prev_dx == 1 and next_dy == -1)
                or (prev_dy == 1 and next_dx == -1)
            ):
                return img_body_topleft

            if (
                (prev_dx == -1 and next_dy == -1)
                or (prev_dy == 1 and next_dx == 1)
            ):
                return img_body_topright

            if (
                (prev_dx == 1 and next_dy == 1)
                or (prev_dy == -1 and next_dx == -1)
            ):
                return img_body_bottomleft

            if (
                (prev_dx == -1 and next_dy == 1)
                or (prev_dy == -1 and next_dx == 1)
            ):
                return img_body_bottomright

            return img_body_horizontal  # fallback

        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True

            # Draw background
            screen.fill((0, 0, 0))
            if render_mode == "classic":
                for y in range(self.board_size):
                    for x in range(self.board_size):
                        pygame.draw.rect(screen,
                                         grass_colors[y][x],
                                         (x * cell_size,
                                          y * cell_size,
                                          cell_size,
                                          cell_size))

            # Draw apples and snake
            for y in range(self.board_size):
                for x in range(self.board_size):
                    value = self.board[y][x]
                    rect = pygame.Rect(x * cell_size,
                                       y * cell_size,
                                       cell_size,
                                       cell_size)
                    if render_mode == "basic":
                        if value == 'S':
                            pygame.draw.rect(screen, (255, 255, 0), rect)
                        elif value == 'A':
                            pygame.draw.rect(screen, (0, 255, 0), rect)
                        elif value == 'R':
                            pygame.draw.rect(screen, (255, 0, 0), rect)
                    elif render_mode == "classic":
                        if value == 'A':
                            screen.blit(img_green_apple, rect.topleft)
                        elif value == 'R':
                            screen.blit(img_red_apple, rect.topleft)

            # Draw snake with proper assets in classic mode
            if (render_mode == "classic"
               and hasattr(self, 'snake')
               and self.snake):
                for i, (sy, sx) in enumerate(self.snake):
                    rect = pygame.Rect(sx * cell_size, sy * cell_size,
                                       cell_size,
                                       cell_size)
                    screen.blit(get_snake_part_image(self.snake, i),
                                rect.topleft)

            # Draw text
            text_surface = font.render(
                f"Steps: {self.steps} Snake length: {len(self.snake)}",
                True,
                (255, 255, 255),
            )
            screen.blit(text_surface, (10, self.board_size * cell_size + 5))

            pygame.display.flip()
            clock.tick(10)

        pygame.quit()
