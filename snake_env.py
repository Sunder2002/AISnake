# snake_env.py
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from enum import Enum
from collections import namedtuple

# --- Reusable Game Logic Components (adapted from snake_game_ai.py) ---
Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
# --------------------------------------------------------------------

class SnakeEnv(gym.Env):
    """
    Custom Environment for Snake Game that follows gym interface.
    Action space: 0=Straight, 1=Right Turn, 2=Left Turn (relative)
    Observation space: 11-dim vector (danger_straight, danger_right, danger_left,
                                     dir_l, dir_r, dir_up, dir_down,
                                     food_l, food_r, food_up, food_down)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15} # Adjusted FPS

    def __init__(self, grid_size=20, render_mode=None):
        super().__init__()

        self.grid_size = grid_size
        self.w = self.grid_size * BLOCK_SIZE
        self.h = self.grid_size * BLOCK_SIZE

        # Define action and observation space
        # They must be gym.spaces objects
        # Action: 0=Straight, 1=Right Turn, 2=Left Turn
        self.action_space = spaces.Discrete(3)

        # Observation: 11 features described above
        # Use float32 for observations as it's standard for SB3
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Pygame setup (only if rendering)
        self.window = None
        self.clock = None
        self.font = None
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption('SnakeEnv - Stable Baselines3')
            self.window = pygame.display.set_mode((self.w, self.h))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 25)

        # Internal game state variables are initialized in reset()
        self.direction = None
        self.head = None
        self.snake = None
        self.score = None
        self.food = None
        self.frame_iteration = None
        self.last_dist_to_food = None # For reward shaping

    def _get_obs(self):
        """ Calculates the 11-dimensional observation vector. """
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        obs = [
            # Danger Straight
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),

            # Danger Right
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),

            # Danger Left
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),

            # Move direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location
            self.food.x < head.x,  # Food left
            self.food.x > head.x,  # Food right
            self.food.y < head.y,  # Food up
            self.food.y > head.y   # Food down
        ]
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        """ Returns auxiliary information (optional). """
        # Calculate distance to food (Manhattan distance)
        dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        return {"score": self.score, "distance_to_food": dist / BLOCK_SIZE}

    def reset(self, seed=None, options=None):
        """ Resets the environment to an initial state and returns initial observation. """
        super().reset(seed=seed) # Important for reproducibility

        # Init game state
        self.direction = Direction.RIGHT
        center_x = (self.w // 2 // BLOCK_SIZE) * BLOCK_SIZE
        center_y = (self.h // 2 // BLOCK_SIZE) * BLOCK_SIZE
        self.head = Point(center_x, center_y)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.last_dist_to_food = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _place_food(self):
        """ Places food randomly, avoiding the snake. """
        while True:
            x = random.randrange(0, self.w, BLOCK_SIZE)
            y = random.randrange(0, self.h, BLOCK_SIZE)
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def _move(self, action):
        """ Updates the snake's direction based on the discrete action. """
        # Action mapping: 0=Straight, 1=Right Turn, 2=Left Turn
        # Directions ordered clockwise: RIGHT, DOWN, LEFT, UP
        clockwise_dirs = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_idx = clockwise_dirs.index(self.direction)
        new_idx = current_idx

        if action == 1: # Right Turn
            new_idx = (current_idx + 1) % 4
        elif action == 2: # Left Turn
            new_idx = (current_idx - 1 + 4) % 4 # +4 to handle negative index
        # else action == 0: No change

        self.direction = clockwise_dirs[new_idx]

        # Update head position
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)


    def _is_collision(self, pt=None):
        """ Checks if the given point causes a collision. """
        if pt is None:
            pt = self.head
        # Hits boundary
        if not (0 <= pt.x < self.w and 0 <= pt.y < self.h):
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def step(self, action):
        """ Executes one time step within the environment. """
        self.frame_iteration += 1

        # 1. Move snake based on action
        self._move(action)
        self.snake.insert(0, self.head)

        # 2. Check if game over
        terminated = False # Game over state
        reward = 0
        # Collision or too long without food
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            terminated = True
            reward = -10.0 # Penalty for dying/stalling
            self.score = self.score # Keep score as is on termination
            observation = self._get_obs() # Get final observation
            info = self._get_info()
            return observation, reward, terminated, False, info # False for truncated

        # 3. Check if food is eaten
        food_eaten = (self.head == self.food)
        if food_eaten:
            self.score += 1
            reward = 10.0 # Reward for eating food
            self._place_food()
            self.frame_iteration = 0 # Reset frame counter
            self.last_dist_to_food = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y) # Reset dist
        else:
            self.snake.pop() # Remove tail segment if no food eaten
            # Reward shaping: Give small reward/penalty for getting closer/further from food
            current_dist_to_food = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            if current_dist_to_food < self.last_dist_to_food:
                reward += 0.1 # Small reward for getting closer
            else:
                reward -= 0.15 # Slightly larger penalty for moving away/staying same dist
            self.last_dist_to_food = current_dist_to_food


        # 4. Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        # 5. Render (if applicable)
        if self.render_mode == "human":
            self._render_frame()

        # Gymnasium API returns: observation, reward, terminated, truncated, info
        truncated = False # We use termination for game end, not truncation
        return observation, reward, terminated, truncated, info

    def render(self):
        """ Renders the environment (needed for gym compatibility). """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        # Human rendering is handled in step()

    def _render_frame(self):
        """ Internal method to draw the current game state. """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption('SnakeEnv - Stable Baselines3')
            self.window = pygame.display.set_mode((self.w, self.h))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if self.font is None and self.render_mode == "human":
             self.font = pygame.font.Font(None, 25)

        # Create canvas
        canvas = pygame.Surface((self.w, self.h))
        canvas.fill(BLACK)

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(canvas, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(canvas, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)) # Inner square

        # Draw food
        pygame.draw.rect(canvas, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        if self.font:
            text = self.font.render("Score: " + str(self.score), True, WHITE)
            canvas.blit(text, [5, 5])

        if self.render_mode == "human":
            # Update the window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump() # Process internal pygame events
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """ Closes the rendering window. """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None # Prevent further rendering attempts
            print("Pygame window closed.")