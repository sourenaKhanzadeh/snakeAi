import random

# snake size
SIZE = 20

# ranges for defining close and far
CLOSE_RANGE = (0, 2)
FAR_RANGE = (CLOSE_RANGE[1], 9)

set_size  = lambda x: SIZE * x

DEFAULT_WINDOW_SIZES = (32, 24)

# set to None to change to Default
WINDOW_N_X = 12
WINDOW_N_Y = 12


SCREEN_WIDTH = set_size(WINDOW_N_X) or set_size(DEFAULT_WINDOW_SIZES[0])
SCREEN_HEIGHT = set_size(WINDOW_N_Y) or set_size(DEFAULT_WINDOW_SIZES[1])

DEFAULT_KILL_FRAME = 100
DEFAULT_SPEED = 50 # change the speed of the game
DEFAULT_N_FOOD = 1
DECREASE_FOOD_CHANCE = 0.8

# Neural Networks Configuration
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

GAMMA = 0.9

EPSILON = 80

EPS_RANGE = (0, 200)
is_random_move = lambda eps, eps_range: random.randint(eps_range[0], eps_range[1]) < eps


DEFAULT_END_GAME_POINT = 300