
SIZE = 20

set_size  = lambda x: SIZE * x

DEFAULT_WINDOW_SIZES = (32, 24)

# set to None to change to Default
WINDOW_N_X = 12
WINDOW_N_Y = 12


SCREEN_WIDTH = set_size(WINDOW_N_X) or set_size(DEFAULT_WINDOW_SIZES[0])
SCREEN_HEIGHT = set_size(WINDOW_N_Y) or set_size(DEFAULT_WINDOW_SIZES[1])

DEFAULT_SPEED = 40

# Neural Networks Configuration
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

GAMMA = 0.9
EPSILON = 80
