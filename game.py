import pygame
from pygame.locals import *
from pygame.color import Color
import numpy as np
from enum import Enum
from settings import *


class Dir(Enum):
    """
    Dir enum
    directions of the snake
    L: left
    R: right
    U: up
    D: down
    """
    L = 0
    R = 1
    U = 2
    D = 3


class DefaultImediateReward:
    """
    DefaultImediateReward class
    these are the default imediate
    reward for the snakes
    """
    COLLISION_WALL  = -10
    COLLISION_SELF = -10
    LOOP = -10
    SCORED = 10
    CLOSE_TO_FOOD = 0
    FAR_FROM_FOOD = 0
    MID_TO_FOOD = 0
    VERY_FAR_FROM_FOOD = 0
    EMPTY_CELL = 0
    DEFAULT_MOVING_CLOSER = 0
    MOVING_AWAY = 0


class Snake:
    """
    Snake class
    responsible for 
    movement, controls, food generation
    and frame additions
    """
    TITLE = "Snake Game"

    def __init__(self, w=SCREEN_WIDTH, h=SCREEN_HEIGHT, n_food=None):
        """
        (Snake, int, int, int) -> None
        Initialize all the attributes
        w: screen width to see how is calculated refer to settings.py
        h: screen height to see how is calculated refer to settings.py
        n_food: number of food display in the window 
        """
        self.size = SIZE
        self.speed = DEFAULT_SPEED

        self.w = w
        self.h = h

        # set screen caption and initialize Clock
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(Snake.TITLE)
        self.clock = pygame.time.Clock()

        # reset game once
        self.n_food = DEFAULT_N_FOOD if n_food == None else n_food
        self.reset()
        self.frame = 0

    def reset(self):
        """
        (Snake) -> None
        reset all Snake attributes
        """
        self.dir = Dir.R

        x, y = self.head = (self.w/2, self.h/2)
        self.body = [self.head, (x-self.size, y),(x-(2*self.size), y)]

        self.score = 0
        self.le = 3
        self.food = []
        self.food.clear()

        self.gen_food()
        self.frame = 0

    def gen_food(self):
        """
        (Snake) -> None
        generate food in the 
        environment
        """
        # generate a random point in the screen
        x = random.randint(0, (self.w-self.size )//self.size )*self.size
        y = random.randint(0, (self.h-self.size )//self.size )*self.size

        # append food
        self.food.append((x, y))

        # if food found in snake body then regenerate recursiveley
        if self.food[-1] in self.body:
            self.gen_food()
            
        if self.n_food == 1 and len(self.food) > 1:
            del self.food[1:]

    def get_state(self):
        """
        (Snake) -> np.array(dtype=int): (1, 11)
        return the current state of the snake 
        as following
        [danger straight, danger right, danger left, dir_l, dir_r, dir_u, dir_d, food left, food right, food up, food down]
        for example:
            [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
        """
        # take head of the snake and all the foods locations
        x, y = self.head
        fx, fy = zip(*self.food)
        # conver food locs into numpy arrays
        fx, fy = np.array(fx), np.array(fy)

        # the tile which the snake will land on
        point_l = (x - self.size, y)
        point_r = (x + self.size, y)
        point_u = (x, y - self.size)
        point_d = (x, y + self.size)

        # the current direction of the snake
        dir_l = self.dir == Dir.L
        dir_r = self.dir == Dir.R
        dir_u = self.dir == Dir.U
        dir_d = self.dir == Dir.D

        state = [
            # Danger straight
            (dir_r and self.collision(point_r)) or
            (dir_l and self.collision(point_l)) or
            (dir_u and self.collision(point_u)) or
            (dir_d and self.collision(point_d)),

            # Danger right
            (dir_u and self.collision(point_r)) or
            (dir_d and self.collision(point_l)) or
            (dir_l and self.collision(point_u)) or
            (dir_r and self.collision(point_d)),

            # Danger left
            (dir_d and self.collision(point_r)) or
            (dir_u and self.collision(point_l)) or
            (dir_r and self.collision(point_u)) or
            (dir_l and self.collision(point_d)),

            # Move Dir
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            any(fx < x),  # food left
            any(fx > x),  # food right
            any(fy < y),  # food up
            any(fy > y)   # food down
            ]
                
        return np.array(state, dtype=int)

    def play_step(self, action, kwargs={None:None}):
        """
        (Snake, np.array(dtype=int): (1, 3), dict()) -> (float, bool, int)
        play the current Snake frame and returns its immediate reward, terminal check
        and the current score
        action: numpy array which specifies whether snake 
                turns left, right or keeps straight
        kwargs: dynamic parameters of snake
        """
        self.frame += 1

        # pop food if more than maximum
        if len(self.food) < self.n_food:
            self.gen_food()

        # Quit out of the game
        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit()
                quit()
            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    pygame.quit()
                elif ev.key == K_q:
                    pygame.quit()
                quit()

        # move snake
        self.move(action)

        # insert a new head
        self.body.insert(0, self.head)

        # get immdeiate reward for when snake is very_far from the food
        reward = kwargs.get('very_far_range', DefaultImediateReward.VERY_FAR_FROM_FOOD)
        terminal = False

        # check if snake collided with itself or the wall
        if self.collision():
            # end game and return appropiate immediate reward and final score
            terminal = True
            reward = kwargs.get('col_wall', DefaultImediateReward.COLLISION_WALL)
            return reward, terminal, self.score

        # snake is stuck on the loop
        if self.frame > kwargs.get('kill_frame', DEFAULT_KILL_FRAME)*len(self.body):
            # game ends return everything
            terminal = True
            reward = kwargs.get('loop',  DefaultImediateReward.LOOP)
            return reward, terminal, self.score

        # for all the food in the snake 
        # if player its a food then 
        # add score regenerate food 
        # delete old food
        for fx, fy in self.food:
            if self.head == (fx, fy):
                self.score += 1
                self.le += 1
                reward = kwargs.get('scored',  DefaultImediateReward.SCORED)
                del self.food[self.food.index((fx, fy))]
                self.gen_food()

        # check the length of the snake delete extras
        if len(self.body) > self.le:
            self.body.pop()

        self.update()
        # speed of the snake
        self.clock.tick(self.speed)

        distance = np.array(self.distance())//self.size # distance in tiles

        # food close to snake head
        if any(CLOSE_RANGE[0] <= distance) and  any(distance < CLOSE_RANGE[1]):
            reward = kwargs.get('close_range', DefaultImediateReward.CLOSE_TO_FOOD)
            return reward, terminal, self.score
        # food far to snake head
        elif any(FAR_RANGE[0] <= distance) and any(distance <FAR_RANGE[1]):
            reward = kwargs.get('far_range', DefaultImediateReward.FAR_FROM_FOOD) 
            return reward, terminal, self.score

        # turn direction mode
        # give snake immediate reward the closer or farther it gets
        # from the food
        if kwargs.get('is_dir', False):
            for fx, fy in self.food:
                x, y = self.head
                if self.dir == Dir.R or self.dir == Dir.L:
                    if  x > fx and self.dir == Dir.R:
                        reward = kwargs.get("moving_away", DefaultImediateReward.MOVING_AWAY) # moving closer
                    if x < fx and self.dir == Dir.L:
                        reward = kwargs.get("moving_away", DefaultImediateReward.MOVING_AWAY) # moving closer
                    if x > fx and self.dir == Dir.L:
                        reward = kwargs.get("moving_closer", DefaultImediateReward.DEFAULT_MOVING_CLOSER) # moving further away
                    if x < fx and self.dir == Dir.R:
                        reward = kwargs.get("moving_closer", DefaultImediateReward.DEFAULT_MOVING_CLOSER) # moving further away
                if self.dir == Dir.U or self.dir == Dir.D:
                    if y > fy and self.dir == Dir.D:
                        reward = kwargs.get("moving_away", DefaultImediateReward.MOVING_AWAY) # moving closer 
                    if y < fy and self.dir == Dir.U:
                        reward = kwargs.get("moving_away", DefaultImediateReward.MOVING_AWAY) # moving closer 
                    if y > fy and self.dir == Dir.U:
                        reward = kwargs.get("moving_closer", DefaultImediateReward.DEFAULT_MOVING_CLOSER) # moving further
                    if y < fy and self.dir == Dir.D:
                        reward = kwargs.get("moving_closer", DefaultImediateReward.DEFAULT_MOVING_CLOSER) # moving further
                return reward, terminal, self.score

        # very far
        return reward, terminal, self.score

    def draw(self):
        """
        (Snake) -> None
        draw the snake and its food every frame
        """
        # paint snake red and paint food black
        # draw all foods and bodies of the snake
        for x, y in self.body:
            pygame.draw.rect(self.screen, Color('red'), (x, y, self.size, self.size))
        for x, y in self.food:
            pygame.draw.rect(self.screen, Color('black'), (x, y, self.size, self.size))

    def distance(self):
        """
        (Snake) -> None
        get the eucladian distance of the food
        from the snakes head
        """
        dists = []
        x, y = self.head
        for fx, fy in self.food:
            dists.append(((fx - x)**2 + (fy - y)**2)**0.5)
        return dists

    def update(self):
        """
        (Snake) -> None
        Update the frames
        """
        self.screen.fill(Color('white'))
        self.draw()
        pygame.display.update()

    def collision(self, pt=None):
        """
        (Snake, (int, int)) -> bool
        pt: point of snakes head or what ever we want to check for collision
        check if snake collides with self or the wall
        """
        if not pt:
            x, y = self.head
        else:
            x, y = pt
        # hits boundary
        if x > self.w - self.size or x < 0 or y > self.h - self.size or y < 0:
            return True
        # hits itself
        if (x, y) in self.body[1:]:
            return True

        return False

    def move(self, action):
        """
        (Snake, np.array(dtype=int): (1, 3))
        move the snake
        """
        # move the snake counter clockwise
        # depending on the current direction it
        # is at
        clock_wise = [Dir.R, Dir.D, Dir.L, Dir.U]
        idx = clock_wise.index(self.dir)

        # [1, 0, 0] == move straight
        # [0, 1, 0] == right turn
        # [0, 0, 1] == left turn
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.dir = new_dir

        # get snake head
        x, y = self.head

        # add or subtract tiles up or down, left or right
        if self.dir == Dir.R:
            x += self.size
        elif self.dir == Dir.L:
            x -= self.size
        elif self.dir == Dir.U:
            y -= self.size
        elif self.dir == Dir.D:
            y += self.size

        # assign a new head
        self.head = (x, y)

            
    def control(self):
        """
        (Snake) -> None
        manual control of
        the snake, not very good
        under construction
        """
        keys = pygame.key.get_pressed()

        if keys[K_LEFT]:
            if self.dir != Dir.L:
                action = np.array([0, 0, 1])
            else:
                action = np.array([1, 0, 0])

        elif keys[K_RIGHT]:
            if self.dir != Dir.R:
                action = np.array([0, 1, 0])
            else:
                action = np.array([1, 0, 0])
        elif keys[K_UP]:
            if self.dir != Dir.R:
                action = np.array([0, 1, 0])
            else:
                action = np.array([0, 0, 1])
        elif keys[K_DOWN]:
            if self.dir != Dir.L:
                action = np.array([0, 1, 0])
            else:
                action = np.array([0, 0, 1])

        else:
            action = np.array([1, 0, 0])

        return action
