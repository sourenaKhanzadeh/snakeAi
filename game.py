import pygame
from pygame.locals import *
from pygame.color import Color
import numpy as np
from enum import Enum
from settings import *


class Dir(Enum):
    L = 0
    R = 1
    U = 2
    D = 3


class DefaultImediateReward(Enum):
    COLLISION_WALL  = -10
    COLLISION_SELF = -10
    LOOP = -10
    SCORED = 10
    CLOSE_TO_FOOD = 0
    FAR_FROM_FOOD = 0
    MID_TO_FOOD = 0
    VERY_FAR_FROM_FOOD = 0
    EMPTY_CELL = 0



class Snake:
    TITLE = "Snake Game"

    def __init__(self, w=SCREEN_WIDTH, h=SCREEN_HEIGHT, n_food=None):
        self.size = SIZE
        self.speed = DEFAULT_SPEED

        self.w = w
        self.h = h


        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(Snake.TITLE)
        self.clock = pygame.time.Clock()

        self.reset()
        self.frame = 0
        self.n_food = n_food or DEFAULT_N_FOOD

    def reset(self):
        self.dir = Dir.R

        x, y = self.head = (self.w/2, self.h/2)
        self.body = [self.head, (x-self.size, y),(x-(2*self.size), y)]

        self.score = 0
        self.le = 3
        self.food = []

        self.gen_food()
        self.frame = 0

    def gen_food(self):
         x = random.randint(0, (self.w-self.size )//self.size )*self.size
         y = random.randint(0, (self.h-self.size )//self.size )*self.size

         self.food.append((x, y))
         if self.food[-1] in self.body:
             self.gen_food()


    def get_state(self):
        x, y = self.head
        fx, fy = zip(*self.food)
        fx, fy = np.array(fx), np.array(fy)

        point_l = (x - self.size, y)
        point_r = (x + self.size, y)
        point_u = (x, y - self.size)
        point_d = (x, y + self.size)

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
            any(fy > y)  # food down
            ]

        return np.array(state, dtype=int)

    def play_step(self, action, kwargs={None:None}):
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

        self.body.insert(0, self.head)


        reward = kwargs.get('very_far_range', None) or DefaultImediateReward.VERY_FAR_FROM_FOOD.value
        terminal = False

        if self.collision():
            terminal = True
            reward = kwargs.get('col_wall', None) or DefaultImediateReward.COLLISION_WALL.value
            return reward, terminal, self.score

        if self.frame > 100*len(self.body):
            terminal = True
            reward = kwargs.get('loop', None) or DefaultImediateReward.LOOP.value
            return reward, terminal, self.score

        for fx, fy in self.food:
            if self.head == (fx, fy):
                self.score += 1
                self.le += 1
                reward = kwargs.get('scored', None) or DefaultImediateReward.SCORED.value
                del self.food[self.food.index((fx, fy))]
                self.gen_food()

        if len(self.body) > self.le:
            self.body.pop()

        self.update()
        self.clock.tick(self.speed)

        distance = np.array(self.distance())//self.size # distance in tiles

        # close
        if any(CLOSE_RANGE[0] <= distance) and  any(distance < CLOSE_RANGE[1]):
            reward = kwargs.get('close_range', None) or DefaultImediateReward.CLOSE_TO_FOOD.value
            return reward, terminal, self.score
        # far
        elif any(FAR_RANGE[0] <= distance) and any(distance <FAR_RANGE[1]):
            reward = kwargs.get('far_range', None) or DefaultImediateReward.FAR_FROM_FOOD.value
            return reward, terminal, self.score

        # very far
        return reward, terminal, self.score

    def draw(self):
        for x, y in self.body:
            pygame.draw.rect(self.screen, Color('red'), (x, y, self.size, self.size))
        for x, y in self.food:
            pygame.draw.rect(self.screen, Color('black'), (x, y, self.size, self.size))

    def distance(self):
        dists = []
        x, y = self.head
        for fx, fy in self.food:
            dists.append(((fx - x)**2 + (fy - y)**2)**0.5)
        return dists

    def update(self):
        self.screen.fill(Color('white'))
        self.draw()
        pygame.display.update()

    def collision(self, pt=None):
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
        clock_wise = [Dir.R, Dir.D, Dir.L, Dir.U]
        idx = clock_wise.index(self.dir)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.dir = new_dir

        x, y = self.head

        if self.dir == Dir.R:
            x += self.size
        elif self.dir == Dir.L:
            x -= self.size
        elif self.dir == Dir.U:
            y -= self.size
        elif self.dir == Dir.D:
            y += self.size

        self.head = (x, y)

        # if len(self.body) > self.le:
            # self.body.pop()
            
    def control(self):
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
        # if keys[K_UP]:
        #     self.dir = Dir.U
        # if keys[K_DOWN]:
        #     self.dir = Dir.D

        return action

if __name__ == "__main__":
    game = Snake()

    while True:
        action = game.control()
        reward, done, score = game.play_step(action)
        if done:
            game.reset()
