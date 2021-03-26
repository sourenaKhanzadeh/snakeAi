import pygame
from pygame.locals import *
from pygame.color import Color
import numpy as np
from enum import Enum
from settings import *
import random


class Dir(Enum):
    L = 0
    R = 1
    U = 2
    D = 3


class Snake:
    TITLE = "Snake Game"

    def __init__(self, w=SCREEN_WIDTH, h=SCREEN_HEIGHT):
        self.size = SIZE
        self.speed = DEFAULT_SPEED

        self.w = w
        self.h = h


        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(Snake.TITLE)
        self.clock = pygame.time.Clock()

        self.reset()
        self.frame = 0


    def reset(self):
        self.dir = Dir.R

        x, y = self.head = (self.w/2, self.h/2)
        self.body = [self.head, (x-self.size, y),(x-(2*self.size), y)]

        self.score = 0
        self.le = 3
        self.food = None

        self.gen_food()
        self.frame = 0

    def gen_food(self):
         x = random.randint(0, (self.w-self.size )//self.size )*self.size
         y = random.randint(0, (self.h-self.size )//self.size )*self.size

         self.food = (x, y)
         if self.food in self.body:
             self.gen_food()

    def play_step(self, action):
        self.frame += 1

        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit()
                quit()

        self.move(action)

        self.body.insert(0, self.head)


        reward = 0
        terminal = False

        if self.collision() or self.frame > 100*len(self.body):
            terminal = True
            reward = -10
            return reward, terminal, self.score


        if self.head == self.food:
            self.score += 1
            self.le += 1
            reward = 10
            self.gen_food()
        else:
            self.body.pop()

        self.update()
        self.clock.tick(self.speed)

        return reward, terminal, self.score

    def draw(self):
        for x, y in self.body:
            pygame.draw.rect(self.screen, Color('red'), (x, y, self.size, self.size))
        x, y = self.food
        pygame.draw.rect(self.screen, Color('black'), (x, y, self.size, self.size))

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
