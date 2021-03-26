import pygame
from pygame.locals import *
import multiprocessing as mp
from multiprocessing import Pool, Process
import os
# Just a sample multiprocessing pygame

class Game:
    def __init__(self):
        self.scene = []
        self.screen = pygame.display.set_mode((100, 100))


    def update(self):
        run = True
        while run:
            for ev in pygame.event.get():
                if ev.type == QUIT:
                    run = False

            self.screen.fill((0xff,0xff,0xff))
            pygame.display.update()

def exec():
    g = Game()
    g.update()
    pygame.quit()
    quit()


if __name__ == "__main__":
    processes = []
    for i in range(10):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100+100*i,300)
        p  = Process(target=exec)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
