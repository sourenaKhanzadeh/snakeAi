from game import Snake
from agent import Agent
from settings import *
from enum import Enum
import multiprocessing as mp
from multiprocessing import Pool, Process
import os

class Windows(Enum):
    W1 = (20, 20, 3, 1)
    W2 = (10, 10, 2, 1)
    W3 = (30, 30, 1, 1)


class Game:
    def __init__(self, lv=1):
        self.lv = lv
        self.awake()


    def awake(self):
        processes = []
        for window in Windows:
            if window.name == "W" + str(self.lv):
                n, m, k, l = window.value
                n, m = (set_size(n), set_size(m))
                for i in range(k):
                    for j in range(l):
                        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (400+(n+m)*2*i,100+(n+m)*2*j)
                        p  = Process(target=self.train, args=(n, m))
                        p.start()
                        processes.append(p)
                break
        for p in processes:
            p.join()


    def train(self, n, m):
        record = 0
        game = Snake(n, m)
        agent = Agent(game)
        while True:
            # get old state
            state_old = game.get_state()

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = game.get_state()

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                # if score > record:
                #     record = score
                #     agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)


if __name__ == "__main__":
    g = Game(3)
