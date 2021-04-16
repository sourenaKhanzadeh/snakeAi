from game import Snake
from agent import Agent
from settings import *
from enum import Enum
import multiprocessing as mp
from multiprocessing import Pool, Process
import os
import json
import random

class Windows(Enum):
    W1 = (20, 20, 1, 3)
    W2 = (20, 20, 3, 1)
    W3 = (30, 30, 1, 1)
    W4 = (20, 20, 1, 1)


class Game:
    def __init__(self, lv=1):
        self.lv = lv
        self.awake()


    def awake(self):
        processes = []

        file = open('par_lev.json', 'r')
        json_pars = json.load(file)
        file.close()

        for window in Windows:

            pars = json_pars.get(window.name, None) or [{}]


            if window.name == "W" + str(self.lv):
                n, m, k, l = window.value
                n, m = (set_size(n), set_size(m))
                index = 0
                for i in range(k):
                    for j in range(l):
                        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100+(n+m)*i,100+(n+m)*j)
                        if index < len(pars) and len(pars) > 0:
                            p  = Process(target=self.train, args=(n, m, pars[index]))
                        elif len(pars) >= index:
                            p  = Process(target=self.train, args=(n, m, {}))
                        else:
                            p  = Process(target=self.train, args=(n, m, pars[0]))
                        p.start()
                        processes.append(p)
                        index += 1
                break
        for p in processes:
            p.join()
    
    # save run stats to a txt file at a specified path
    # txt files are used by build_graphs.py to build graphs
    def save_to_file(self, path, game_num, score, record):
        file = open(path, "a+")
        file.write("%s %s %s\n" % (game_num, score, record))
        file.close()

    def train(self, n, m, pars):
        record = 0
        game = Snake(n, m, pars.get('n_food', None))
        agent = Agent(game)

        while True:
            # get old state
            state_old = game.get_state()

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move, pars)
            state_new = game.get_state()

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if pars.get('num_games', DEFAULT_END_GAME_POINT) != -1:
                if agent.n_games > pars.get('num_games', DEFAULT_END_GAME_POINT):
                    quit()
                    break
            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    #agent.model.save()

                # takes away food depending on given probability, up until 1 food remains
                decrease_probability = pars.get('decrease_food_chance', None) or DECREASE_FOOD_CHANCE
                if (game.n_food > 1) and (random.random() < decrease_probability):
                    game.n_food -= 1

                # prints game information to console
                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                # appends game information to txt filen at specified path
                self.save_to_file(f"graphs/{pars.get('graph', 'test')}.txt", agent.n_games, score, record)    
    

if __name__ == "__main__":
    # for i in range(2, 3):
        # Game(i)
    Game(1)
