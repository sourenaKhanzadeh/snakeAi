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
    """
    Windows enums
    W_i = (m, n, s, k)
    where m is number of row tiles
          n is number of column tiles
          s is number of row processors
          k is number of column processors
    """
    #W1 = (20, 20, 1, 3)
    # W2 = (20, 20, 3, 1)
    # W5 = (20, 20, 3, 1)
    #W4 = (20, 20, 1, 1)
    # W6 = (20, 20, 3, 1)
    # W7 = (20, 20, 3, 1)
    # W8 = (20, 20, 3, 1)
    # W9 = (20, 20, 3, 1)
    # W10 = (20, 20, 3, 1)
    # W11 = (20, 20, 3, 1)
    # W12 = (20, 20, 3, 1)
    # W13 = (20, 20, 3, 1)
    W14 = (20, 20, 1, 1)


class Game:
    """
    Run the Game
    Read the json file par_lev.json
    run all the processors with .json
    parameters
    """
    def __init__(self, lv=1):
        """
        (Game, int) -> None
        initialize the game and what 
        world you want the game to run in
        lv: level selected in pygame
            by default it is set to world 1
        """
        self.lv = lv
        self.awake()


    def awake(self):
        """
        (Game) -> None
        read json file get the worlds
        with their parameters
        iterate through all the world until we found
        the world we initialized create the enviroments
        run processors parallel with each other using multiprocessing
        """
        processes = []

        file = open('par_lev.json', 'r')
        json_pars = json.load(file)
        file.close()

        # get all enum Windows
        for window in Windows:

            # get specific window from json file
            pars = json_pars.get(window.name, [{}])

            # check if window exist
            if window.name == "W" + str(self.lv):
                # take the window and unpack values
                n, m, k, l = window.value
                # set screen size with nxm tiles
                n, m = (set_size(n), set_size(m))
                index = 0
                # change position of the screen of each processor window
                # accordingly
                # NOTE: the margin might not be what u excpect 
                for i in range(k):
                    for j in range(l):
                        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100+(n+m)*i,100+(n+m)*j)
                        # create the processors and add in the pool
                        if index < len(pars) and len(pars) > 0:
                            p  = Process(target=self.train, args=(n, m, pars[index]))
                        elif len(pars) >= index:
                            p  = Process(target=self.train, args=(n, m, {}))
                        else:
                            p  = Process(target=self.train, args=(n, m, pars[0]))
                        # start proceesors
                        p.start()
                        processes.append(p)
                        index += 1
                break
        for p in processes:
            # join every processors
            p.join()
    
    # save run stats to a txt file at a specified path
    # txt files are used by build_graphs.py to build graphs
    def save_to_file(self, path, game_num, score, record):
        """
        (Game, str, int, int, int) -> None
        save the file as .txt file
        save game, score, record as following format
        g s r  respectively
        path: path of the txt file
        game_num: total number of generations
        score: current score taken from the game
        record: highest score
        """
        file = open(path, "a+")
        file.write("%s %s %s\n" % (game_num, score, record))
        file.close()

    def train(self, n, m, pars):
        """
        (Game, int, int, dict()) -> None
        train game and run each step as
        sequence of frames
        n: row tiles of the screen
        m: col tiles of the screen
        pars" parameters passed in for each processors
        """
        # initialize
        record = 0
        game = Snake(n, m, pars.get('n_food', None))
        agent = Agent(game, pars)

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

            # end game if reached num_games from pars or DEFAULT_END_GAME_POINT
            # if set to -1 then run for ever
            if pars.get('num_games', DEFAULT_END_GAME_POINT) != -1:
                if agent.n_games > pars.get('num_games', DEFAULT_END_GAME_POINT):
                    quit()
                    break

            # when game is over
            if done:
                # reset game attributes
                # increase game generation
                # train the long memory
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                # new highscore
                if score > record:
                    record = score
                    # save the best model_state
                    #agent.model.save()

                # takes away food depending on given probability, up until 1 food remains
                decrease_probability = pars.get('decrease_food_chance', DECREASE_FOOD_CHANCE)
                if (game.n_food > 1) and (random.random() < decrease_probability):
                    game.n_food -= 1
                
                # prints game information to console
                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                # appends game information to txt filen at specified path
                self.save_to_file(f"graphs/{pars.get('graph', 'test')}.txt", agent.n_games, score, record)    
    

if __name__ == "__main__":
    # for i in range(2, 3):
        # Game(i)
    Game(14)
