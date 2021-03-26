from game import Snake
from agent import Agent
from setting import *
from enum imort Enum
import os

class Windows(Enum):
    W1 = (12, 12, 5, 5)
    W2 = (6, 6, 10, 10)


class Game:
    def __init__(self, lv=1):
        self.snakes = []
        self.agents = []

        self.lv = lv


    def awake(self):
        for window in Windows:
            if window.name == "W" + str(self.lv):
                n, m, k, l = window.value
                n, m = (set_size(n), set_size(m))

                for i in range(k):
                    for j in range(l):
                        game = Snake(n, m)
                        self.snake.append((i, j, game))
                        self.agent.append((i, j, Agent(game)))
                break


    def train(self):
        record = 0
        game = Snake()
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

                if score > record:
                    record = score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)


    def run(self):
        pass


    def quit(self):
        pass

if __name__ == "__main__":
    pass
