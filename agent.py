import torch
import numpy as np
from collections import deque
from game import Snake, Dir
from model import Linear_QNet, QTrainer
from settings import *

class Agent:
    """
    Agent class
    agent running and the snake
    """

    def __init__(self, game, pars=dict()):
        """
        (Agent, Snake, dict()) -> None
        Initialize everything
        get everything that is passed from 
        json file to modify attributes and train model
        """
        self.n_games = 0
        self.epsilon = pars.get('eps', EPSILON)
        self.eps = pars.get('eps', EPSILON)
        self.gamma = pars.get('gamma', GAMMA) # discount rate
        self.eps_range = pars.get('eps_range', EPS_RANGE)
        print(self.epsilon ,self.eps)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(len(game.get_state()), pars.get('hidden_size', HIDDEN_SIZE), OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, lr=pars.get('lr',LR), gamma=self.gamma)

        self.game = game

    def remember(self, *args):
        """
        (Agent, (float, float, float, float, bool)) -> None
        state: current state
        action: current actions
        reward: current immediate rewards
        next_state: get the next state
        done: terminal state point
        append all this attributes to the queue: memory
        do this every frame
        """
        state, action, reward, next_state, done = args
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        (Agent) -> None
        train after every game is finished
        """
        # get memory
        # if memory is above a certain BATCH SIZE then
        # randomly sample BACTCH SIZE memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        # get all states actions, rewards, etc...
        # and train the step using QTrainer
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, *args):
        """
        (Agent, (float, float, float, float, bool)) -> None
        state: current state
        action: current actions
        reward: current immediate rewards
        next_state: get the next state
        done: terminal state point
        train agent every game frame
        """
        state, action, reward, next_state, done = args
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        (Agent, float) -> np.array(dtype=int): (1, 3)
        get an action either from the policy or randomly
        """
        # tradeoff exploration / exploitation based on epsilon and eps_range
        self.epsilon = self.eps - self.n_games
        final_move = [0,0,0]
        # check if should move randomly
        if is_random_move(self.epsilon, self.eps_range):
            # if so then randomly turn one of the bits
            # to go right left or straight
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # else get the best move from the
            # NN by taking its argmax and setting 
            # its bits
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move