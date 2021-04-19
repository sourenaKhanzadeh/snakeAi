import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

"""
NOTE: this entire file has been copied from a git repo
Credit: 
    Authur: Patrick Loeber
    source: https://github.com/python-engineer/snake-ai-pytorch/blob/main/model.py
    Vocation: Python Engineer
    Company Name: Python Engineer
We do not own this code and do not take any credit from it
everything in this file is copy pasted code from above github repo
it is a public free to share code, only the documentation has been
added manually.
"""

class Linear_QNet(nn.Module):
    """
    Linear_QNet nn.Module class
    Model to use
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        (Linear_QNet, int, int, int) -> None
        input_size: size of the game states
        hidden_size: one layer network hidden layer size
        output_size: output size of the NN which is the number of snakes action
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # @Override
    def forward(self, x):
        """
        (Linear_QNet, *input) -> *output
        override forward function
        add relu activation
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """
        (Linear_QNet, str) -> None
        file_name: path of the save state files
        save the model state to a file_name
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """
    QTrainer class
    train the model
    """
    def __init__(self, model, lr, gamma):
        """
        (QTrainer, Linear_QNet, float, float) -> None
        initialize all model parameters
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        (QTrainer, float, long, float, float, bool) -> None
        state: current state of agent
        action: current action taken by the agent
        reward: current immediate reward
        next_state: next state of the agent
        done: terminal boolean
        """
        # turn into tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()

        # update Q values
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        self.optimizer.zero_grad()
        # calculate the loss
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
