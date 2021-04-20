# snakeAi
Reinforcement Learning with the classic snake game

## Preview
<blockquote>
  starting from left, middle, right, 80, 120, 286 number of generations
</blockquote>

![Snake Game](snake_game.gif)

<object data="http://yoursite.com/the.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="http://yoursite.com/the.pdf">
        <p>Final Report: <a href="http://yoursite.com/the.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Installations 
---

![pytorch](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)
install pytorch from here: https://pytorch.org/

```bash
pip install -r requirements.txt
```


## Run The Game
---
```bash
python main.py
```

## Configurations
All static settings are in settings.py
```python
import random

# snake size
SIZE = 20

# ranges for defining close and far
CLOSE_RANGE = (0, 2)
FAR_RANGE = (CLOSE_RANGE[1], 9)

set_size  = lambda x: SIZE * x

DEFAULT_WINDOW_SIZES = (32, 24)

# set to None to change to Default
WINDOW_N_X = 12
WINDOW_N_Y = 12


SCREEN_WIDTH = set_size(WINDOW_N_X) or set_size(DEFAULT_WINDOW_SIZES[0])
SCREEN_HEIGHT = set_size(WINDOW_N_Y) or set_size(DEFAULT_WINDOW_SIZES[1])

DEFAULT_KILL_FRAME = 100
DEFAULT_SPEED = 50 # change the speed of the game
DEFAULT_N_FOOD = 1
DECREASE_FOOD_CHANCE = 0.8

# Neural Networks Configuration
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

GAMMA = 0.9

EPSILON = 80

EPS_RANGE = (0, 200)
is_random_move = lambda eps, eps_range: random.randint(eps_range[0], eps_range[1]) < eps


DEFAULT_END_GAME_POINT = 300
```

## Adding New Windows

Go to main.py and add more windows and processors
```python

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
```

### Run The Window

The following will run window 14
```python

if __name__ == "__main__":
    Game(14)
```

## Adding Parameters To Processors
---
In par_lev.json:
Add parameters for instance "world 1" where world is (20, 20, 1, 3) the following: 
first processor will have all these parameters, second the epsilon changes to 80 and graph name different and third parameters will use the 
default, look at the report for more info on default values and settings.py.
```json
{
"W1":[
    {
      "empty_cell":0,
      "very_far_range":0,
      "close_range":0,
      "far_range":0,
      "col_wall":-10,
      "loop":-10,
      "scored":10,
      "gamma":0.5,
      "eps":200,
      "eps_range":[0, 200],
      "hidden_size":256,
      "n_food":1,
      "decrease_food_chance":-1,
      "kill_frame":100,
      "num_games":-1,
      "is_dir":true,
      "moving_closer":10,
      "moving_away":-10,
      "lr":0.0005,
      "graph":"epsilon__200__0"
    },
    {
      "eps":80,
      "graph":"epsilon__200__1"
    },
    {
    }    
  ]
 }
```

# Reinforment Learning Using Snake Game
---

  | Main Contributors  | Emails |
  |---| --- |
  | Ronak Ghatalia  |    ronak.ghatalia@ryerson.ca |
  | Sourena Khanzadeh  | sourena.khanzadeh@ryerson.ca |
  |  Fahad Jamil  |   f1jamil@ryerson.ca | 
  |  Anthony Greco  | anthony1.greco@ryerson.ca |


# Introduction
---

Reinforcement learning (RL) is an area of machine learning where an agent aims to make the optimal decision in an uncertain environment to get the maximum cumulative reward. Since RL requires an agent to operate and learn in its environment, it&#39;s often easier to implement agents inside of simulations on computers than in real world situations.

A common technique to test various RL algorithms is to use an existing video game where the agent learns from its interactions with the game. Video games are an ideal environment to test RL algorithms because (1) they provide complex problems for the agent to solve, (2) they provide safe, controllable environments for the agent to act in, and (3) they provide instantaneous feedback to the agent which makes training much faster.

For the reasons listed above, we have chosen to implement an RL agent on the game Snake. Snake is an arcade game originally released in 1976 where the player maneuvers a line to eat food that both increases their score and the size of the snake. The game ends when the snake either runs into a wall or into itself and the score is equal to the number of food collected. Since each food collected increases the size of the snake, the game gets increasingly difficult as time goes on.

We chose to use Deep Q-Learning as our algorithm to allow our agent to make optimal decisions solely from raw input data. No rules about the Snake game are given to the agent. This approach consists solely on giving the agent information about the state of the game and giving it negative or positive feedback based on its actions taken.

# Related Research
---

A paper was published that used a learned policy gradient that is used to update the value function and learn from it with bootstrapping. It found that this &quot;generalizes effectively to complex Atari games&quot; [1].

Research has expanded from simple games to more complex strategy games. Amato and Shani conducted research on using reinforcement learning to strategy games. They found that it was helpful to use different policies depending on different contexts [2].

Deep learning is also being used in reinforcement learning. A paper was published that used a convolutional neural network with Q learning to play Atari games [3]. The paper found that it did better than other reinforcement learning approaches.

There have been some attempts to use bandits in video games. A paper was published that used semi-bandit and bandit and both cases converge to a Nash equilibrium [4].


# Method
---

A common algorithm that has been used is q-learning, and now has been expanded to include neural networks with deep Q learning methods. We decided that we could experiment with this new method that is gaining in use based on the research done with Atari games [3].

To do this we first used PyGame to create our own simple snake game with the basic rules of movement. The snake can only move forward, left, or right, and if it hits the wall or itself the game ends. As it consumes food it grows larger. The goal is to get the snake to eat as much food as possible without ending the game.

Once we created the game we then created a deep Q network using PyTroch. We created a network with an input layer of 11 which defines the current state of the snake, 1 hidden layer of 256 nodes, and an output layer of 3 to determine which action to take.

Since the game has continuous movement we discretized the states by calculating a state based on each frame, determined by the frames per second of the game. We defined our default state parameters to be 1 or 0 for 11 parameters based on the snake direction of movement forward, left, or right. The location of danger which we define as a collision that would occurred with an action in the next frame, to a wall or with the snake, which would be forward, left or right, and the location of food left, right, front, or back. We kept our input layer to allow for any size state parameters that are sent. This allowed us to include additional parameters to test for our state definitions. Our 3 actions are the directions of movement for the snake to move forward, left, or right.

When starting the training for the first time each state is passed to the Q-network and the best prediction is taken as the action. This information is then saved in memory. When returning to the game to continue training all the information learned from the states can be taken from memory and passed to the network to continue the training process.

Our game was designed so the frame rate can be changed so we can change how often the states are updated during the game. The amount of food generated can also be changed. For the model we can adjust epsilon, learning rate, gamma, the batch size, and rewards.


# Experiments
---

Based on our model we wanted to see how well this method allows our agent to learn, and the best way to optimize the performance of the agent. We have a base case of no learning to see how a random untrained agent would perform. Then we changed different parameters holding others constant to see how they impact the performance.

Below is the performance of a random agent.

![No Training](graphs/no_training.png)

This graph shows that our agent performed randomly each game with no learning. Its top score was only 1 food. Each game was up and down with no growth. There was no improvement in performance.

We did all our experiments with 3 agents, and use the average as our result to prevent any random change events in the learning. Since we were constrained with processing and time limitations we felt that 300 games would be a sufficient number of games for seeing differences in our experiments.

We decided to set our default parameters as the following, and made changes to individual parameters to see how they would change the performance:

```python
  Gamma = 0.9
  Epsilon = 0.4
  Food_amount = 1
  Learning_Rate = 0.001
  Reward_for_food (score) = 10
  Collision_to_wall = -10
  Collision_to_self = -10
  Snake_going_in_a_loop = -10
```

The graph below shows how our agent performed with the default parameters. The graphs of our experiments show the 5 game moving average of the score at the top to smooth out the graph. The bottom graph shows the running top score record.

![Default](graphs/default.png)

 
## Gamma
---

![Gamma](graphs/gamma.png)

In this experiment we changed gamma to see how this changed the results. We decided to test 0, 0.5, 0.99. The gamma of 0 meant that the agent was only focused on the immediate rewards which we assumed would do the best, because this is not a long term strategy game the focus is just to get the food each time. Based on the results the best performance was with a gamma of 0.5 which showed much better performance than the other two. This shows that putting weights to the future rewards improves the performance.

## Epislon
---

![Eps](graphs/epsilon.png)

We wanted to test how exploration and exploitation would impact the performance. We decided to try no exploration with epsilon of 0 to see what would happen with no exploration, 0.5 to be a balance of both, and 1 to be always exploring with random actions.

Based on the results the epsilon of 0 finds some actions and does not explore so from the graph you can see a small amount of learning. With 0.5 our agent finds optimal values but explores and the graph shows much better learning. With an epsilon of 1 our agent keeps taking random actions so the learning is very slow and it takes a longer for it to find the optimal policy but once it does it seems to get up to where the epsilon of 0.5 does.

 
## Rewards
---

![Rewards](graphs/immediate_rewards.png)

In this experiment we decided to change the immediate rewards. Our immediate rewards were for score (S) getting the food, collision with wall or itself (C), and getting stuck in a loop (L). We found that having a large difference between the positive and negative rewards does not do well, possibly because the agent will learn to focus on the larger of the negative or positive rewards, having the reward of equal magnitude allows for better performance. We also found that having rewards that are small in scale do better than rewards that are large in scale. The best performance we found was with rewards of 5 and -5, performance seed to decrease below that at 1 but the performance was quite close, the larger rewards of 30 and -30 performed much worse.

 
## Learning rate
---

First, confirm that you have the correct template for your

![Lr](graphs/learning_rate.png)

Changing the learning rate would impact if our agent was able to find an optimal policy and how fast it would learn. We found that a learning rate of 0.05 was too high since this did not produce any learning, and performed similar to our random agent. This was probably because the agent was skipping over the optimal by taking too large a step size. We decide to lower the rate and found that it improves the learning, with the best being the lowest rate we tested of 0.0005, which did significantly better.


## Direction
---

![Direction](graphs/food_direction.png)

We decide to try adding a reward for direction of movement of the snake. If it was moving closer to the food we would give it a positive reward, and if it was moving away from the food we would give it a negative reward. The reward of 5 and -5 seemed to perform the best with the fastest learning. This did provide learning for our agent but it did worse than our default parameters that did not have this reward.

 
## Distance
---

![Distance](graphs/distance_food.png)

We added rewards for the distance the snake was from the food. We added rewards for score getting the food, close range was 1-2 steps away, far range was 2-9 steps away, and very far range was anything further. This did not show any learning and had a performance similar to our random agent. We did not continue to experiment with this because of the poor performance.


## Food Generation
---

![Food Gen](graphs/multiple_food.png)

Since the game is played with providing the player with one food to get at a time. We decided to see if the agent would learn better by providing it with more food at the start and then decaying the amount of food, thinking that this would improve performance. The agent seemed to not start learning until the food decayed to 1 in all tests. This is probably because there were so many rewards around the agent that it was not able to figure out the optimal policy for when there was only 1 food, causing it to perform poorly and not learn until the food decayed to 1.


# Implementation and Code
---

We started out by making the snake arcade game using pygame and numpy packages, the game is created inside game.py, the snake is represented as a class, this class is responsible for snake movement, eating and collision functionalities. The window is n x m pixels where is the number of tiles of size S2 where S is the length of the snake, all static configuration is in settings.py where is a mix of constants and few lambda functions. The snake moves by tiles of size S2 being appended to the head of the snake and the rest of the body popped until the current size of the snake is satisfied in the direction of the movement. There are 4 directions: &quot;left&quot;, &quot;right&quot;, &quot;up&quot;, &quot;down&quot;. For simplicity the snake can only turn right, left, or keep moving straight as an action. Food generation is also implemented inside of the snake, where it will generate food randomly inside of the environment and will regenerate if it is inside of the body of the snake. Snake also creates an array size of (11, 1) bits that is stored as the states. Danger in right turn, Danger in straight, Danger in straight, currently moving, left, currently moving right, currently moving up, currently moving down, food on left side of snake, food on right side of snake, food above snake, food below snake. The agent.py is responsible for short term memory and long term memory of the snake, it is also responsible for getting a random action or a predicted action from the model. Model.py is responsible for training of the model and deep-learning itself which we used pytorch and numpy packages for. Main.py is responsible for putting everything together, we implemented a multiprocessing multi level script which reads the parameters of every processor and runs them on separate windows.


# Conclusion
---

## UML
![class uml](classes_snakeAi_fixed.png)
![Packages UML](packages_snakeAi.png)

##### References

1. [https://arxiv.org/pdf/2007.08794.pdf](https://arxiv.org/pdf/2007.08794.pdf)
2. [https://www.researchgate.net/profile/Christopher-Amato/publication/221455879\_High-level\_reinforcement\_learning\_in\_strategy\_games/links/02e7e51e8095ce66df000000/High-level-reinforcement-learning-in-strategy-games.pdf](https://www.researchgate.net/profile/Christopher-Amato/publication/221455879_High-level_reinforcement_learning_in_strategy_games/links/02e7e51e8095ce66df000000/High-level-reinforcement-learning-in-strategy-games.pdf)
3. [https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602)
4. [https://papers.nips.cc/paper/2017/file/39ae2ed11b14a4ccb41d35e9d1ba5d11-Paper.pdf](https://papers.nips.cc/paper/2017/file/39ae2ed11b14a4ccb41d35e9d1ba5d11-Paper.pdf)
5. [https://medium.com/@hugo.sjoberg88/using-reinforcement-learning-and-q-learning-to-play-snake-28423dd49e9b](https://medium.com/@hugo.sjoberg88/using-reinforcement-learning-and-q-learning-to-play-snake-28423dd49e9b)
6. [https://towardsdatascience.com/learning-to-play-snake-at-1-million-fps-4aae8d36d2f1](https://towardsdatascience.com/learning-to-play-snake-at-1-million-fps-4aae8d36d2f1)
7. http://cs229.stanford.edu/proj2016spr/report/060.pdf
8. [https://mkhoshpa.github.io/RLSnake/](https://mkhoshpa.github.io/RLSnake/)
9. [https://docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html)
10. [https://github.com/eidenyoshida/Snake-Reinforcement-Learning](https://github.com/eidenyoshida/Snake-Reinforcement-Learning)
11. [https://github.com/python-engineer/snake-ai-pytorch/blob/main/model.py](https://github.com/python-engineer/snake-ai-pytorch/blob/main/model.py)

