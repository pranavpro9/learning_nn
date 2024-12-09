import torch
import random
import numpy as np
from collections import deque 
from AI_snake_game import SnakeGame_AI, Direction, Point
from model import Linear_Qnet, Q_Trainer
from helper import plot

MAX_MEM = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
BLOCK_SIZE = 20

class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0 # controls the randomness
        self.gamma = 0.9 # disc rate must be <1
        self.mem = deque(maxlen=MAX_MEM) # deque auto popleft()
        self.model = Linear_Qnet(11, 256, 3)
        self.trainer = Q_Trainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)
    
    def get_state(self, game):
        head = game.snake[0] # so that agent doesnt need to keep calling game.head
        pt_l = Point(head.x - BLOCK_SIZE, head.y)
        pt_r = Point(head.x + BLOCK_SIZE, head.y)
        pt_u = Point(head.x, head.y - BLOCK_SIZE)
        pt_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight (bool values show danger as 1 and no danger as 0)
            (dir_l and game.is_collision(pt_l)) or
            (dir_r and game.is_collision(pt_r)) or
            (dir_u and game.is_collision(pt_u)) or
            (dir_d and game.is_collision(pt_d)),

            # danger right
            (dir_l and game.is_collision(pt_u)) or
            (dir_r and game.is_collision(pt_d)) or
            (dir_u and game.is_collision(pt_r)) or
            (dir_d and game.is_collision(pt_l)),

            # danger left
            (dir_l and game.is_collision(pt_d)) or
            (dir_r and game.is_collision(pt_u)) or
            (dir_u and game.is_collision(pt_l)) or
            (dir_d and game.is_collision(pt_r)),

            # current moving dir (bool values only one of which can be True i.e 1 rest of which are 0)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location bool values True i.e. 1 for food in given direction (L,R,U,D) False i.e. 0 for no food in given direction
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
            ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.mem.append((state, action, reward, next_state, game_over)) # append variables to memory as a tuple

    def train_short_mem(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)
    
    def train_long_mem(self):
        if len(self.mem) > BATCH_SIZE:
            mini_sample = random.sample(self.mem, BATCH_SIZE) # func returns a list of tuples
        else:
            mini_sample = self.mem

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)  
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def get_action(self,state):
        # tradeoff betwn exploration vs exploitation i.e. randomness level of moves needs to be adjusted
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = [ ]
    score = []
    mean_score = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGame_AI()
    while True:    
        
        # get initial_state
        old_state = agent.get_state(game)

        # get action based on the state
        final_action = agent.get_action(old_state)
        
        # performs move, returning vars
        reward, game_over, score = game.play_step(final_action)
        
        # use vars to get new state
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_mem(old_state, final_action, reward, new_state, game_over)

        # remember 
        agent.remember(old_state, final_action, reward, new_state, game_over)

        if game_over:
            # train long memory and plot the results
            game.reset()
            agent.n_games += 1
            agent.train_long_mem()
            if score > best_score:
                best_score = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'High Score', best_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
         


if __name__ == '__main__':
     train()