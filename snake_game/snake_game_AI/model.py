import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_Qnet(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Q_Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # loss function

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            # append 1D tensor shape
            state = torch.unsqueeze(state, 0) # takes in input tensor and index at which to insert singleton dimention i.e. [1,2,3,4] --> [[1], [2], [3], [4]]
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over, )

            # Bellman eqn implemented: NewQ(s,a) = Q(s,a) + alpha[R(s,a) + {gamma * maxQprime(sprime,aprime)} - Q(s,a)]
            # Q = model.predict(state0)
            

            # 1: get predicted Q values with current state
            prediction = self.model(state)

            target = prediction.clone()
            for i in range(len(game_over)):
                Q_new = reward[i]
                if not game_over[i]:
                    Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

                target[i][torch.argmax(action[i]).item()] = Q_new


            # 2: apply formula: Qnew = R + gamma * maxQ(state1) --> to be done ONLY when game_over == False
            # prediction.clone()
            # predictions[argmax(action)] = Qnew --> (sets index of predictions of action direction to move as the new Q value)
            self.optimizer.zero_grad() # empty the gradient
            loss = self.criterion(target, prediction) # loss func (Qnew - Q)^2
            loss.backward() # back propagation
            self.optimizer.step()


