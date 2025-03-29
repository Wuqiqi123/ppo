import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim = 64):
        super(Actor, self).__init__()
        
        self.proj_in = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, num_actions), 
            nn.Softmax(dim=-1)
        )

    @torch.no_grad()
    def forward_eval(self, state):
        training = self.training
        self.eval()
        out = self.forward(state)
        self.train(training)
        return out
    
    def forward(self, state):
        '''
            state: [batch_size, state_dim]
            return: [batch_size, num_actions]
        '''
        x = self.proj_in(state)
        return self.action_head(x)
    

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim = 64, dim_pred = 1):
        super(Critic, self).__init__()
        
        self.proj_in = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.value_head = nn.Linear(hidden_dim, dim_pred)

    def forward(self, state):
        x = self.proj_in(state)
        x = self.value_head(x)
        return x



