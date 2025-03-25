import gymnasium as gym
from tqdm import tqdm
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from collections import deque
from torch.distributions import Categorical
from 

from ppo.ppo import Actor, Critic


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@dataclass
class Buffer: 
    ## (s_t, a_t, r_t, s_t+1, log_pi_t, done_t, V_t)
    states:  torch.Tensor
    actions: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    action_log_prob: torch.Tensor
    done: torch.Tensor
    value: torch.Tensor


class Agent:
    def __init__(self, env, actor_hidden_dim = 64, critic_hidden_dim = 256):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.actor = Actor(self.state_dim, self.num_actions, actor_hidden_dim).to(device)
        self.critic = Critic(self.state_dim, critic_hidden_dim).to(device)
    

    @torch.no_grad()
    def get_action(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob
    
if __name__ == "__main__":
    num_episodes = 50000
    update_timesteps = 2048
    max_timesteps = 500

    # Initialise the environment
    env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = "./videos",
            name_prefix = 'lunar-video',
            episode_trigger = lambda eps_num: eps_num % 100 == 0,
            disable_logger = True
        )

    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)

    agent = Agent(env)
    time = 0
    replay_buffer = deque([])

    for eps in tqdm(range(num_episodes), desc = 'episodes'):

        for timestep in range(max_timesteps):
            time += 1
            state = torch.from_numpy(state).to(device)

    # Close the environment
    env.close()

