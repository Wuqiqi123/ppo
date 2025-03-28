import gymnasium as gym
from tqdm import tqdm
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from collections import deque
from torch.distributions import Categorical
from ppo.ema import EMA
import numpy as np

from ppo.ppo import Actor, Critic


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@dataclass
class Buffer: 
    ## (s_t, a_t, r_t, s_t+1, log_pi_t, terminated, V_t)
    states:  torch.Tensor
    actions: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    action_log_prob: torch.Tensor
    terminated: torch.Tensor
    value: torch.Tensor


class PPOAgent:
    def __init__(self, env, actor_hidden_dim = 64, critic_hidden_dim = 256,
            ppo_clip = 0.2,
            ppo_epochs = 4,
            ppo_batch_size = 64,
            ppo_lr = 3e-4,
            ppo_eps = 1e-5,
            gae_lambda = 0.98,
            entropy_coef = 0.01,
    ):
        self.env = env
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.ppo_lr = ppo_lr
        self.ppo_eps = ppo_eps
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

        self.state_dim = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.actor = Actor(self.state_dim, self.num_actions, actor_hidden_dim).to(device)
        self.critic = Critic(self.state_dim, critic_hidden_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr = 3e-4, eps = 1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr = 3e-4, eps = 1e-5)
        self.target_critic = EMA(self.critic)
        self.target_critic.add_to_optimizer_post_step_hook(self.optimizer_critic)
        
        

    @torch.no_grad()
    def get_action(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob

    def optimize(self, memories, next_state):
        pass
    
if __name__ == "__main__":
    num_episodes = 50000
    update_timesteps = 2048
    max_timesteps = 1000

    # Initialise the environment
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = "./videos",
            name_prefix = 'video',
            episode_trigger = lambda eps_num: eps_num % 100 == 0,
            disable_logger = True
        )

    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)


    agent = PPOAgent(env)
    time = 0
    replay_buffer = deque([])
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc = 'episodes'):
        state, info = env.reset()
        for timestep in range(max_timesteps):
            time += 1
            # import ipdb; ipdb.set_trace()   
            state = torch.from_numpy(state).to(device)
            action_probs = agent.actor.forward_eval(state)
            value = agent.target_critic.forward_eval(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()
            
            next_state, reward, terminated, truncated, info = env.step(action)

            buffer = Buffer(state, action, reward, next_state, action_log_prob, terminated, value)
            replay_buffer.append(buffer)

            state = next_state

            if time % update_timesteps == 0:
                agent.optimize(replay_buffer, next_state)
                num_policy_updates += 1
                replay_buffer.clear()

            if terminated:
                break

    # Close the environment
    env.close()

