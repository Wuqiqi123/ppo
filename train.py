import gymnasium as gym
from tqdm import tqdm
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from collections import deque
from torch.distributions import Categorical
from ppo.ema import EMA
from collections import deque, namedtuple
import numpy as np

from ppo.ppo import Actor, Critic


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Buffer = namedtuple('Buffer', ['states', 'actions', 'reward', 'next_state', 'action_log_prob', 'done'])

class ExperienceDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))


class PPOAgent:
    def __init__(self, env, 
                actor_hidden_dim = 64,
                critic_hidden_dim = 256,
                ppo_clip = 0.2,
                ppo_epochs = 4,
                ppo_batch_size = 64,
                ppo_lr = 3e-4,
                ppo_eps = 1e-5,
                gamma = 0.99,
                gae_lambda = 0.98,
                entropy_coef = 0.01):
        self.env = env
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.ppo_lr = ppo_lr
        self.ppo_eps = ppo_eps
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.gamma = gamma

        self.state_dim = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.actor = Actor(self.state_dim, self.num_actions, actor_hidden_dim).to(device)
        self.critic = Critic(self.state_dim, critic_hidden_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr = 3e-4, eps = 1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr = 3e-4, eps = 1e-5)
        self.target_critic = EMA(self.critic)
        self.target_critic.add_to_optimizer_post_step_hook(self.optimizer_critic)
        
    

    @torch.no_grad()
    def take_action(self, state):
        state = self.to_tensor(state)
        probs = self.actor(state)
        action_dist = Categorical(probs)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        return action.item(), action_log_prob


    def compute_advantage(gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)
    
    def to_tensor(self, data):
        return torch.tensor(data, device=device, dtype=torch.float)

    def optimize(self, replay_buffer):
        (states, actions, rewards, next_states, action_log_probs, dones) = zip(*replay_buffer)

        states = self.to_tensor(states)
        actions = self.to_tensor(actions)
        rewards = self.to_tensor(rewards)
        next_states = self.to_tensor(next_states)
        action_log_probs = self.to_tensor(action_log_probs)
        dones = self.to_tensor(dones)

        import pdb; pdb.set_trace()
        td_target = rewards + self.gamma * self.target_critic(next_states) * (1 - dones)

        td_delta = td_target - self.critic(states)
        advantages = self.compute_advantage(self.gamma, self.gae_lambda, td_delta)

        dataset = ExperienceDataset(zip(states, actions, rewards, next_states, action_log_probs, advantages, dones))

        dl = torch.utils.data.DataLoader(dataset, batch_size = self.ppo_batch_size, shuffle = True)
        for _ in range(self.ppo_epochs):
            for i, (states, actions, old_log_probs, rewards, old_values) in enumerate(dl):
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
    replay_buffer = deque([])
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc = 'episodes'):
        state, info = env.reset()
        for timestep in range(max_timesteps):
            
            done = False
            while not done:
                action, action_log_prob = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                buffer = Buffer(state, action, reward, next_state, action_log_prob, done)
                replay_buffer.append(buffer)
                state = next_state

            agent.optimize(replay_buffer)

    # Close the environment
    env.close()

