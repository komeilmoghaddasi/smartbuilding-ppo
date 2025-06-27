# agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import params

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    def policy(self, x):
        x = self.forward(x)
        logits = self.policy_head(x)
        return F.softmax(logits, dim=-1)
    def value(self, x):
        x = self.forward(x)
        return self.value_head(x)

class PPOAgent:
    def __init__(self, state_dim, num_actions, params):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = params["gamma_discount"]
        self.gae_lambda = params["gae_lambda"]
        self.ppo_clip_eps = params["ppo_clip_eps"]
        self.ppo_epochs = params["ppo_epochs"]
        self.ppo_batch_size = params["ppo_batch_size"]
        self.buffer_capacity = params["buffer_capacity"]
        self.lr = params["learning_rate"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPOActorCritic(state_dim, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.reset_buffer()
    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.model.policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.model.value(state_tensor)
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        return action.item()
    def store_reward_done(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    def finish_episode(self):
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = self.rewards
        dones = self.dones
        values = torch.FloatTensor(self.values).to(self.device)
        # GAE-Lambda advantage and return calculation
        advantages = []
        gae = 0
        next_value = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i].item()
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i].item()
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + values
        # PPO update
        dataset_size = states.size(0)
        for _ in range(self.ppo_epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.ppo_batch_size):
                end = start + self.ppo_batch_size
                batch_idx = idxs[start:end]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                probs = self.model.policy(batch_states)
                dist = torch.distributions.Categorical(probs)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(batch_actions)
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.ppo_clip_eps, 1 + self.ppo_clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value = self.model.value(batch_states).squeeze()
                value_loss = F.mse_loss(value, batch_returns)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        self.reset_buffer()
    def get_state_dict(self):
        return self.model.state_dict()
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
