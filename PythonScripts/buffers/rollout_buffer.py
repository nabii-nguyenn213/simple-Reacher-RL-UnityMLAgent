import numpy as np 
import torch

class RolloutBuffer: 
    def __init__(self, buffer_size, obs_dim, act_dim, gamma=0.99, gae_lambda=0.95): 
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.counter = 0 

    def reset(self): 
        self.counter = 0 

    def add(self, obs, raw_action, reward, done, log_prob, value): 
        if self.counter >= self.buffer_size: 
            raise ValueError("RolloutBuffer is full. Call reset() before adding more data.")
        self.obs[self.counter] = obs 
        self.actions[self.counter] = raw_action
        self.rewards[self.counter] = reward
        self.dones[self.counter] = float(done)
        self.log_probs[self.counter] = log_prob
        self.values[self.counter] = value

        self.counter += 1

    def compute_returns_and_advantages(self, last_value, last_done):
        gae = 0.0 
        for i in reversed(range(self.counter)): 
            if i == self.counter - 1: 
                next_non_terminal = 1.0 - float(last_done) 
                next_value = last_value
            else: 
                next_non_terminal = 1.0 - self.dones[i + 1] 
                next_value = self.values[i + 1] 
            delta = self.rewards[i] + self.gamma * next_value * next_non_terminal - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae 
            self.advantages[i] = gae 
        self.returns[:self.counter] = self.advantages[:self.counter] + self.values[:self.counter]

    def get(self, device): 
        if self.counter == 0: 
            raise ValueError("RolloutBuffer is empty")
        data = {"observations": torch.as_tensor(self.obs[:self.counter], dtype=torch.float32, device=device), 
                "actions": torch.as_tensor(self.actions[:self.counter], dtype=torch.float32, device=device), 
                "log_probs": torch.as_tensor(self.log_probs[:self.counter], dtype=torch.float32, device=device), 
                "values": torch.as_tensor(self.values[:self.counter], dtype=torch.float32, device=device), 
                "advantages": torch.as_tensor(self.advantages[:self.counter], dtype=torch.float32, device=device), 
                "returns": torch.as_tensor(self.returns[:self.counter], dtype=torch.float32, device=device), 
                }
        return data

