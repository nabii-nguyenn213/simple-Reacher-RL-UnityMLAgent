from __future__ import annotations
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim) :
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.counter = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        self.observations[self.counter] = obs
        self.actions[self.counter] = action
        self.rewards[self.counter] = reward
        self.next_observations[self.counter] = next_obs
        self.dones[self.counter] = float(done)
        self.counter = (self.counter + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise RuntimeError(
                f"Not enough samples in ReplayBuffer. size={self.size}, batch_size={batch_size}"
            )
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "observations": torch.as_tensor(
                self.observations[indices],
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.as_tensor(
                self.actions[indices],
                dtype=torch.float32,
                device=device,
            ),
            "rewards": torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float32,
                device=device,
            ),
            "next_observations": torch.as_tensor(
                self.next_observations[indices],
                dtype=torch.float32,
                device=device,
            ),
            "dones": torch.as_tensor(
                self.dones[indices],
                dtype=torch.float32,
                device=device,
            ),
        }
        return batch
