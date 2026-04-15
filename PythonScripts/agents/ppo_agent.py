from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from models.actor_critic import ActorVCritic
from utils.logger import TensorboardLogger

@dataclass
class PPOConfig:
    obs_dim: int
    act_dim: int

    hidden_dims: tuple[int, ...] = (128, 128)
    activation: str = "ReLU"
    state_independent_log_std: bool = True
    init_log_std: float = -0.5

    lr: float = 3e-4
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    update_epochs: int = 10
    mini_batch_size: int = 64
    normalize_advantages: bool = True

    device: str = "auto"


class PPOAgent:
    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)

        self.model = ActorVCritic(
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            state_independent_log_std=config.state_independent_log_std,
            init_log_std=config.init_log_std,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def select_action(self, obs: np.ndarray):
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            env_action, raw_action, log_prob, value = self.model.act(obs_tensor)

        assert log_prob is not None

        return (
            env_action.squeeze(0).cpu().numpy(),
            raw_action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def select_action_deterministic(self, obs: np.ndarray):
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            env_action, value = self.model.act_deterministic(obs_tensor)

        return env_action.squeeze(0).cpu().numpy(), float(value.item())

    def update(
        self,
        rollout_buffer,
        logger: TensorboardLogger | None = None,
        step: int | None = None,
    ) -> dict[str, float]:
        data = rollout_buffer.get(self.device)

        observations = data["observations"]
        raw_actions = data["actions"]
        old_log_probs = data["log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]

        if old_log_probs.dim() > 1:
            old_log_probs = old_log_probs.squeeze(-1)
        if advantages.dim() > 1:
            advantages = advantages.squeeze(-1)
        if returns.dim() > 1:
            returns = returns.squeeze(-1)

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = observations.shape[0]
        batch_size = min(self.config.mini_batch_size, num_samples)

        last_metrics: dict[str, float] = {}

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(num_samples, device=self.device)

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_obs = observations[batch_idx]
                batch_raw_actions = raw_actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                new_log_probs, entropy, values = self.model.evaluate_actions(
                    batch_obs,
                    batch_raw_actions,
                )

                new_log_probs = new_log_probs.squeeze(-1)
                entropy = entropy.squeeze(-1)
                values = values.squeeze(-1)

                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                ) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                clip_fraction = (
                    ((ratio - 1.0).abs() > self.config.clip_coef)
                    .float()
                    .mean()
                    .item()
                )

                last_metrics = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy_bonus.item()),
                    "approx_kl": float(approx_kl),
                    "clip_fraction": float(clip_fraction),
                }

        if logger is not None and step is not None:
            logger.log_scalars(last_metrics, step=step, prefix="ppo/update")

        return last_metrics

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
