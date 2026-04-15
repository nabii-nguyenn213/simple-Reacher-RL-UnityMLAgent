from __future__ import annotations
import copy
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from models.actor_critic import ActorDoubleQCritic
from utils.logger import TensorboardLogger

@dataclass
class SACConfig:
    obs_dim: int
    act_dim: int

    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "ReLU"
    state_independent_log_std: bool = False
    init_log_std: float = -0.5

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    auto_alpha: bool = True
    alpha: float = 0.2
    target_entropy: float | None = None

    device: str = "auto"


class SACAgent:
    def __init__(self, config: SACConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)

        self.model = ActorDoubleQCritic(
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            state_independent_log_std=config.state_independent_log_std,
            init_log_std=config.init_log_std,
        ).to(self.device)

        self.target_q1 = copy.deepcopy(self.model.critic1).to(self.device)
        self.target_q2 = copy.deepcopy(self.model.critic2).to(self.device)

        for param in self.target_q1.parameters():
            param.requires_grad = False
        for param in self.target_q2.parameters():
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(
            self.model.actor.parameters(),
            lr=config.actor_lr,
        )

        critic_params = list(self.model.critic1.parameters()) + list(self.model.critic2.parameters())
        self.critic_optimizer = optim.Adam(
            critic_params,
            lr=config.critic_lr,
        )

        if config.target_entropy is None:
            target_entropy_value = -float(config.act_dim)
        else:
            target_entropy_value = float(config.target_entropy)

        self.target_entropy = torch.tensor(
            target_entropy_value,
            dtype=torch.float32,
            device=self.device,
        )

        self.auto_alpha = config.auto_alpha
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(config.alpha),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self._fixed_alpha = float(config.alpha)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @property
    def alpha(self) -> torch.Tensor:
        if self.auto_alpha:
            assert self.log_alpha is not None
            return self.log_alpha.exp()
        return torch.tensor(self._fixed_alpha, dtype=torch.float32, device=self.device)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            action, _, _ = self.model.act(obs_tensor, deterministic=deterministic)

        return action.squeeze(0).cpu().numpy()

    def update(
        self,
        replay_buffer,
        batch_size: int,
        logger: TensorboardLogger | None = None,
        step: int | None = None,
    ) -> dict[str, float]:
        batch = replay_buffer.sample(batch_size, self.device)

        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_observations"]
        dones = batch["dones"]

        alpha = self.alpha

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.model.actor.sample(
                next_obs,
                deterministic=False,
                squashed=True,
                with_logprob=True,
            )
            assert next_log_probs is not None

            target_q1 = self.target_q1(next_obs, next_actions)
            target_q2 = self.target_q2(next_obs, next_actions)
            target_q_min = torch.min(target_q1, target_q2)

            target_v = target_q_min - alpha * next_log_probs
            q_target = rewards + self.config.gamma * (1.0 - dones) * target_v

        current_q1 = self.model.critic1(obs, actions)
        current_q2 = self.model.critic2(obs, actions)

        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi_actions, log_probs, _ = self.model.actor.sample(
            obs,
            deterministic=False,
            squashed=True,
            with_logprob=True,
        )
        assert log_probs is not None

        q1_pi = self.model.critic1(obs, pi_actions)
        q2_pi = self.model.critic2(obs, pi_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (alpha * log_probs - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.auto_alpha:
            assert self.log_alpha is not None
            assert self.alpha_optimizer is not None

            entropy_term = (log_probs + self.target_entropy).detach()
            alpha_loss = -(self.log_alpha * entropy_term).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        self.soft_update(self.model.critic1, self.target_q1, self.config.tau)
        self.soft_update(self.model.critic2, self.target_q2, self.config.tau)

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q1_mean": float(current_q1.mean().item()),
            "q2_mean": float(current_q2.mean().item()),
            "reward_mean": float(rewards.mean().item()),
        }

        if logger is not None and step is not None:
            logger.log_scalars(metrics, step=step, prefix="sac/update")

        return metrics

    @torch.no_grad()
    def soft_update(
        self,
        source: torch.nn.Module,
        target: torch.nn.Module,
        tau: float,
    ) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - tau)
            tgt_param.data.add_(tau * src_param.data)

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "target_q1_state_dict": self.target_q1.state_dict(),
            "target_q2_state_dict": self.target_q2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "config": self.config.__dict__,
            "auto_alpha": self.auto_alpha,
        }

        if self.auto_alpha:
            assert self.log_alpha is not None
            assert self.alpha_optimizer is not None
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        else:
            checkpoint["fixed_alpha"] = self._fixed_alpha

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_q1.load_state_dict(checkpoint["target_q1_state_dict"])
        self.target_q2.load_state_dict(checkpoint["target_q2_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        if self.auto_alpha:
            assert self.log_alpha is not None
            assert self.alpha_optimizer is not None
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
