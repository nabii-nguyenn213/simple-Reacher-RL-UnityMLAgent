import torch 
import torch.nn as nn
from torch.nn.modules import activation

from models.networks import ActorNetwork, VCriticNetwork, QCriticNetwork

class ActorVCritic(nn.Module): 
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256], activation="ReLU", 
                 state_independent_log_std = True, init_log_std = -0.5): 
        super().__init__()
        self.actor = ActorNetwork(obs_dim, act_dim, hidden_dims, activation, "Linear", 
                                  state_independent_log_std, init_log_std)
        self.critic = VCriticNetwork(obs_dim, hidden_dims, activation)

    def get_value(self, obs): 
        return self.critic(obs)

    def act(self, obs): 
        raw_action, log_prob, _ = self.actor.sample(obs, deterministic=False, squashed=False, with_logprob=True)
        value = self.critic(obs)
        env_action = torch.clamp(raw_action, -1.0, 1.0)
        return env_action, raw_action, log_prob, value

    def act_deterministic(self, obs): 
        raw_action, _, _ = self.actor.sample(obs, deterministic=True, squashed=False, with_logprob=False)
        value = self.critic(obs)
        env_action = torch.clamp(raw_action, -1.0, 1.0)
        return env_action, value 

    def evaluate_actions(self, obs, raw_action): 
        dist = self.actor.get_dist(obs)
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = self.critic(obs)

        return log_prob, entropy, value

class ActorDoubleQCritic(nn.Module): 
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256], activation="ReLU", 
                 state_independent_log_std = False, init_log_std=-0.5): 
        super().__init__()
        self.actor = ActorNetwork(obs_dim, act_dim, hidden_dims, activation, 
                                  "Linear", state_independent_log_std, init_log_std)
        self.critic1 = QCriticNetwork(obs_dim, act_dim, hidden_dims, activation)
        self.critic2 = QCriticNetwork(obs_dim, act_dim, hidden_dims, activation)

    def act(self, obs, deterministic=False): 
        action, log_prob, mean_action = self.actor.sample(obs, deterministic, squashed=True, with_logprob=True)
        return action, log_prob, mean_action

    def act_deterministic(self, obs): 
        action, _, mean_action = self.actor.sample(
            obs,
            deterministic=True,
            squashed=True,
            with_logprob=False,
        )
        return action, mean_action

    def get_q_values(self,obs, act):
        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        return q1, q2

    def evaluate_actor(self, obs: torch.Tensor):
        action, log_prob, _ = self.actor.sample(obs,deterministic=False,squashed=True,with_logprob=True)
        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        return action, log_prob, q1, q2
