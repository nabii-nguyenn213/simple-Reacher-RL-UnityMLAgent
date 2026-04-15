import torch 
import torch.nn as nn 
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

activations = {
    "Linear"    : nn.Identity(), 
    "ReLU"      : nn.ReLU(), 
    "ELU"       : nn.ELU(), 
    "LeakyReLU" : nn.LeakyReLU(), 
    "Sigmoid"   : nn.Sigmoid(), 
    "Softmax-1" : nn.Softmax(dim=-1),
    "Softmax0"  : nn.Softmax(dim=0),
    "Softmax1"  : nn.Softmax(dim=1),
    "Softmax2"  : nn.Softmax(dim=2)
}

class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256], 
                 activation="ReLU", output_activation = "Linear"): 
        super().__init__()
        if hidden_dims: 
            dims = [input_dim, *hidden_dims, output_dim]
            layers = []
            for i in range(len(dims)-1): 
                act = activation if i+2!=len(dims) else output_activation 
                layers.append(nn.Linear(dims[i], dims[i+1], bias=True)) 
                layers.append(activations[act]) 
            self.mlp = nn.Sequential(*layers) 
        else: 
            self.mlp = nn.Sequential(nn.Linear(input_dim, output_dim), 
                                     activations[output_activation])

    def forward(self, x): 
        return self.mlp(x)

class ActorNetwork(nn.Module): 
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256], activation="ReLU", 
                 output_activation="Linear", state_independent_log_std = True, init_log_std=-0.5): 
        super().__init__()
        if len(hidden_dims) == 0: 
            raise ValueError("hidden_dims must not be empty")
        self.state_independent_log_std = state_independent_log_std
        self.backbone = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1], activation, output_activation)
        self.mu_head = nn.Linear(hidden_dims[-1], act_dim)
        if state_independent_log_std: 
            self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)
        else: 
            self.log_std_head = nn.Linear(hidden_dims[-1], act_dim)

    def forward(self, obs): 
        features = self.backbone(obs)
        mu = self.mu_head(features)
        if self.state_independent_log_std: 
            log_std = self.log_std.expand_as(mu)
        else: 
            log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def get_dist(self, obs): 
        mu, log_std = self(obs)
        std = log_std.exp()
        return Normal(mu, std)

    def sample(self, obs, deterministic=False, squashed=False, with_logprob=False): 
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mu, std)

        if deterministic: 
            # Use mean action 
            raw_action = mu 
        else: 
            # Sample from Gaussian
            raw_action = dist.rsample() # reparameterization trick
        if squashed: 
            # Tanh-squashed action (for SAC)
            action = torch.tanh(raw_action)
            mean_action = torch.tanh(mu)

            if with_logprob: 
                # Compute log probability
                log_prob = dist.log_prob(raw_action).sum(dim = -1, keepdim=True)
                # Tanh correlation term for SAC 
                log_prob -= torch.sum(torch.log(1.0 - action.pow(2) + 1e-6), dim = -1, keepdim=True)
            else: 
                log_prob = None 
        else: 
            # raw Gaussian action (for PPO)
            action = raw_action 
            mean_action = raw_action

            if with_logprob: 
                log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
            else: 
                log_prob = False 
        return action, log_prob, mean_action

class QCriticNetwork(nn.Module): 
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256], activation="ReLU"): 
        super().__init__()
        self.q_net = MLP(input_dim=obs_dim + act_dim, output_dim=1, 
                         hidden_dims=hidden_dims, activation=activation, 
                         output_activation="Linear")

    def forward(self, obs, act): 
        x = torch.cat([obs, act], dim = -1)
        return self.q_net(x)

class VCriticNetwork(nn.Module): 
    def __init__(self, obs_dim, hidden_dims=[256, 256], activation="ReLU"): 
        super().__init__()
        self.v_net = MLP(input_dim=obs_dim, output_dim=1, 
                         hidden_dims=hidden_dims, activation = activation, 
                         output_activation="Linear")

    def forward(self, obs): 
        return self.v_net(obs)
