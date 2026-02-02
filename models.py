from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5


def safe_torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class RunningNorm:
    def __init__(self, shape, eps: float = 1e-5):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        self.perception = nn.Sequential(*layers)

        self.actor_mean = nn.Linear(last, act_dim)
        self.critic = nn.Linear(last, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.perception(obs)
        mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        return action, logp, value

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return logp, entropy, value
