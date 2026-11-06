"""History-conditioned distribution over the initial OMG diffusion noise.

This is the deployment-side copy of the training actor in
``omg_posttrain.learning.noise_ppo.policy``.  It has no dependency on the
training package: the state dict is loaded directly from the noise PPO
checkpoint.  The frozen generator is not differentiated through.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


HISTORY_DIM = 10 * 38
NOISE_DIM = 100 * 38


def joint_log_probability(action, mean, log_std):
    """Diagonal Gaussian joint log density over one 100x38 noise sample."""
    action, mean, log_std = action.detach().double(), mean.double(), log_std.double()
    return (
        -0.5 * ((action - mean) * (-log_std).exp()).square()
        - log_std
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)


@dataclass(frozen=True)
class JointDiagonalGaussian:
    mean: torch.Tensor
    log_std: torch.Tensor

    @property
    def std(self):
        return self.log_std.exp()

    def sample(self, generator=None, base_noise=None):
        """Draw a non-reparameterized action, optionally with fixed base noise."""
        if base_noise is not None:
            if generator is not None:
                raise ValueError("supply base_noise or generator, not both")
            if base_noise.shape != self.mean.shape:
                raise ValueError("fixed base noise must match the full action shape")
            noise = base_noise.detach().to(
                device=self.mean.device, dtype=self.mean.dtype
            )
        else:
            noise = torch.randn(
                self.mean.shape,
                device=self.mean.device,
                dtype=self.mean.dtype,
                generator=generator,
            )
        return (self.mean + self.std * noise).detach()

    def log_prob(self, action):
        if action.shape != self.mean.shape:
            raise ValueError("action shape must match the Gaussian mean")
        return joint_log_probability(action, self.mean, self.log_std)


def _backbone(observation_dim, hidden_dims):
    if observation_dim <= 0 or not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError("positive observation and hidden dimensions required")
    layers = []
    width = observation_dim
    for next_width in hidden_dims:
        layers.extend((nn.Linear(width, next_width, dtype=torch.float32), nn.SiLU()))
        width = next_width
    return nn.Sequential(*layers)


class NoiseActor(nn.Module):
    """Raw 10x38 history -> unbounded diagonal Gaussian over 100x38 noise.

    Zero final mean weights and ``log_std=0`` give exactly ``N(0, I)``,
    independently of the hidden-network initialization.  The trained
    checkpoint may move the mean; the standard deviation remains close to
    one, so deployment must sample from the distribution rather than use the
    mean as an action.
    """

    def __init__(
        self,
        observation_dim=HISTORY_DIM,
        action_dim=NOISE_DIM,
        hidden_dims=(512, 512),
    ):
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.backbone = _backbone(self.observation_dim, tuple(hidden_dims))
        self.mean_head = nn.Linear(hidden_dims[-1], self.action_dim, dtype=torch.float32)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32))

    def distribution(self, observation):
        if observation.ndim != 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"actor observation must be [B, {self.observation_dim}]"
            )
        mean = self.mean_head(self.backbone(observation.float()))
        return JointDiagonalGaussian(mean, self.log_std.expand_as(mean))

    def sample(self, observation, generator=None, base_noise=None):
        distribution = self.distribution(observation)
        action = distribution.sample(generator=generator, base_noise=base_noise)
        return (
            action,
            distribution.log_prob(action),
            distribution.mean.detach().clone(),
            distribution.log_std.detach().clone(),
        )
