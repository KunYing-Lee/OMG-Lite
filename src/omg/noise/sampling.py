"""Deployment-side noise sampling helpers.

Kept separate from the planner so the sampling contract can be tested without
Hydra or a generated model.
"""

from __future__ import annotations

import torch

from omg.noise.actor import NOISE_DIM


def sample_action_noise(distribution, *, seed, plan_id, device):
    """Sample one 100x38 action from ``N(mean, std^2)``.

    The base normal variates are keyed by ``seed + plan_id`` so a replay of
    the same planner request is reproducible.  This is a deployment seed
    policy, not the training/evaluation paired-episode protocol.
    """
    device = torch.device(device)
    noise_seed = int(seed) + int(plan_id)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    base = torch.randn(
        (1, NOISE_DIM), device=device, dtype=torch.float32, generator=generator
    )
    action = distribution.mean + distribution.std * base
    return action, base, noise_seed
