"""20-step FP32 planner for the initial-noise policy.

The frozen generator is the same SFT step-400 checkpoint used by the residual
planner, but the noise policy was trained with deterministic DDIM-20, FP32,
CFG 2.5 and no DiT cache.  This class keeps those settings explicit instead of
reusing the 50-step autocast planner.  It intentionally does not enable CUDA
graphs: the explicit-noise path must remain the same mapping that was audited
in training.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from omg.generation.architecture import validate_checkpoint_architecture_contract
from omg.noise.actor import HISTORY_DIM, NOISE_DIM, NoiseActor
from omg.noise.sampling import sample_action_noise
from omg.pipeline.schedule import HISTORY, HORIZON, motion


TRAIN_CAPTION = "A classical dancer kicks one leg forward."
DDIM_STEPS = 20
CFG_SCALE = 2.5


def load_noise_actor(path, device="cuda:0"):
    """Load only the actor tensors from a noise PPO ``policy.pt``.

    The checkpoint also contains the critic, optimizer state and curriculum;
    none of them is needed for deployment.  ``strict=True`` catches a wrong
    architecture before the service starts.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "actor" not in payload:
        raise ValueError(f"{path}: not a noise PPO checkpoint (missing 'actor')")
    actor = NoiseActor(observation_dim=HISTORY_DIM, action_dim=NOISE_DIM)
    actor.load_state_dict(payload["actor"], strict=True)
    actor.to(device).eval().requires_grad_(False)
    iteration = int(payload.get("iteration", -1))
    fingerprint = payload.get("fingerprint")
    return actor, iteration, fingerprint


class NoiseCheckpointPlanner:
    """Frozen SFT generator + frozen noise actor, native 50 Hz, DDIM-20."""

    def __init__(
        self,
        config,
        checkpoint,
        noise_checkpoint,
        *,
        device="cuda:0",
        seed=202609150,
        caption=TRAIN_CAPTION,
    ):
        self.config = OmegaConf.load(config)
        self.device = torch.device(device)
        self.model = (
            instantiate(self.config.model)
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.contract = validate_checkpoint_architecture_contract(payload, self.model)
        self.step = int(payload["global_step"])

        self.rep = self.model.representation
        stats = json.loads(Path(self.rep.stats_path).read_text(encoding="utf-8"))
        if stats.get("fps") != 50 or stats.get("sequence_length") != 100:
            raise ValueError("noise planner requires native50Hz seq100 statistics")
        if (self.rep.num_prev_states, self.rep.sequence_length) != (HISTORY, HORIZON):
            raise ValueError("noise planner requires history10/horizon100")
        if (
            self.model.use_audio
            or self.model.use_human_motion
            or self.model.diffusion_target != "future"
        ):
            raise ValueError("noise planner requires text-only future prediction")
        if self.model.diffusion.objective != "pred_x0":
            raise ValueError("noise planner requires pred_x0")
        if self.model.diffusion.ddim_eta != 0:
            raise ValueError("noise planner requires deterministic DDIM eta=0")
        if self.model.diffusion.cfg_scale != CFG_SCALE:
            raise ValueError(f"noise planner requires CFG={CFG_SCALE}")
        if self.model.diffusion.sample_timestep_map.numel() != DDIM_STEPS:
            raise ValueError(
                "noise planner requires a 20-step DDIM config; got "
                f"{self.model.diffusion.sample_timestep_map.numel()} steps"
            )

        # The training audit uses FP32 and disables TF32.  Keep the same
        # arithmetic mode so the explicit-noise mapping is the audited one.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.seed = int(seed)
        self.caption = str(caption)
        self.actor, self.actor_iteration, self.actor_fingerprint = load_noise_actor(
            noise_checkpoint, self.device
        )
        self.noise_checkpoint = str(noise_checkpoint)

    @property
    def ddim_steps(self):
        return int(self.model.diffusion.sample_timestep_map.numel())

    @torch.inference_mode()
    def plan(self, history, text, plan_id):
        start = time.perf_counter()
        prev = torch.from_numpy(motion(history, HISTORY)).to(self.device)[None]
        body = self.rep.kinematics.forward_body_positions(prev)
        features, pos, quat = self.rep.codec.prev_state_features_from_history(
            prev, body, None, fps=torch.tensor([50.0], device=self.device)
        )
        batch = {
            "history_features": features,
            # The policy is trained for one caption.  Keep the generator
            # conditioning identical to training even if the caller's prompt
            # string differs; the mismatch is logged in metadata.
            "caption": [self.caption],
            "canon_root_pos": pos,
            "canon_root_quat": quat,
        }
        obs = features.flatten(1)
        distribution = self.actor.distribution(obs)
        action, base, noise_seed = sample_action_noise(
            distribution, seed=self.seed, plan_id=plan_id, device=self.device
        )
        noise = action.reshape(1, HORIZON, self.rep.feat_dim)
        sample = self.model._sample_future_norm(
            batch, HORIZON, cfg_scale=CFG_SCALE, initial_noise=noise
        )
        qpos = (
            self.rep.compose_qpos_36(self.rep.decode(sample), pos, quat)[0]
            .float()
            .cpu()
            .numpy()
        )
        latency = time.perf_counter() - start
        return {
            "qpos": motion(qpos, HORIZON),
            "latency_s": latency,
            "noise_seed": noise_seed,
            "metadata": {
                "noise_policy": "initial_noise",
                "noise_policy_iteration": self.actor_iteration,
                "noise_seed": noise_seed,
                "noise_base_norm": float(base.norm()),
                "noise_mean_norm": float(distribution.mean.norm()),
                "noise_std_mean": float(distribution.std.mean()),
                "noise_action_norm": float(action.norm()),
                "caption": self.caption,
                "text_matches_caption": str(text) == self.caption,
                "ddim_steps": self.ddim_steps,
                "cfg_scale": CFG_SCALE,
                "generator_step": self.step,
            },
        }
