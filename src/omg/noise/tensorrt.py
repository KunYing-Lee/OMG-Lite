"""20-step TensorRT planner with an explicit initial-noise actor.

The OMG-Lite TensorRT planner already exposes the 20-step SFT denoiser and the
measured batch-2 CFG path.  This adapter keeps that engine but:

* takes the 100x38 initial noise from the noise-PPO actor instead of an RNG,
* disables the DiT step-skipping cache (the noise policy was trained without
  it), and
* runs exactly the 20 deterministic DDIM steps from the export metadata.

The result is the same ``plan(history, text, plan_id)`` interface as the
PyTorch noise planner, so the existing realtime server can serve either backend.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from omg.noise.planner import TRAIN_CAPTION, load_noise_actor
from omg.noise.sampling import sample_action_noise
from omg.pipeline.schedule import HISTORY, HORIZON, motion
from omg.pipeline.tensorrt import TensorRTPlanner


class NoiseTensorRTPlanner(TensorRTPlanner):
    """Frozen SFT generator (TensorRT, 20 steps) + frozen noise actor."""

    def __init__(
        self,
        onnx,
        *,
        noise_checkpoint,
        metadata=None,
        text_encoder=None,
        engine_cache=None,
        device="cuda:0",
        seed=202609150,
        caption=TRAIN_CAPTION,
        cfg_scale=2.5,
    ):
        super().__init__(
            onnx,
            metadata=metadata,
            text_encoder=text_encoder,
            engine_cache=engine_cache,
            device=device,
            seed=seed,
            cfg_scale=cfg_scale,
        )
        self.actor, self.actor_iteration, self.actor_fingerprint = load_noise_actor(
            noise_checkpoint, self.device
        )
        self.caption = str(caption)
        self.generator_step = -1
        self.ddim_steps = int(len(self.timesteps))

    def _sample_no_cache(self, history, conditioning, initial_noise):
        """Run the 20 DDIM steps without DiT cache and with explicit noise."""
        x = np.asarray(initial_noise, dtype=np.float32).reshape(1, HORIZON, 38)
        for step in range(len(self.timesteps) - 1, -1, -1):
            feeds = dict(
                conditioning,
                x=np.repeat(x, 2, axis=0),
                timesteps=np.full(
                    (2, HORIZON), int(self.timesteps[step]), dtype=np.int64
                ),
                valid_mask=np.ones((2, HORIZON), dtype=bool),
                history_features=np.repeat(history, 2, axis=0),
            )
            raw = self.session.run(["pred"], feeds)[0].astype(np.float32, copy=False)
            pred_x0 = raw[1:2] + self.cfg_scale * (raw[:1] - raw[1:2])
            alpha_bar = float(self.alphas[step])
            alpha_bar_prev = float(self.previous_alphas[step])
            eps = (x - np.sqrt(alpha_bar) * pred_x0) / max(
                np.sqrt(1.0 - alpha_bar), 1e-8
            )
            # The export uses deterministic DDIM (eta=0): sigma is exactly 0.
            mean_pred = np.sqrt(alpha_bar_prev) * pred_x0 + np.sqrt(
                max(1.0 - alpha_bar_prev, 0.0)
            ) * eps
            x = mean_pred.astype(np.float32, copy=False)
        return x

    @torch.inference_mode()
    def plan(self, history, text, plan_id):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        prev = torch.from_numpy(motion(history, HISTORY)).to(self.device)[None]
        conditioning = self._text(self.caption)
        body = self.rep.kinematics.forward_body_positions(prev)
        features, pos, quat = self.rep.codec.prev_state_features_from_history(
            prev, body, None, fps=torch.tensor([50.0], device=self.device)
        )
        distribution = self.actor.distribution(features.flatten(1))
        action, base, noise_seed = sample_action_noise(
            distribution, seed=self.seed, plan_id=plan_id, device=self.device
        )
        sample = self._sample_no_cache(
            features.cpu().numpy().astype(np.float32, copy=False),
            conditioning,
            action.cpu().numpy(),
        )
        decoded = self.rep.decode(torch.from_numpy(sample).to(self.device))
        qpos = self.rep.compose_qpos_36(decoded, pos, quat)[0].cpu().numpy()
        torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - started
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
                "cfg_scale": self.cfg_scale,
                "backend": "tensorrt",
                "dit_cache": False,
            },
        }
