"""Native checkpoint inference; supports both 38D and 125D representations."""

import json
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from omg.generation.architecture import validate_checkpoint_architecture_contract
from omg.pipeline.schedule import HISTORY, HORIZON, motion


class CheckpointPlanner:
    def __init__(
        self,
        config,
        checkpoint,
        device="cuda:0",
        seed=1234,
        cfg_scale=2.5,
        cuda_graph=True,
    ):
        self.config = OmegaConf.load(config)
        self.model = (
            instantiate(self.config.model).to(device).eval().requires_grad_(False)
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.contract = validate_checkpoint_architecture_contract(payload, self.model)
        if cuda_graph and torch.device(device).type == "cuda":
            from omg.pipeline.cuda_graph import CUDAGraphDenoiser

            self.model.denoiser = CUDAGraphDenoiser(self.model.denoiser)
        self.step = int(payload["global_step"])
        self.device = torch.device(device)
        self.seed, self.cfg_scale = seed, cfg_scale
        self.rep = self.model.representation
        stats = json.loads(Path(self.rep.stats_path).read_text())
        if stats.get("fps") != 50 or stats.get("sequence_length") != 100:
            raise ValueError(
                "Checkpoint inference requires native50Hz seq100 statistics"
            )
        if (self.rep.num_prev_states, self.rep.sequence_length) != (HISTORY, HORIZON):
            raise ValueError(
                "Only 50Hz, history10, horizon100 checkpoints are supported"
            )
        if (
            self.model.use_audio
            or self.model.use_human_motion
            or self.model.diffusion_target != "future"
        ):
            raise ValueError("Pipeline requires text-only future prediction")
        if self.model.diffusion.sample_timestep_map.numel() != 50:
            raise ValueError("Pipeline requires 50 DDIM steps")

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
            "caption": [text],
            "canon_root_pos": pos,
            "canon_root_quat": quat,
        }
        rng = torch.Generator(device=self.device).manual_seed(self.seed + plan_id)
        noise = torch.randn(
            (1, HORIZON, self.rep.feat_dim), device=self.device, generator=rng
        )
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            sample = self.model._sample_future_norm(
                batch, HORIZON, cfg_scale=self.cfg_scale, initial_noise=noise
            )
            qpos = (
                self.rep.compose_qpos_36(self.rep.decode(sample), pos, quat)[0]
                .float()
                .cpu()
                .numpy()
            )
        return {
            "qpos": motion(qpos, HORIZON),
            "latency_s": time.perf_counter() - start,
            "noise_seed": self.seed + plan_id,
        }
