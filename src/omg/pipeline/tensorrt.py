"""Native 38D/50Hz deployment, ported from OMG 38afa98948fe.

The batch2 CFG ordering, NumPy DDIM arithmetic/RNG and pred_x0 cache match the
measured deployment backend. No trajectory blending or continuation warm-start.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from omg.generation.conditions.t5 import FrozenT5TextEncoder
from omg.generation.export import load_export_metadata
from omg.motion.representation import G1MotionRepresentation
from omg.pipeline.schedule import HISTORY, HORIZON, motion
from omg.runtime.onnx_providers import (
    DEFAULT_TENSORRT_ONNX_PROVIDERS,
    prepare_onnx_provider_runtime,
    validate_active_onnx_providers,
    validate_onnx_providers,
)


def validate_metadata(m):
    required = dict(format="omg.denoiser_step", diffusion_type="GuidedDiffusion",
                    diffusion_target="future", objective="pred_x0", batch_size=2,
                    feat_dim=38, sequence_length=100, num_prev_states=10)
    for key, value in required.items():
        if m.get(key) != value:
            raise ValueError(f"Deployment requires {key}={value}, got {m.get(key)}")
    if m.get("use_audio") or m.get("use_human_motion"):
        raise ValueError("Deployment requires text-only conditioning")
    if float(m.get("ddim_eta", -1)) != 0:
        raise ValueError("Deployment requires deterministic DDIM eta=0")
    arrays = [np.asarray(m[k]) for k in (
        "sample_timestep_map", "sample_alphas_cumprod", "sample_alphas_cumprod_prev")]
    if any(a.shape != (20,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Deployment requires 20-step metadata; re-export with diffusion.test_timestep_respacing=20")
    t, alpha, prev = arrays
    if not np.array_equal(t, np.round(np.linspace(0, 999, 20))) or not np.all(np.diff(alpha) < 0):
        raise ValueError("Invalid DDIM schedule ordering")
    if not np.all((alpha > 0) & (alpha < 1)) or not np.array_equal(prev, np.r_[1., alpha[:-1]]):
        raise ValueError("Invalid DDIM cumulative alphas")


def cosine_similarity(left, right):
    a = np.asarray(left, dtype=np.float64).reshape(left.shape[0], -1)
    b = np.asarray(right, dtype=np.float64).reshape(right.shape[0], -1)
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denominator > 0
    return float(np.mean(numerator[valid] / denominator[valid])) if np.any(valid) else 0.


class TensorRTPlanner:
    def __init__(self, onnx, *, metadata=None, text_encoder=None,
                 engine_cache=None, device="cuda:0", seed=1234, cfg_scale=2.5):
        import onnxruntime as ort

        self.metadata = m = load_export_metadata(onnx, metadata_path=metadata)
        validate_metadata(m)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TensorRT deployment requires a CUDA device")
        self.seed, self.cfg_scale = int(seed), float(cfg_scale)
        if not np.isfinite(self.cfg_scale):
            raise ValueError("CFG scale must be finite")
        providers = list(DEFAULT_TENSORRT_ONNX_PROVIDERS)
        validate_onnx_providers(providers, ort.get_available_providers())
        prepare_onnx_provider_runtime(providers)
        cache = Path(engine_cache or Path(onnx).parent / "trt_engine").resolve()
        cache.mkdir(parents=True, exist_ok=True)
        gpu = str(self.device.index or 0)
        options = {"device_id": gpu, "trt_fp16_enable": "True",
                   "trt_engine_cache_enable": "True", "trt_engine_cache_path": str(cache),
                   "trt_timing_cache_enable": "True", "trt_timing_cache_path": str(cache)}
        self.session = ort.InferenceSession(str(onnx), providers=providers,
                                           provider_options=[options, {"device_id": gpu}])
        validate_active_onnx_providers(providers, self.session.get_providers())
        self.session.disable_fallback()
        expected = {"x", "timesteps", "valid_mask", "history_features", "text_context", "text_mask"}
        if {i.name for i in self.session.get_inputs()} != expected:
            raise ValueError("Unexpected deployment ONNX input contract")
        shapes = {i.name: i.shape for i in self.session.get_inputs()}
        if shapes['x'] != [2, 100, 38]:
            raise ValueError("Deployment ONNX must have fixed batch2, horizon100, feat38")
        self.rep = G1MotionRepresentation(
            stats_path=m["stats_path"], kinematics_path=m["kinematics_path"],
            num_prev_states=HISTORY, canonical_frame_idx=int(m["canonical_frame_idx"]),
            feat_dim=38, sequence_length=HORIZON,
            rotation_representation=m["rotation_representation"],
        ).to(self.device).eval()
        stats = json.loads(Path(self.rep.stats_path).read_text())
        if stats.get("fps") != 50 or stats.get("sequence_length") != 100:
            raise ValueError("Deployment requires native50Hz seq100 statistics")
        encoder = text_encoder or m.get("text_encoder_model")
        if not encoder:
            raise ValueError("A T5 text encoder is required")
        self.text_encoder = FrozenT5TextEncoder(
            model_name=str(encoder), max_length=int(m["text_max_length"]),
            output_dim=int(m["text_dim"]),
        ).to(self.device).eval()
        self.text_cache = {}
        self.timesteps = np.asarray(m["sample_timestep_map"], dtype=np.int64)
        self.alphas = np.asarray(m["sample_alphas_cumprod"], dtype=np.float64)
        self.previous_alphas = np.asarray(m["sample_alphas_cumprod_prev"], dtype=np.float64)

    def _text(self, text):
        if text not in self.text_cache:
            has_text = torch.tensor([text != ""], dtype=torch.bool, device=self.device)
            branches = [self.text_encoder([text], has_text=has_text, force_null_text=null,
                                          device=self.device) for null in (False, True)]
            if len(self.text_cache) >= 128:
                del self.text_cache[next(iter(self.text_cache))]
            self.text_cache[text] = {
                "text_context": np.concatenate([b["context"].cpu().numpy().astype(np.float32) for b in branches]),
                "text_mask": np.concatenate([b["mask"].cpu().numpy().astype(bool) for b in branches]),
            }
        return self.text_cache[text]

    def _sample(self, history, text, rng):
        x = rng.standard_normal((1, HORIZON, 38)).astype(np.float32)
        previous = None
        similarity = 0.
        consecutive = full = skipped = 0
        for step in range(len(self.timesteps) - 1, -1, -1):
            if previous is not None and full >= 4 and consecutive < 2 and similarity >= .995:
                pred_x0 = previous
                consecutive += 1
                skipped += 1
            else:
                feeds = dict(text, x=np.repeat(x, 2, axis=0),
                             timesteps=np.full((2, HORIZON), int(self.timesteps[step]), dtype=np.int64),
                             valid_mask=np.ones((2, HORIZON), dtype=bool),
                             history_features=np.repeat(history, 2, axis=0))
                raw = self.session.run(["pred"], feeds)[0].astype(np.float32, copy=False)
                pred_x0 = raw[1:2] + self.cfg_scale * (raw[:1] - raw[1:2])
                if previous is not None:
                    similarity = cosine_similarity(pred_x0, previous)
                previous = pred_x0
                consecutive = 0
                full += 1
            alpha_bar = float(self.alphas[step])
            alpha_bar_prev = float(self.previous_alphas[step])
            eps = (x - np.sqrt(alpha_bar) * pred_x0) / max(np.sqrt(1. - alpha_bar), 1e-8)
            # Preserve the measured eta=0 sampler's arithmetic and RNG consumption.
            sigma = (0. * np.sqrt(max((1. - alpha_bar_prev) / max(1. - alpha_bar, 1e-8), 0.))
                     * np.sqrt(max(1. - alpha_bar / max(alpha_bar_prev, 1e-8), 0.)))
            mean_pred = (np.sqrt(alpha_bar_prev) * pred_x0
                         + np.sqrt(max(1. - alpha_bar_prev - sigma * sigma, 0.)) * eps)
            x = mean_pred + sigma * rng.standard_normal(x.shape).astype(np.float32) if step > 0 else mean_pred
            x = x.astype(np.float32, copy=False)
        return x, full, skipped

    @torch.inference_mode()
    def plan(self, history, text, plan_id):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        prev = torch.from_numpy(motion(history, HISTORY)).to(self.device)[None]
        conditioning = self._text(text)
        body = self.rep.kinematics.forward_body_positions(prev)
        features, pos, quat = self.rep.codec.prev_state_features_from_history(
            prev, body, None, fps=torch.tensor([50.], device=self.device))
        noise_seed = self.seed + plan_id
        sample, full, skipped = self._sample(features.cpu().numpy().astype(np.float32, copy=False),
                                             conditioning, np.random.default_rng(noise_seed))
        decoded = self.rep.decode(torch.from_numpy(sample).to(self.device))
        qpos = self.rep.compose_qpos_36(decoded, pos, quat)[0].cpu().numpy()
        torch.cuda.synchronize(self.device)
        return dict(qpos=motion(qpos, HORIZON), latency_s=time.perf_counter() - started,
                    noise_seed=noise_seed, metadata=dict(sampler="ddim", sampling_steps=20,
                    cfg_batch_size=2, dit_cache=True, dit_cache_threshold=.995,
                    dit_cache_warmup_steps=4, dit_cache_max_consecutive=2,
                    dit_cache_executed_steps=full, dit_cache_skipped_steps=skipped))
