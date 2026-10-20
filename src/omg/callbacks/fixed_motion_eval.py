"""Reproducible held-out diffusion and generation evaluation at sample milestones."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning import Callback

from omg.data.datamodule import motion_collate_fn


@contextmanager
def fixed_rng(seed: int, device: torch.device):
    """Isolate evaluation RNG without changing the subsequent training stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def fixed_indices(size: int, count: int, seed: int) -> list[int]:
    if not 0 < count <= size:
        raise ValueError(f"Invalid evaluation count {count} for dataset size {size}")
    return random.Random(seed).sample(range(size), count)


class FixedValidationRNG(Callback):
    def __init__(self, seed: int = 271828):
        self.seed = int(seed)
        self._context = None

    def on_validation_start(self, trainer, pl_module):
        self._context = fixed_rng(self.seed + trainer.global_rank, pl_module.device)
        self._context.__enter__()

    def on_validation_end(self, trainer, pl_module):
        self._restore()

    def on_exception(self, trainer, pl_module, exception):
        self._restore()

    def _restore(self):
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None


class FixedMotionEvaluation(Callback):
    def __init__(self, dirpath: str, milestone_steps: list[int],
                 expected_batches_per_epoch: int = 16477,
                 seed: int = 314159, loss_windows: int = 4096,
                 generation_windows: int = 64, batch_size: int = 16):
        self.dirpath = Path(dirpath)
        self.milestone_steps = [int(x) for x in milestone_steps]
        if self.milestone_steps != sorted(set(self.milestone_steps)):
            raise ValueError("Milestones must be sorted and unique")
        self.expected_batches_per_epoch = int(expected_batches_per_epoch)
        self.seed = int(seed)
        self.loss_windows = int(loss_windows)
        self.generation_windows = int(generation_windows)
        self.batch_size = int(batch_size)
        self.completed = []

    def state_dict(self):
        return {"completed": self.completed}

    def load_state_dict(self, state_dict):
        self.completed = list(state_dict["completed"])

    def on_train_start(self, trainer, pl_module):
        if trainer.num_training_batches != self.expected_batches_per_epoch:
            raise ValueError(f"Unexpected epoch length: {trainer.num_training_batches}")
        if not pl_module._motion_loss_disabled():
            raise ValueError("Fixed diffusion-loss protocol requires diffusion-only training")
        if len(trainer.datamodule.valsets) != 1:
            raise ValueError("Expected exactly one held-out dataset")
        world = trainer.world_size
        if self.loss_windows % (world * self.batch_size) or self.generation_windows % world:
            raise ValueError("Evaluation sample counts must divide evenly across ranks")
        dataset = trainer.datamodule.valsets[0]
        self.indices = fixed_indices(len(dataset), self.loss_windows, self.seed)
        if self.generation_windows > self.loss_windows:
            raise ValueError("Generation subset must fit inside loss subset")
        self.dirpath.mkdir(parents=True, exist_ok=True)
        if trainer.is_global_zero:
            manifest = {"split": "val", "dataset_windows": len(dataset),
                        "seed": self.seed, "world_size": world,
                        "loss_indices": self.indices,
                        "generation_indices": self.indices[:self.generation_windows],
                        "milestone_steps": self.milestone_steps,
                        "batches_per_epoch": self.expected_batches_per_epoch,
                        "start_global_step": int(trainer.global_step),
                        "optimizer_state_entries_at_start": len(trainer.optimizers[0].state),
                        "initial_lr": trainer.optimizers[0].param_groups[0]["lr"],
                        "batch_per_rank": trainer.train_dataloader.batch_size,
                        "workers_per_rank": trainer.train_dataloader.num_workers,
                        "pin_memory": trainer.train_dataloader.pin_memory,
                        "prefetch_factor": trainer.train_dataloader.prefetch_factor,
                        "generation_protocol": "configured DDIM sampler, 50 steps, CFG 2.5; fixed noise",
                        "generation_mse_note": "paired-reference diagnostic, not a semantic or physical quality score"}
            (self.dirpath / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @torch.no_grad()
    def evaluate(self, trainer, model, step: int):
        dataset = trainer.datamodule.valsets[0]
        rank, world = trainer.global_rank, trainer.world_size
        flags = [(m, m.training) for m in model.modules()]
        model.eval()
        report_dir = self.dirpath / f"step_{step:06d}"
        report_dir.mkdir(parents=True, exist_ok=True)
        totals = torch.zeros(4, device=model.device, dtype=torch.float64)
        try:
            with fixed_rng(self.seed + rank, model.device), trainer.precision_plugin.forward_context():
                local = self.indices[rank::world]
                for start in range(0, len(local), self.batch_size):
                    batch = motion_collate_fn([dataset[i] for i in local[start:start+self.batch_size]])
                    batch = model.transfer_batch_to_device(batch, model.device, 0)
                    target, valid, history_len = model._target_sequence(batch)
                    terms = model.diffusion.training_losses(model.denoiser, target,
                        model._conditions(batch), valid, history_len=history_len)
                    totals[0] += terms["diffusion_loss"].double() * len(target)
                    totals[1] += len(target)
            with fixed_rng(self.seed + 10000 + rank, model.device), trainer.precision_plugin.forward_context():
                indices = self.indices[:self.generation_windows][rank::world]
                batch = motion_collate_fn([dataset[i] for i in indices])
                batch = model.transfer_batch_to_device(batch, model.device, 0)
                target = model.representation.encode(batch)
                noise = torch.randn(target.shape, device=model.device, dtype=target.dtype)
                generated = model._sample_future_norm(batch, target.shape[1], cfg_scale=2.5, initial_noise=noise)
                valid = batch["mask"]["valid"].bool().unsqueeze(-1).expand_as(target)
                totals[2] = ((generated.float()-target.float()).square() * valid).sum().double()
                totals[3] = valid.sum()
                decoded = model.representation.decode(generated)
                qpos = model.representation.compose_qpos_36(decoded, batch["canon_root_pos"], batch["canon_root_quat"])
                finite = torch.tensor(int(torch.isfinite(qpos).all() and torch.isfinite(totals).all()), device=model.device)
                if dist.is_initialized():
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("Non-finite fixed evaluation")
                torch.save({"indices": indices, "caption": batch.get("caption", batch.get("text")),
                            "generated_features": model.representation.denormalize_features(generated).cpu(),
                            "target_features": batch["motion_features"].cpu(), "qpos36": qpos.cpu(),
                            "valid": batch["mask"]["valid"].cpu(), "fps": 50,
                            "initial_noise": noise.cpu(), "global_step": step},
                           report_dir / f"generation_rank{rank}.pt")
            if dist.is_initialized():
                dist.all_reduce(totals)
            values = totals.cpu().tolist()
            result = {"global_step": step, "epoch_fraction": step/self.expected_batches_per_epoch,
                      "diffusion_loss": values[0]/values[1], "loss_windows": int(values[1]),
                      "paired_generation_normalized_mse": values[2]/values[3],
                      "generation_windows": self.generation_windows, "seed": self.seed}
            if trainer.is_global_zero:
                (report_dir/"metrics.json").write_text(json.dumps(result, indent=2))
                if trainer.logger is not None:
                    trainer.logger.log_metrics({"fixed_eval/"+k:v for k,v in result.items()
                                            if k in ("diffusion_loss", "paired_generation_normalized_mse")}, step=step)
            return result
        finally:
            for module, training in flags:
                module.training = training

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step not in self.milestone_steps or step in self.completed:
            return
        self.evaluate(trainer, pl_module, step)
        self.completed.append(step)
        trainer.save_checkpoint(str(self.dirpath/f"epoch_{self.milestone_steps.index(step)+1}_step_{step:06d}.ckpt"))
