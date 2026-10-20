from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from omg.cli.generation.train import _trainer_config


GENERATION_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "generation"


@pytest.mark.parametrize(
    ("exp_name", "train_batch_size", "accumulate_grad_batches"),
    [
        ("50m", 192, 1),
        ("100m", 192, 1),
        ("300m", 128, 1),
        ("500m", 64, 3),
        ("1b", 32, 6),
    ],
)
def test_experiment_presets_share_omnimodal_sinusoidal_training_protocol(
    exp_name: str,
    train_batch_size: int,
    accumulate_grad_batches: int,
):
    with initialize_config_dir(config_dir=str(GENERATION_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=[f"exp={exp_name}"])

    assert all(float(value) == 0.0 for key, value in cfg.loss.items() if key != "_target_")
    assert cfg.model.history_pos_encoding == "sinusoidal"
    assert cfg.model.use_audio is True
    assert cfg.model.use_human_motion is True
    assert cfg.denoiser.self_attention_qk_norm is True
    assert cfg.denoiser.cross_attention_qk_norm is True
    assert cfg.trainer.devices == 4
    assert cfg.trainer.max_steps == 180000
    assert cfg.trainer.accumulate_grad_batches == accumulate_grad_batches
    assert cfg.model.optimizer.lr == 1.0e-4
    assert cfg.model.scheduler.t_max == 180000
    assert cfg.data.loader_opts.train.batch_size == train_batch_size
    assert cfg.data.loader_opts.train.num_workers == 2
    assert cfg.data.loader_opts.train.prefetch_factor == 2
    assert list(cfg.data.dataset_opts.train) == ["omg_data_materialized_omnimodal_train"]


def test_trainer_config_disables_sampler_replacement_for_custom_train_sampler():
    cfg = OmegaConf.create({"trainer": {"accelerator": "gpu", "devices": 8, "strategy": "ddp"}})
    datamodule = SimpleNamespace(train_sampler=object())

    trainer_cfg = _trainer_config(cfg, datamodule)

    assert trainer_cfg["use_distributed_sampler"] is False


def test_trainer_config_preserves_explicit_sampler_replacement_setting():
    cfg = OmegaConf.create(
        {"trainer": {"accelerator": "gpu", "devices": 8, "strategy": "ddp", "use_distributed_sampler": True}}
    )
    datamodule = SimpleNamespace(train_sampler=object())

    trainer_cfg = _trainer_config(cfg, datamodule)

    assert trainer_cfg["use_distributed_sampler"] is True


@pytest.mark.parametrize("preset", ["base", "50m_50hz_text_3epochs"])
def test_50hz_default_matches_three_epoch_protocol(preset):
    with initialize_config_dir(config_dir=str(GENERATION_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=[f"exp={preset}"])
    OmegaConf.to_container(cfg, resolve=True)
    assert cfg.ckpt_path is None and cfg.init_weights_only_ckpt is None
    assert cfg.seed == 1234
    assert (cfg.representation.sequence_length, cfg.representation.num_prev_states,
            cfg.representation.canonical_frame_idx) == (100, 10, 9)
    assert (cfg.denoiser.num_layers, cfg.denoiser.hidden_dim, cfg.denoiser.num_heads) == (10, 512, 8)
    assert cfg.model.history_pos_encoding == "sinusoidal"
    assert cfg.denoiser.self_attention_qk_norm and cfg.denoiser.cross_attention_qk_norm
    assert not cfg.model.use_audio and not cfg.model.use_human_motion
    assert all(v == 0 for k, v in cfg.loss.items() if k != "_target_")
    assert cfg.trainer.devices * cfg.data.loader_opts.train.batch_size == 6144
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.data.loader_opts.train.num_workers == 4
    assert cfg.data.loader_opts.train.pin_memory
    assert cfg.data.loader_opts.train.prefetch_factor == 2
    assert cfg.data.train_sampler_policy == "balanced_shards"
    assert cfg.model.optimizer.lr == 1e-4
    assert cfg.model.scheduler.warmup_steps == 250
    assert cfg.trainer.max_steps == cfg.model.scheduler.t_max == 49431
    assert cfg.trainer.max_epochs == 3
    assert cfg.trainer.val_check_interval == cfg.callbacks.checkpoint.every_n_train_steps == 250
    assert list(cfg.callbacks.fixed_motion_evaluation.milestone_steps) == [16477, 32954, 49431]
    assert cfg.diffusion.test_timestep_respacing == "50"
    assert cfg.diffusion.cfg_scale == 2.5
    assert cfg.logger.enabled
