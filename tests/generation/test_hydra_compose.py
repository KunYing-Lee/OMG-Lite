from pathlib import Path

from hydra import compose, initialize_config_dir


def test_compose_transformer():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["exp=base", "logger=none", "trainer=1gpu"])
    assert cfg.model._target_.endswith("MotionGenerator")
    assert cfg.denoiser._target_.endswith("MotionTransformerDenoiser")
    assert cfg.model.text_encoder.model_name == f"{cfg.paths.models_root}/t5-base-local"
    assert cfg.model.scheduler.type == "linear_warmup_cosine"
    assert cfg.model.scheduler.warmup_steps == 250


def test_compose_300m_diffusion_only():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["exp=300m", "logger=none", "trainer=8gpu"])
    assert cfg.denoiser._target_.endswith("MotionTransformerDenoiser")
    assert cfg.loss.simple_root_pos == 0.0
    assert cfg.loss.seam_body_pos == 0.0
    assert cfg.model.use_audio is False
    assert cfg.model.use_human_motion is False
    assert cfg.trainer.devices == 8
    assert cfg.trainer.strategy == "ddp_find_unused_parameters_true"


def test_compose_100m_native50_training():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=["exp=100m", "data=omg_data_50hz", "logger=none", "trainer=4gpu"],
        )
    dataset = cfg.data.dataset_opts.train.omg_lerobot_train
    assert cfg.model.use_audio is False
    assert cfg.model.use_human_motion is False
    assert dataset.use_text is True
    assert dataset.use_audio is False
    assert dataset.use_human_motion is False
    assert dataset.revision == "sha256:ff4243bdc4dd1fb242ca9419cd99080329300de55bb7b70a417ca5055e8a843e"
