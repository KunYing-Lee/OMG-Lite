from types import SimpleNamespace
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from omg.generation.architecture import apply_checkpoint_architecture_config, build_model_architecture_contract, MODEL_ARCHITECTURE_KEY
from omg.generation.export.onnx import _ieee_fp32_validation


def test_checkpoint_config_and_conflicts():
    m = SimpleNamespace(denoiser=SimpleNamespace(self_attention_qk_norm=True, cross_attention_qk_norm=True), history_pos_encoding="sinusoidal", frame_cond_injection="per_layer_film")
    c = {MODEL_ARCHITECTURE_KEY: build_model_architecture_contract(m)}
    cfg = OmegaConf.create({"denoiser": {"self_attention_qk_norm": False, "cross_attention_qk_norm": False}, "model": {"history_pos_encoding": "none", "frame_cond_injection": "per_layer_film"}})
    with pytest.raises(RuntimeError, match="conflicts"):
        apply_checkpoint_architecture_config(cfg, c, explicit_overrides=["model.history_pos_encoding=none"])
    apply_checkpoint_architecture_config(cfg, c)
    assert cfg.model.history_pos_encoding == "sinusoidal"
    assert cfg.denoiser.self_attention_qk_norm is True


def test_legacy_requires_both_declarations():
    cfg = OmegaConf.create({})
    with pytest.raises(RuntimeError, match="legacy-attention-contract"):
        apply_checkpoint_architecture_config(cfg, {})
    with pytest.raises(RuntimeError, match="history_pos_encoding"):
        apply_checkpoint_architecture_config(cfg, {}, legacy_attention_contract="none")


def test_fp32_precision_restored_on_error():
    old = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        @_ieee_fp32_validation
        def fail():
            assert torch.get_float32_matmul_precision() == "highest"
            raise ValueError("test")
        with pytest.raises(ValueError):
            fail()
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(old)
