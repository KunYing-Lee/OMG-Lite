from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from omg.motion.feature_codec import G1MotionFeatureCodec
from omg.robots.g1.kinematics import G1Kinematics
from omg.utils.rotation_conversions import axis_angle_to_quaternion


def pose_features():
    torch.manual_seed(24)
    kin = G1Kinematics()
    codec = G1MotionFeatureCodec(kin, rotation_representation="rot6d", include_body_positions=False)
    qpos = torch.cat([
        torch.randn(2, 3, 3) * 0.1,
        axis_angle_to_quaternion(torch.randn(2, 3, 3) * 0.2),
        torch.randn(2, 3, 29) * 0.1,
    ], dim=-1)
    fk = kin.forward_kinematics(qpos)
    anchor_pos, anchor_quat = qpos[:, :1, :3], qpos[:, :1, 3:7]
    components = codec.canonicalize(qpos, fk["body_pos_w"], fk["body_quat_w"], anchor_pos, anchor_quat, 50.)
    return codec, codec.assemble_features(components), components, qpos, anchor_pos, anchor_quat


def test_pose38_fk_matches_world_fk_after_canonical_transform():
    codec, features, original, qpos, pos, quat = pose_features()
    decoded = codec.split_features(features)
    assert features.shape == (2, 3, 38)
    torch.testing.assert_close(decoded.body_link_pos_local, original.body_link_pos_local, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(codec.decode_to_world_qpos36(decoded, pos, quat), qpos, atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError, match="Expected 38"):
        codec.split_features(torch.zeros(2, 3, 125))


def test_pose38_fk_gradient_matches_finite_differences():
    codec, features, *_ = pose_features()
    x = features[:1, :1].double().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda v: codec.split_features(v).body_link_pos_local,
        (x,), eps=1e-6, atol=1e-5, rtol=1e-4,
    )


def test_default_and_opt_in_fk_configuration_and_backward():
    config_dir = Path(__file__).resolve().parents[2] / "configs/generation"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        default = compose(config_name="train")
        enabled = compose(config_name="train", overrides=["loss=fk"])
    assert default.representation.feat_dim == default.denoiser.input_dim == 38
    assert default.representation.include_body_positions is False
    assert default.representation.rot6d_gradient_mode == "vanilla"
    assert all(v == 0 for k, v in default.loss.items() if k != "_target_")
    for split in ("train", "val", "test"):
        assert all(not cfg.include_body_positions for cfg in default.data.dataset_opts[split].values())
    representation = instantiate(enabled.representation, stats_path="assets/stats/g1_38d_stats.json")
    loss = instantiate(enabled.loss)
    codec, target, _, qpos, pos, quat = pose_features()
    pred = target.clone()
    pred[..., 0] += .1
    pred = pred.requires_grad_(True)
    fk = codec.kinematics.forward_kinematics(qpos)
    batch = {"mask": {"valid": torch.ones(2, 3, dtype=torch.bool)}, "fps": torch.full((2,), 50.),
             "canon_root_pos": pos, "canon_root_quat": quat, "qpos_36": qpos, **fk}
    result = loss(pred, target, batch, representation)
    torch.testing.assert_close(result["motion_loss"], torch.tensor(.01 / 3), atol=1e-7, rtol=1e-5)
    assert result["body_pos_consistency_loss"] == 0
    result["motion_loss"].backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[..., :3].norm() > 0
    assert pred.grad[..., 3:9].norm() > 0
    assert pred.grad[..., 9:].norm() > 0


def test_diffusion_only_step_does_not_decode_fk(monkeypatch):
    from omg.generation.models.motion_generator import MotionGenerator
    config_dir = Path(__file__).resolve().parents[2] / "configs/generation"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train")
    representation = instantiate(cfg.representation, stats_path="assets/stats/g1_38d_stats.json")
    denoiser = instantiate(cfg.denoiser, hidden_dim=32, num_layers=1, num_heads=4, text_dim=32)
    generator = MotionGenerator(
        representation, denoiser, instantiate(cfg.diffusion), instantiate(cfg.loss),
        condition_dim=32, history_mask_prob=0.,
    )
    _, features, *_ = pose_features()
    batch = {"B": 2, "motion_features": features, "history_features": features[:, :2],
             "mask": {"valid": torch.ones(2, 3, dtype=torch.bool)}}

    def forbidden(*args, **kwargs):
        raise AssertionError("diffusion-only training must not decode or run FK loss")

    monkeypatch.setattr(representation.codec, "split_features", forbidden)
    monkeypatch.setattr(generator.motion_loss, "forward", forbidden)
    logged = {}
    monkeypatch.setattr(generator, "log", lambda name, value, **kwargs: logged.update({name: value}))
    loss = generator._shared_step(batch, "val")
    torch.testing.assert_close(loss, logged["val/diffusion_loss"])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.norm() > 0 for p in denoiser.parameters())
    assert any(p.grad is not None and p.grad.norm() > 0 for p in generator.history_projector.parameters())


def test_stats_projection_preserves_matching_normalization(tmp_path):
    import json
    from omg.cli.generation.project_pose_stats import project_pose_stats
    source = Path("assets/stats/g1_125d_stats.json")
    output = tmp_path / "stats38.json"
    project_pose_stats(source, output)
    old, new = json.loads(source.read_text()), json.loads(output.read_text())
    assert new["mean"] == old["mean"][:38]
    assert new["std"] == old["std"][:38]
    assert new["sequence_length"] == old["sequence_length"]
    assert new["count"] == old["count"]
    with pytest.raises(FileExistsError):
        project_pose_stats(source, output)
