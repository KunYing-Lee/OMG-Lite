"""Unit tests for residual50.py (numpy only; no onnxruntime, no robot).

Run: python test_residual50.py
"""

import json
import pathlib

import numpy as np
import torch

import residual50 as r50

HERE = pathlib.Path(__file__).parent


# --------------------------------------------------------------------------- #
# reference implementation copied from the training repo (torch)
# --------------------------------------------------------------------------- #
def torch_bounded_residual(base, latent, lower, upper, scale, gain):
    from torch import as_tensor, broadcast_tensors, minimum, maximum, where, tanh, cat

    base = as_tensor(np.asarray(base, np.float32))
    latent = as_tensor(np.asarray(latent, np.float32))
    lower = as_tensor(np.asarray(lower, np.float32))
    upper = as_tensor(np.asarray(upper, np.float32))
    base_joints, latent = broadcast_tensors(base[..., 7:], latent)
    normalized = tanh(gain * latent)
    admissible_lower = minimum(lower, base_joints)
    admissible_upper = maximum(upper, base_joints)
    available = where(normalized >= 0.0, admissible_upper - base_joints, base_joints - admissible_lower)
    delta = scale * normalized * available
    candidate = cat((base[..., :7], base_joints + delta), dim=-1)
    return candidate.numpy(), delta.numpy()


def make_manifest(**overrides):
    manifest = dict(
        name="unit-test",
        onnx_file="ppo_actor.onnx",
        onnx_sha256="0" * 64,
        raw_width=3455,
        action_frames=25,
        future_frames=35,
        residual_period_frames=25,
        residual_scale=0.1,
        residual_latent_gain=0.3,
        action_clip=10.0,
        switch_seconds=0.5,
        prompt="A classical dancer kicks one leg forward.",
        joint_lower_limits=[-2.0] * 29,
        joint_upper_limits=[2.0] * 29,
    )
    manifest.update(overrides)
    return manifest


class FakeSession:
    """Mimics the exported ONNX graph; records feeds so tests can inspect the observation."""

    def __init__(self, manifest, latent_value=0.0):
        self.manifest = manifest
        self.latent_value = latent_value
        self.last_feeds = None

    def run(self, _names, feeds):
        self.last_feeds = {k: v.copy() for k, v in feeds.items()}
        base = feeds["base"]
        latent = np.full((base.shape[0], r50.ACTION_FRAMES * 29), self.latent_value, np.float32)
        corrected, delta = r50.bounded_joint_residual(
            base, latent.reshape(-1, r50.ACTION_FRAMES, 29),
            self.manifest["joint_lower_limits"], self.manifest["joint_upper_limits"],
            residual_scale=self.manifest["residual_scale"],
            residual_latent_gain=self.manifest["residual_latent_gain"])
        return latent, corrected, delta


def make_knot_manifest(**overrides):
    manifest = make_manifest(
        residual_knots=3,
        plan_frames=25,
        action_frames=3,
        residual_interpolation="cubic_hermite",
    )
    manifest.update(overrides)
    return manifest


class FakeKnotSession:
    """Knot actor: one latent output, no corrected/delta tensors."""

    def __init__(self, manifest, latent_value=0.0):
        self.manifest = manifest
        self.latent_value = latent_value
        self.last_feeds = None

    def run(self, _names, feeds):
        self.last_feeds = {k: v.copy() for k, v in feeds.items()}
        width = int(self.manifest["action_frames"]) * 29
        value = np.asarray(self.latent_value, np.float32)
        latent = (np.full((1, width), float(value), np.float32)
                  if value.ndim == 0 else np.broadcast_to(value, (1, width)).copy())
        return [latent]


def torch_hermite_knot(knots, anchor, slope, frames):
    """Reference port of omg_posttrain.policies.residual.decode_hermite_knot_latent."""
    positions = r50.knot_positions(frames, knots.shape[1])
    points = torch.cat((anchor[:, None], knots), dim=1)
    tangents = [slope]
    for index in range(1, knots.shape[1]):
        tangents.append((points[:, index + 1] - points[:, index - 1])
                        / (positions[index + 1] - positions[index - 1]))
    tangents.append((points[:, knots.shape[1]] - points[:, knots.shape[1] - 1])
                    / (positions[knots.shape[1]] - positions[knots.shape[1] - 1]))
    out = []
    interval = 0
    for t in range(frames):
        while not positions[interval] < t <= positions[interval + 1]:
            interval += 1
        left, right = positions[interval], positions[interval + 1]
        h = right - left
        u = (t - left) / h
        h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
        h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
        out.append(h00 * points[:, interval] + h10 * h * tangents[interval]
                   + h01 * points[:, interval + 1] + h11 * h * tangents[interval + 1])
    return torch.stack(out, dim=1), tangents[-1]


def make_ref(frames=200, seed=7):
    rng = np.random.default_rng(seed)
    ref = np.zeros((frames, 36), np.float32)
    ref[:, 2] = 0.78
    ref[:, 3] = 1.0
    ref[:, 7:] = rng.normal(0, 0.2, (frames, 29)).astype(np.float32)
    return ref


def make_state(ref, frame=0):
    return dict(
        joint_pos=ref[frame, 7:].copy(),
        joint_vel=np.zeros(29, np.float32),
        root_quat_wxyz=np.array([1, 0, 0, 0], np.float32),
        gyro=np.zeros(3, np.float32),
    )


def test_reference_block_matches_training_builder():
    golden = HERE / "golden_obs.npz"
    if not golden.exists():
        print("  golden_obs.npz absent (generated by make_golden.py); skipped")
        return
    data = np.load(golden)
    block = r50.reference_block(data["ref"][int(data["frame"]): int(data["frame"]) + 36], data["executed"][3:7])
    err = float(np.abs(block - data["expected"]).max())
    assert block.shape == (36, 39)
    assert err < 1e-6, err
    print(f"  reference block vs torch golden: max err {err:.2e}")


def test_observation_layout():
    preview = np.random.default_rng(0).normal(size=(36, 39)).astype(np.float32)
    queue = np.random.default_rng(9).normal(size=(36, 39)).astype(np.float32)
    history = np.random.default_rng(1).normal(size=(5, 122)).astype(np.float32)
    locked = (np.arange(36) < r50.LOCKED_FRAMES).astype(np.float32)
    obs = r50.build_observation(history, preview, locked, queue * locked[:, None])
    assert obs.shape == (1, 3455)
    assert np.array_equal(obs[0, :610], history.reshape(-1))
    assert np.array_equal(obs[0, 610:2014], preview.reshape(-1))
    assert np.array_equal(obs[0, 2014:3418], (queue * locked[:, None]).reshape(-1))
    assert np.array_equal(obs[0, 3418:3454], locked)
    assert obs[0, 3454] == r50.PHASE == 0.0
    print("  observation layout 610|1404|1404|36|1 OK")


def test_bounded_residual_matches_torch():
    rng = np.random.default_rng(2)
    for trial in range(6):
        base = np.zeros((1, 25, 36), np.float32)
        base[0, :, 2] = 0.75
        base[0, :, 3] = 1.0
        base[0, :, 7:] = rng.normal(0, 1.2, (25, 29))
        if trial == 5:
            base[0, 0, 7:12] = 2.5
        latent = rng.normal(0, 1.0, (1, 25 * 29)).astype(np.float32)
        lower = np.linspace(-2.5, -0.5, 29)
        upper = np.linspace(0.5, 2.5, 29)
        got_c, got_d = r50.bounded_joint_residual(
            base, latent.reshape(1, 25, 29), lower, upper,
            residual_scale=0.1, residual_latent_gain=0.3)
        ref_c, ref_d = torch_bounded_residual(base, latent.reshape(1, 25, 29), lower, upper, 0.1, 0.3)
        assert np.abs(got_c - ref_c).max() < 1e-6, np.abs(got_c - ref_c).max()
        assert np.abs(got_d - ref_d).max() < 1e-6
        before = np.abs(base[..., 7:] - np.clip(base[..., 7:], lower, upper))
        after = np.abs(got_c[..., 7:] - np.clip(got_c[..., 7:], lower, upper))
        assert np.all(after <= before + 1e-6)
    print("  bounded residual matches torch on 6 random windows (incl. out-of-limit base)")


def test_zero_latent_is_identity():
    base = np.random.default_rng(3).normal(size=(1, 25, 36)).astype(np.float32)
    base[0, :, 3] = 1.0
    corrected, delta = r50.bounded_joint_residual(
        base, np.zeros((1, 25, 29), np.float32),
        [-2.0] * 29, [2.0] * 29, residual_scale=0.1, residual_latent_gain=0.3)
    assert np.array_equal(corrected, base)
    assert np.all(delta == 0.0)
    print("  zero latent is an exact identity")


def test_decision_schedule():
    manifest = make_manifest()
    actor = r50.ResidualActor(manifest, FakeSession(manifest))
    corrector = r50.ResidualCorrector(actor)
    assert corrector.decision_due(0)
    assert not any(corrector.decision_due(f) for f in range(1, 14))
    assert corrector.decision_due(14)
    assert not any(corrector.decision_due(f) for f in range(15, 39))
    assert corrector.decision_due(39) and corrector.decision_due(64)
    print("  decision schedule: frame 0 and every switch-11 (14, 39, 64, ...)")


def test_corrector_contract():
    manifest = make_manifest()
    session = FakeSession(manifest, latent_value=1.0)
    actor = r50.ResidualActor(manifest, session)
    corrector = r50.ResidualCorrector(actor)
    raw = make_ref()
    tape = raw.copy()
    tape[13, 7:] += 0.01                      # simulate an earlier correction
    state = make_state(raw, 14)

    # non-decision frame: no actor call, history still updated
    assert corrector.step(frame=13, raw_reference=raw, corrected_tape=tape,
                          robot_state=make_state(raw, 13), last_action_onnx=np.zeros(29)) is None
    assert session.last_feeds is None

    correction = corrector.step(frame=14, raw_reference=raw, corrected_tape=tape,
                                robot_state=state, last_action_onnx=np.zeros(29))
    assert correction is not None and correction.start == 25       # frame + 11
    assert correction.joints_g1.shape == (25, 29) and correction.delta.shape == (25, 29)
    feeds = session.last_feeds
    assert np.array_equal(feeds["base"][0], raw[25:50])            # base = raw plan [frame+11, +25)
    obs = feeds["raw"][0]
    locked = obs[3418:3454]
    assert np.array_equal(locked, (np.arange(36) < 11).astype(np.float32))
    queue = obs[2014:3418].reshape(36, 39)
    tape_block = r50.reference_block(tape[14:50], state["root_quat_wxyz"])
    assert np.abs(queue[:11] - tape_block[:11]).max() < 1e-6       # queue comes from the TAPE
    assert np.all(queue[11:] == 0.0)
    preview = obs[610:2014].reshape(36, 39)
    raw_block = r50.reference_block(raw[14:50], state["root_quat_wxyz"])
    assert np.abs(preview - raw_block).max() < 1e-6                # preview comes from the RAW plan
    # previous residual = corrected_tape[13] - raw[13] injected into the last proprio sample
    assert np.allclose(obs[4 * 122 + 64: 4 * 122 + 93], 0.01, atol=1e-6)

    # reset decision: start 0, nothing locked, queue all zero
    corrector.reset()
    first = corrector.step(frame=0, raw_reference=raw, corrected_tape=raw,
                           robot_state=make_state(raw, 0), last_action_onnx=np.zeros(29))
    assert first.start == 0
    obs0 = session.last_feeds["raw"][0]
    assert np.all(obs0[3418:3454] == 0.0)
    assert np.all(obs0[2014:3418] == 0.0)
    print("  corrector: decision frame/start offsets, tape-vs-plan sources, flags, prev residual OK")


def test_short_reference_fails_closed():
    manifest = make_manifest()
    actor = r50.ResidualActor(manifest, FakeSession(manifest))
    corrector = r50.ResidualCorrector(actor)
    raw = make_ref(60)
    try:
        corrector.step(frame=30, raw_reference=raw, corrected_tape=raw,
                       robot_state=make_state(raw), last_action_onnx=np.zeros(29))
        raise AssertionError("expected StateNotReady")
    except r50.StateNotReady:
        pass
    print("  short reference fails closed")


def test_registry_and_manifest_validation():
    root = HERE / "_registry_tmp"
    (root / "model-a").mkdir(parents=True, exist_ok=True)
    (root / "model-a" / "actor-export.json").write_text(json.dumps(make_manifest(name="model-a")))
    (root / "bad").mkdir(exist_ok=True)
    (root / "bad" / "actor-export.json").write_text(json.dumps(make_manifest(name="bad", raw_width=1039)))
    registry = r50.ResidualRegistry.discover(root)
    assert "model-a" in registry.names() and "bad" not in registry.names()
    assert registry.resolve("off") is None
    assert registry.resolve("model-a").name == "model-a"
    assert registry.resolve("auto", "A classical dancer kicks one leg forward.").name == "model-a"
    assert registry.resolve("auto", "something else") is None
    try:
        registry.resolve("nope")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    print("  registry: discover/off/auto/name/invalid-entry OK")


def test_knot_decode_matches_torch():
    rng = np.random.default_rng(11)
    for knots in (3, 5, 25):
        k = rng.normal(size=(2, knots, 29)).astype(np.float32)
        anchor = rng.normal(size=(2, 29)).astype(np.float32)
        slope = rng.normal(size=(2, 29)).astype(np.float32)
        got, got_end = r50.decode_hermite_knot_latent(k, anchor, slope, 25)
        expected, expected_end = torch_hermite_knot(
            torch.from_numpy(k), torch.from_numpy(anchor), torch.from_numpy(slope), 25)
        assert np.abs(got - expected.numpy()).max() < 1e-6
        assert np.abs(got_end - expected_end.numpy()).max() < 1e-6
    print("  hermite knot decode matches torch (K=3/5/25)")


def test_knot_zero_is_identity_and_anchor_continues():
    manifest = make_knot_manifest()
    session = FakeKnotSession(manifest, latent_value=0.0)
    actor = r50.ResidualActor(manifest, session)
    corrector = r50.ResidualCorrector(actor)
    raw = make_ref()

    first = corrector.step(frame=0, raw_reference=raw, corrected_tape=raw,
                           robot_state=make_state(raw, 0), last_action_onnx=np.zeros(29))
    assert first is not None and first.start == 0
    assert first.latent.shape == (87,)
    assert first.joints_g1.shape == (25, 29) and first.delta.shape == (25, 29)
    assert np.array_equal(first.joints_g1, raw[0:25, 7:])
    assert np.all(first.delta == 0.0)
    assert np.array_equal(corrector._anchor, np.zeros(29, np.float32))
    assert np.array_equal(corrector._slope, np.zeros(29, np.float32))
    assert corrector._written_end == 25
    assert set(session.last_feeds) == {"raw"}          # no base feed for a knot actor

    second = corrector.step(frame=14, raw_reference=raw, corrected_tape=raw,
                            robot_state=make_state(raw, 14), last_action_onnx=np.zeros(29))
    assert second is not None and second.start == 25
    assert corrector._written_end == 50

    corrector._written_end = 26
    try:
        corrector.step(frame=14, raw_reference=raw, corrected_tape=raw,
                       robot_state=make_state(raw, 14), last_action_onnx=np.zeros(29))
        raise AssertionError("expected non-contiguous segment rejection")
    except RuntimeError as error:
        assert "previous segment ends" in str(error)
    print("  zero knots identity; anchor/slope state; non-contiguous segment rejected")


def test_knot_nonzero_updates_anchor_and_slope():
    manifest = make_knot_manifest()
    latent = np.linspace(0.0, 0.2, 87, dtype=np.float32)
    actor = r50.ResidualActor(manifest, FakeKnotSession(manifest, latent_value=latent))
    corrector = r50.ResidualCorrector(actor)
    raw = make_ref()
    corrector.step(frame=0, raw_reference=raw, corrected_tape=raw,
                   robot_state=make_state(raw, 0), last_action_onnx=np.zeros(29))
    assert np.abs(corrector._anchor).max() > 0.0
    assert np.abs(corrector._slope).max() > 0.0
    corrector.step(frame=14, raw_reference=raw, corrected_tape=raw,
                   robot_state=make_state(raw, 14), last_action_onnx=np.zeros(29))
    assert corrector._written_end == 50
    print("  knot actor updates anchor and end slope across segments")


def test_knot_manifest_validation():
    good = make_knot_manifest()
    path = HERE / "_knot_manifest.json"
    path.write_text(json.dumps(good))
    loaded = r50.load_manifest(path)
    assert (loaded["plan_frames"], loaded["action_frames"],
            loaded["residual_knots"], loaded["residual_interpolation"]) == (25, 3, 3, "cubic_hermite")
    for bad, message in (
        (make_knot_manifest(action_frames=25), "action_frames"),
        (make_knot_manifest(residual_interpolation="spline"), "residual_interpolation"),
        (make_knot_manifest(plan_frames=50), "plan_frames"),
    ):
        path.write_text(json.dumps(bad))
        try:
            r50.load_manifest(path)
            raise AssertionError("expected manifest rejection")
        except ValueError as error:
            assert message in str(error)
    path.unlink()
    print("  knot manifest validation (action_frames/plan_frames/interpolation)")


def main():
    print("residual50 tests")
    test_reference_block_matches_training_builder()
    test_observation_layout()
    test_bounded_residual_matches_torch()
    test_zero_latent_is_identity()
    test_decision_schedule()
    test_corrector_contract()
    test_short_reference_fails_closed()
    test_registry_and_manifest_validation()
    test_knot_decode_matches_torch()
    test_knot_zero_is_identity_and_anchor_continues()
    test_knot_nonzero_updates_anchor_and_slope()
    test_knot_manifest_validation()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
