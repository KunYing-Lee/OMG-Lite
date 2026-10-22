import numpy as np
import pytest

from omg.generation.diffusion.guided import GuidedDiffusion
from omg.pipeline.tensorrt import TensorRTPlanner, validate_metadata


def metadata(steps=20):
    d = GuidedDiffusion(test_timestep_respacing=str(steps))
    m = dict(format='omg.denoiser_step', diffusion_type='GuidedDiffusion',
             diffusion_target='future', objective='pred_x0', batch_size=2,
             feat_dim=38, sequence_length=100, num_prev_states=10, ddim_eta=0.)
    for key in ('sample_timestep_map', 'sample_alphas_cumprod', 'sample_alphas_cumprod_prev'):
        m[key] = getattr(d, key).numpy().tolist()
    return m


def test_native_contract():
    validate_metadata(metadata())
    for key, value in [('batch_size', 1), ('feat_dim', 125), ('ddim_eta', 1.), ('use_audio', True)]:
        with pytest.raises(ValueError):
            validate_metadata(dict(metadata(), **{key: value}))
    with pytest.raises(ValueError, match='20-step'):
        validate_metadata(metadata(50))
    truncated = metadata(50)
    for key in ('sample_timestep_map', 'sample_alphas_cumprod', 'sample_alphas_cumprod_prev'):
        truncated[key] = truncated[key][:20]
    with pytest.raises(ValueError, match='ordering'):
        validate_metadata(truncated)


def test_cache_preserves_ddim_updates_and_resets_per_plan():
    # A constant x0 predictor has an analytic final output: x0. Reusing its
    # prediction must not omit DDIM updates or leak cache state across plans.
    class Session:
        calls = 0
        def run(self, outputs, feeds):
            self.calls += 1
            assert feeds['x'].shape == (2, 100, 38)
            return [np.stack([np.full((100, 38), .2, np.float32),
                              np.full((100, 38), .1, np.float32)])]
    p = TensorRTPlanner.__new__(TensorRTPlanner)
    m = metadata()
    p.timesteps = np.array(m['sample_timestep_map'])
    p.alphas = np.array(m['sample_alphas_cumprod'])
    p.previous_alphas = np.array(m['sample_alphas_cumprod_prev'])
    p.cfg_scale, p.session = 2.5, Session()
    for _ in range(2):
        x, full, skipped = p._sample(np.zeros((1, 10, 38), np.float32), {}, np.random.default_rng(1234))
        np.testing.assert_allclose(x, .35, atol=1e-7)
        assert (full, skipped) == (9, 11)
    assert p.session.calls == 18
