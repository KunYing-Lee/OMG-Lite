"""Numerical oracle and sampling-contract tests for DPM-Solver++(2M)."""
import pytest
import torch

from omg.generation.diffusion.guided import GuidedDiffusion


class Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.states = []

    def forward(self, x, timesteps, conditions, valid_mask=None):
        self.calls.append(timesteps.clone())
        self.states.append(x.clone())
        return 0.2 * x.tanh() + timesteps[..., None] / 2000 + conditions['value']


def process(steps=8, **kwargs):
    return GuidedDiffusion(sampler='dpmpp_2m', test_timestep_respacing=str(steps), **kwargs)


@pytest.mark.parametrize('steps', [1, 2, 4, 8, 10, 20, 50])
@pytest.mark.parametrize('schedule', ['cosine', 'linear'])
def test_matches_diffusers_intermediate_states(steps, schedule):
    # Optional development dependency, never needed at inference.
    diffusers = pytest.importorskip('diffusers')
    d = process(steps, beta_schedule=schedule)
    oracle = diffusers.DPMSolverMultistepScheduler(
        num_train_timesteps=1000, prediction_type='sample', solver_order=2,
        algorithm_type='dpmsolver++', solver_type='midpoint', thresholding=False,
        lower_order_final=False, final_sigmas_type='zero',
    )
    oracle.set_timesteps(steps)
    # Deliberately use the same OMG grid, rather than compare differing schedules.
    oracle.timesteps = d.sample_timestep_map.flip(0)
    a = d.sample_alphas_cumprod.flip(0)
    oracle.sigmas = torch.cat([((1 - a) / a).sqrt(), torch.zeros(1)])
    noise = torch.randn(2, 7, 3, generator=torch.Generator().manual_seed(5))
    model = Denoiser()
    cond = {'value': torch.full((2, 1, 1), 0.3)}
    result = d.sample(model, tuple(noise.shape), cond, initial_noise=noise)
    x = noise.clone()
    reference = Denoiser()
    for i, t in enumerate(oracle.timesteps):
        torch.testing.assert_close(model.states[i], x, atol=3e-6, rtol=3e-5)
        output = reference(x, t.expand(2, 7), cond)
        x = oracle.step(output, t, x).prev_sample
    torch.testing.assert_close(result, x, atol=3e-6, rtol=3e-5)
    assert len(model.calls) == steps
    assert [int(t[0, 0]) for t in model.calls] == oracle.timesteps.tolist()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_cfg_masks_repeatability_and_batching(dtype):
    d = process()
    noise = torch.randn(2, 5, 3).to(dtype)
    saved = noise.clone()
    cond = {'value': torch.full((2, 1, 1), 0.3, dtype=dtype)}
    null = {'value': torch.zeros_like(cond['value'])}
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    kwargs = dict(valid_mask=mask, initial_noise=noise, null_conditions=null, cfg_scale=2.5)
    m = Denoiser()
    a = d.sample(m, tuple(noise.shape), cond, **kwargs)
    b = d.sample(Denoiser(), tuple(noise.shape), cond, **kwargs)
    assert len(m.calls) == 16
    assert a.dtype == dtype and torch.isfinite(a).all()
    assert torch.equal(a, b) and torch.equal(noise, saved)
    assert (a[~mask] == 0).all()
    for i in range(2):
        single = d.sample(Denoiser(), (1, 5, 3), {'value': cond['value'][i:i+1]},
                          valid_mask=mask[i:i+1], initial_noise=noise[i:i+1],
                          null_conditions={'value': null['value'][i:i+1]}, cfg_scale=2.5)
        torch.testing.assert_close(a[i:i+1], single, atol=0, rtol=0)
    branches = d.sample(Denoiser(), tuple(noise.shape), cond,
                        cfg_branches=[(cond, 2.5)], **kwargs)
    torch.testing.assert_close(a, branches, atol=0, rtol=0)


def test_default_ddim_is_unchanged_and_checkpoint_compatible():
    default = GuidedDiffusion()
    explicit = GuidedDiffusion(sampler='ddim')
    dpm = process(50)
    dpm.load_state_dict(default.state_dict(), strict=True)
    assert torch.equal(dpm.sample_timestep_map, default.sample_timestep_map)
    noise = torch.randn(2, 3, 4)
    cond = {'value': torch.zeros(2, 1, 1)}
    torch.manual_seed(99)
    a = default.sample(Denoiser(), tuple(noise.shape), cond, initial_noise=noise)
    torch.manual_seed(99)
    b = explicit.sample(Denoiser(), tuple(noise.shape), cond, initial_noise=noise)
    assert torch.equal(a, b)


@pytest.mark.parametrize('kwargs', [{'ddim_eta': 0.1}, {'zero_terminal_snr': True}])
def test_rejects_unsupported_settings(kwargs):
    with pytest.raises(ValueError):
        process(**kwargs)


def test_analytic_constant_x0_endpoint():
    class Constant(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return torch.full_like(x, 0.7)
    noise = torch.randn(2, 5, 3)
    actual = process().sample(Constant(), tuple(noise.shape), {'value': noise}, initial_noise=noise)
    torch.testing.assert_close(actual, torch.full_like(noise, 0.7))


def test_real_transformer_smoke():
    from omg.generation.denoisers.transformer import MotionTransformerDenoiser
    torch.manual_seed(7)
    model = MotionTransformerDenoiser(input_dim=38, hidden_dim=32, num_layers=1,
                                     num_heads=4, text_dim=32, dropout=0.0).eval()
    conditions = {'text_context': torch.randn(2, 3, 32),
                  'text_mask': torch.ones(2, 3, dtype=torch.bool)}
    result = process(4).sample(model, (2, 10, 38), conditions)
    assert result.shape == (2, 10, 38)
    assert torch.isfinite(result).all()
    assert not result.requires_grad


def test_single_step_starts_at_highest_noise():
    assert process(1).sample_timestep_map.tolist() == [999]
