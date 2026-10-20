from types import SimpleNamespace
import pytest
import torch
import pytorch_lightning as pl
from omg.generation.models.motion_generator import MotionGenerator

@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))])
def test_diagnostic_transfer_preserves_bits_shapes_and_integer_precision(device):
    values = {
        'float': torch.tensor([float('inf'), -0., 1.234567], device=device),
        'double': torch.tensor([1.0000000000000002], dtype=torch.float64, device=device),
        'integer': torch.tensor([2**60+3, -2**60+5], dtype=torch.int64, device=device),
        'bool': torch.tensor([True, False], device=device),
        'noncontiguous': torch.arange(12., device=device).reshape(3, 4).T,
        'scalar': torch.tensor(3., device=device, requires_grad=True),
        'empty': torch.empty(0, 2, device=device),
    }
    result = MotionGenerator._diagnostic_tensors_to_cpu(values)
    for name, value in values.items():
        assert result[name].device.type == 'cpu'
        assert result[name].shape == value.shape
        assert result[name].dtype == value.dtype
        assert not result[name].requires_grad
        assert torch.equal(result[name].contiguous().reshape(-1).view(torch.uint8),
                           value.detach().cpu().contiguous().reshape(-1).view(torch.uint8))

def test_metric_packing_preserves_lightning_dtype_and_original_autograd_values():
    calls = []
    class Strategy:
        def reduce(self, value, reduce_op):
            assert reduce_op == 'mean'
            calls.append(value.clone())
            return value * 2
    model = object.__new__(MotionGenerator)
    pl.LightningModule.__init__(model)
    model._trainer = SimpleNamespace(strategy=Strategy())
    loss = torch.tensor(2., requires_grad=True)
    metrics = {'loss': loss, 'bf16': torch.tensor(3., dtype=torch.bfloat16),
               'count': torch.tensor(4), 'double': torch.tensor(5., dtype=torch.float64)}
    result = model._reduce_step_metrics(metrics)
    assert len(calls) == 2
    assert result['bf16'].dtype == result['count'].dtype == torch.float32
    assert result['double'].dtype == torch.float64
    assert all(result[name].item() == 2 * value.item() for name, value in metrics.items())
    assert loss.item() == 2 and loss.requires_grad
    assert all(not value.requires_grad for value in result.values())

@pytest.mark.parametrize('timesteps', [torch.tensor(2), torch.tensor([[0, 2], [4, 5]])])
def test_timestep_diagnostics_preserve_map_duplicate_and_clamp_semantics(timesteps):
    model = object.__new__(MotionGenerator)
    pl.LightningModule.__init__(model)
    model.diffusion = SimpleNamespace(train_timestep_map=torch.tensor([0, 2, 2, 4]),
        base_sqrt_alphas_cumprod=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        base_sqrt_one_minus_alphas_cumprod=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]))
    result = model._diffusion_timestep_diagnostics({'timesteps': timesteps})
    t = timesteps.reshape(1) if timesteps.ndim == 0 else timesteps
    assert result['timesteps'] == t.tolist()
    assert result['train_timestep_indices'] == [{0:0, 2:2, 4:3}.get(v) for v in t.flatten().tolist()]
    assert result['sqrt_alpha_cumprod'] == model.diffusion.base_sqrt_alphas_cumprod[t.clamp(0,4)].tolist()
    assert result['mean'] == t.float().mean().item()

def test_missing_timestep_diagnostics():
    model = object.__new__(MotionGenerator)
    pl.LightningModule.__init__(model)
    model.diffusion = SimpleNamespace()
    assert model._diffusion_timestep_diagnostics({}) is None
