import random
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from omg.callbacks.fixed_motion_eval import fixed_rng, fixed_indices
from omg.data.datamodule import GenerationDataModule


def test_fixed_rng_repeats_and_preserves_training_stream_even_on_exception():
    random.seed(17); np.random.seed(18); torch.manual_seed(19)
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    def draw(): return random.random(), np.random.rand(), torch.rand(3)
    with fixed_rng(314159, torch.device('cpu')): a = draw()
    with pytest.raises(RuntimeError):
        with fixed_rng(314159, torch.device('cpu')):
            b = draw()
            raise RuntimeError('exercise exception restoration')
    assert a[:2] == b[:2] and torch.equal(a[2], b[2])
    assert random.getstate() == state[0]
    assert np.array_equal(np.random.get_state()[1], state[1][1])
    assert torch.equal(torch.get_rng_state(), state[2])


def test_fixed_indices_unique_and_rank_partition_is_complete():
    ids = fixed_indices(12533543, 4096, 314159)
    assert ids == fixed_indices(12533543, 4096, 314159)
    assert len(set(ids)) == 4096
    rank_sets = [set(ids[r::8]) for r in range(8)]
    assert all(len(x) == 512 for x in rank_sets)
    assert set.union(*rank_sets) == set(ids)
    assert all(not (rank_sets[i] & rank_sets[j]) for i in range(8) for j in range(i))


def test_pin_memory_option_reaches_real_loader():
    dm = object.__new__(GenerationDataModule)
    dm.loader_opts = OmegaConf.create({'val': {'batch_size':2,'num_workers':0,'pin_memory':True}})
    loader = dm._make_loader(torch.utils.data.TensorDataset(torch.zeros(4, 1)), 'val', False)
    assert loader.pin_memory is True
    assert loader.drop_last is False


def test_three_epoch_schedule_preserves_peak_lr_and_reaches_floor():
    from types import SimpleNamespace
    from omg.generation.models.motion_generator import MotionGenerator
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-4)
    cfg = dict(type='linear_warmup_cosine', warmup_steps=250,
               warmup_start_factor=1e-4, t_max=49431, eta_min=1e-6)
    scheduler = MotionGenerator._build_lr_scheduler(SimpleNamespace(scheduler_cfg=cfg), opt)
    assert opt.param_groups[0]['lr'] == pytest.approx(1e-8)
    for step in range(1, 49432):
        opt.step(); scheduler.step()
        if step == 250: assert opt.param_groups[0]['lr'] == pytest.approx(1e-4)
    assert opt.param_groups[0]['lr'] == pytest.approx(1e-6)


@pytest.mark.parametrize('world', [1,2,3,8])
def test_balanced_shards_unequal_sizes_have_equal_length_no_duplicates(monkeypatch, world):
    from omg.data.datamodule import DistributedBalancedShardSampler
    class Unequal(torch.utils.data.Dataset):
        def __len__(self): return 113
        def materialized_shard_spans(self, base_index=0):
            return [(0,3),(3,57),(60,2),(62,31),(93,20)]
    dataset=Unequal()
    epochs=[]
    for epoch in [0,1]:
        per_rank=[]
        for rank in range(world):
            monkeypatch.setenv('WORLD_SIZE',str(world));monkeypatch.setenv('RANK',str(rank))
            sampler=DistributedBalancedShardSampler(dataset,seed=7,block_size=2,interleave_window=3)
            sampler.set_epoch(epoch);ids=list(sampler)
            assert len(ids)==len(sampler)==len(dataset)//world
            assert len(set(ids))==len(ids) and all(0<=x<len(dataset) for x in ids)
            assert ids==list(sampler)
            per_rank.append(ids)
        combined=sum(per_rank,[])
        assert len(set(combined))==len(combined)==(len(dataset)//world)*world
        epochs.append(combined)
    assert epochs[0]!=epochs[1]
