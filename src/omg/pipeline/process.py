"""Keep planner CUDA execution outside the control process."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

_planner = None


def _initialize(config, checkpoint, seed):
    import torch

    from omg.pipeline.checkpoint import CheckpointPlanner

    global _planner
    torch.set_num_threads(1)
    _planner = CheckpointPlanner(config, checkpoint, seed=seed)


def _plan(history, text, plan_id):
    result = _planner.plan(history, text, plan_id)
    result["checkpoint_step"] = _planner.step
    return result


class ProcessPlanner:
    """Single persistent inference process; no CUDA state is forked."""

    def __init__(self, config, checkpoint, *, seed=1234):
        self.pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(config, checkpoint, seed),
        )
        self.step = None

    def plan(self, history, text, plan_id):
        result = self.pool.submit(_plan, history, text, plan_id).result()
        self.step = result["checkpoint_step"]
        return result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
