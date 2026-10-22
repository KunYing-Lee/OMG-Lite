"""Paired native50 checkpoint sampling benchmark (no robot commands).

Uses identical history/text/initial-noise seeds. Errors relative to DDIM50
measure output differences, NOT ground-truth quality or tracking success.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from omg.pipeline.checkpoint import CheckpointPlanner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'checkpoint', 'history', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--text', default='walk forward')
    p.add_argument('--steps', type=int, nargs='+', default=[20, 10, 8, 4])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--cfg-scale', type=float, default=2.5)
    args = p.parse_args()
    if args.repeats < 1 or any(n < 1 or n > 1000 for n in args.steps):
        p.error('positive repeats and steps in [1, 1000] are required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = np.load(args.history)['qpos_36'][-10:]
    planner = CheckpointPlanner(args.config, args.checkpoint, device=args.device,
                                seed=args.seed, cfg_scale=args.cfg_scale, cuda_graph=False)
    cfg = OmegaConf.to_container(planner.config.model.diffusion, resolve=True)
    records, arrays, baseline = [], {}, None
    variants = [('ddim', 50)] + [(s, n) for n in args.steps for s in ['ddim', 'dpmpp_2m']]
    for sampler, steps in variants:
        variant_cfg = dict(cfg, sampler=sampler, test_timestep_respacing=str(steps), ddim_eta=0.0)
        planner.model.diffusion = instantiate(variant_cfg).to(args.device)
        planner.plan(history, args.text, 0)  # Warm T5/model, exclude from latency.
        times, plans = [], []
        for i in range(args.repeats):
            if planner.device.type == 'cuda':
                torch.cuda.synchronize(planner.device)
            start = time.perf_counter()
            result = planner.plan(history, args.text, i)
            if planner.device.type == 'cuda':
                torch.cuda.synchronize(planner.device)
            times.append(time.perf_counter() - start)
            plans.append(result['qpos'])
        values = np.stack(plans)
        if not np.isfinite(values).all():
            raise RuntimeError(f'Non-finite output: {sampler} {steps}')
        if baseline is None:
            baseline = values
        arrays[f'{sampler}_{steps}'] = values
        records.append(dict(sampler=sampler, steps=steps, latency_s=times,
                            median_latency_s=float(np.median(times)),
                            timesteps=planner.model.diffusion.sample_timestep_map.cpu().tolist(),
                            qpos_rmse_vs_ddim50=float(np.sqrt(np.mean((values-baseline)**2)))))
    np.savez_compressed(output / 'plans.npz', history=history, **arrays)
    report = dict(config=str(Path(args.config).resolve()), checkpoint=str(Path(args.checkpoint).resolve()),
                  checkpoint_step=planner.step, seed=args.seed, text=args.text, cfg_scale=args.cfg_scale,
                  device=args.device, torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(planner.device) if planner.device.type == 'cuda' else None,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  records=records, note='Output differences are not quality/tracking metrics. No closed-loop test.')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
