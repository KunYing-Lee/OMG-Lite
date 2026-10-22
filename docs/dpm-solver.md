# DPM-Solver++ sampling

`GuidedDiffusion` supports the opt-in `sampler=dpmpp_2m`: deterministic
second-order multistep midpoint DPM-Solver++, with first-order startup and a
final x0 prediction at zero noise. Existing configurations still default to
DDIM. No checkpoint weights or training losses change. CFG, modality branches,
valid-frame masks and explicit initial noise use the existing interfaces.

Use the normal generation CLI with your checkpoint/experiment and add:

```bash
python -m omg.cli.generation.generate \
  --ckpt_path /path/to/model.ckpt --exp YOUR_EXPERIMENT \
  --text 'walk forward' --seed 1234 \
  diffusion.sampler=dpmpp_2m diffusion.test_timestep_respacing=8
```

The existing integer respacing grid is retained (including endpoints for counts
above one); a single DPM step starts at the highest training noise. This is not
the default Diffusers timestep grid. The DPM branch reads cumulative alphas
from the original schedule: reconstructing them via float32 respaced betas can
round a large cosine jump to beta=1 at 2/4 steps. Coefficients are evaluated
outside low precision, and solver state uses at least float32. It does not
clip/threshold motion features. `ddim_eta != 0` and `zero_terminal_snr=True`
are explicitly rejected for this initial DPM implementation.

The native deployment contract continues to require DDIM50. The following
research-only runner loads that contract/checkpoint first, then swaps the
sampler for offline comparisons without sending robot commands:

```bash
PYTHONPATH=src python scripts/validation/sampler_comparison.py \
  --config /path/to/saved-config.yaml \
  --checkpoint /path/to/model.ckpt \
  --history /path/to/history.npz \
  --output outputs/sampler-comparison \
  --device cuda:0 --steps 20 10 8 4 --repeats 10
```

The history NPZ must contain `qpos_36` with at least ten frames. Run in the
normal Mars5090 project environment with the checkpoint's statistics and local
T5 assets accessible. The runner retains paired noise seeds, full qpos plans,
actual timesteps, checkpoint training step, GPU name and synchronized wall-clock
latencies. Warmup is excluded. Latency includes planning/decoding and CPU output
transfer; this is not a denoiser-only microbenchmark. CUDA graphs are disabled.
RMSE against DDIM50 measures output disagreement only, not motion quality.
Closed-loop tracking/failure/contact evaluation remains a separate experiment.

## Validation

CPU test environment: Python 3.10, PyTorch 2.6.0+cpu. The reference comparison
uses Diffusers 0.35.1 (development-only, not a new runtime dependency):

```bash
python -m pip install pytest diffusers==0.35.1
PYTHONPATH=src python -m pytest -q \
  tests/generation/test_dpmpp.py \
  tests/generation/test_explicit_sampling_noise.py \
  tests/generation/test_guided_diffusion_weights.py
```

Tests compare every intermediate state and final output against Diffusers on
the identical grid at 1/2/4/8/10/20/50 steps for linear and cosine schedules.
They also cover CFG/modality branches, masks, deterministic replay, batching,
fp16/bfloat16 finite outputs, analytic constant-x0 behavior, timestep mapping,
strict state-dict loading and a real small MotionTransformer forward pass.
The oracle tests skip if Diffusers is absent; install it to run the full check.

Mars5090 validation has **not** run: the current execution environment cannot
route to its Tailscale address (`Network is unreachable`). CPU tests are not
GPU/checkpoint/closed-loop acceptance.

References:
- https://github.com/LuChengTHU/dpm-solver
- https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver
