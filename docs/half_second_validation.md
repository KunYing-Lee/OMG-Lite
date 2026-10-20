# Half-second switching validation — 2026-10-20

Isolated branch `codex/native50-half-second`, based on `aa5acd2`.
The ongoing G1 deployment checkout and Mars5090 were not modified.

## Schedule

Native 50Hz; generation horizon 100 frames; history 10 frames; 50-step DDIM unchanged.
Each 25-frame segment requests at phase 0 using raw frames 15:25, and switches at phase 25.
Tracker preview reaches +10 and velocity features additionally reach +1, requiring acceptance
before consuming phase 14 (280 ms after request). Initialization still generates the first plan first.
The same constants drive sync simulation, async simulation, and deployment requests.
There is no added smoothing, interpolation, seam alignment, or late-result splice.

## Reproducibility and scope

Eight 500-frame sync MuJoCo runs on shared A800 GPU 7: four prompts × two schedules.
Baseline is an immutable archive of `aa5acd2`, not the live deployment checkout.
Checkpoint: `epoch_3_step_049431.ckpt`, config `three_epoch_validation/formal_config.json`,
tracker HoloMotion v1.4.1 `model_16200_dynamic_batch.onnx`, seed motion `seed_stand.npz`.
All assets are under `/wuji-vefps-D/wuji-il/leekunying` on the A800 host.
The exact command script, logs, reports, all raw plans, rollouts and videos are in
`omg_50hz_run_20261019/half_second_20261020` there.
Local copy: `OMG/artifacts/omg_50hz_training_20261019/half_second_20261020`.

Seed 1234 and the same per-plan noise seed sequence are used. Initial plans match exactly.
Later histories and physical times of corresponding plan IDs necessarily differ.
These are sync videos; shared-GPU inference timings are not a real-time acceptance test.
Unit tests compare effective tracker features with simulated async latencies 30/50/70/210 ms.
Full suite: 202 passed; after adding 210ms case, pipeline/realtime subset: 13 passed.

## Results

All table pairs are **0.5s / 1s**. Root error is world-frame XYZ RMSE with no trajectory alignment;
joint error is RMS over frames and 29 joint angles. Seam velocity change is the p95 norm
of the reference root finite-difference velocity change at switching boundaries, not a
claim of physical velocity discontinuity. Different schedules have different seam counts.

| Prompt | Root RMSE cm | Joint RMSE rad | Seam root velocity change p95 m/s |
|---|---:|---:|---:|
| walk forward | 190.11 / 81.78 | 0.1485 / 0.1275 | 2.390 / 1.122 |
| squat down and stand up | 8.98 / 12.16 | 0.1268 / 0.1340 | 0.464 / 0.621 |
| raise hands | 5.59 / 4.64 | 0.1450 / 0.1554 | 0.811 / 0.486 |
| turn around | 12.63 / 10.74 | 0.2016 / 0.1052 | 0.767 / 0.747 |

Walking becomes faster and drifts more: reference net XY displacement grows from 12.46m to
20.10m in 10s. More frequent replanning does not establish continuity and is not an overall
quality improvement. Squatting improves slightly in these tracking metrics; hands are mixed.
Turning joint error increases substantially; reference total yaw is -572.5° vs -402.6°,
so neither rollout implements a one-shot 180° turn-and-stop semantic constraint.
Contact sheets show squat/stand cycles, raised arms, walking and repeated turns.

Recompute metrics with `python scripts/validation/half_second_metrics.py <artifact_directory>`.
The HTML gallery presents all four matched comparisons. No hardware trial or promotion
of this branch to the active deployment is implied by this validation.
