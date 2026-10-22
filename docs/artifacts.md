# Released inference artifacts

Use the [38D / 50Hz bundle](https://huggingface.co/THU-MARS/OMG/tree/main/checkpoints/38D).
The best-validation checkpoint is `sstep=075500.ckpt`, SHA256
`8bdd19cc6a47131dacb8f29a2a440c6a3a405dfcd335ecfffc3054828bd59fe7`.

Download the whole directory, including `config.yaml`, normalization stats,
`g1_kinematics.json`, `g1_29dof.urdf`, `seed_stand.npz`, and `infer.py`.
The script resolves paths relative to the bundle, verifies hashes and strictly
loads the checkpoint. T5-base is the generation text encoder; install the
training extras and provide a local T5-base directory for offline operation.
Follow the bundle README or the repository quickstart.

The bundle's validation record reports a 100-frame native50Hz generation from
standing history, with exact output parity against the original training config.
It is not tracker or physical-robot acceptance. The released checkpoint was
trained with diffusion loss only. Optional FK loss remains available for new runs.

For the simulation CLI, export `OMG38_ARTIFACT_DIR` as the absolute bundle
folder and optionally `OMG38_T5` as a local T5-base directory. Pass its
`config.yaml`, checkpoint and seed motion to the pipeline. HoloMotion v1.4.1
`model_16200_dynamic_batch.onnx` and tracking dependencies are obtained separately.
