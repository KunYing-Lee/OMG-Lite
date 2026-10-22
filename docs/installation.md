# Installation

Use Python 3.10 and CUDA-capable Linux for training and tracking.

```bash
git clone https://github.com/KunYing-Lee/OMG-Lite.git
cd OMG-Lite
make venv
source .venv/bin/activate
make install                    # or make install-cn
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
```

For a smaller environment, install PyTorch and select the required extras:

```bash
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -e '.[train,data,render,realtime]'
```

Other extras: `tracking`, `export`, `benchmark`, `dev`.

## Data and models

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data-50Hz
export OMG_MATERIALIZED_ROOT="$OMG_DATA_ROOT/materialized"
export OMG_MODELS_ROOT=/path/to/models
```

Training expects T5-base at `$OMG_MODELS_ROOT/t5-base-local`. Download
[T5-base](https://huggingface.co/google-t5/t5-base) for offline use, then set
`HF_HUB_OFFLINE=1`. See [Data](data.md) and [Artifacts](artifacts.md).

Tracking requires MuJoCo and the HoloMotion motion-tracking ONNX model with
any external weights beside it. Obtain models from
[HoloMotion](https://github.com/HorizonRobotics/HoloMotion) or its
[model repository](https://huggingface.co/HorizonRobotics/HoloMotion_models).
Pass the motion tracker to the pipeline with `--tracker`; the native G1
controller also needs its velocity policy. See [G1 Deployment](realtime_g1.md).
