import argparse

import torch

from omg.pipeline.schedule import Schedule
from omg.realtime.native import PlannerService
from omg.realtime.transport import ZmqPlanServer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("tensorrt", "pytorch"), default="tensorrt")
    p.add_argument("--onnx")
    p.add_argument("--metadata")
    p.add_argument("--text-encoder")
    p.add_argument("--engine-cache")
    p.add_argument("--config")
    p.add_argument("--checkpoint")
    p.add_argument("--bind", default="tcp://0.0.0.0:5571")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--switch-seconds", type=float, default=1.0)
    a = p.parse_args()
    schedule = Schedule(a.switch_seconds)
    torch.set_num_threads(1)
    if a.backend == "tensorrt":
        if not a.onnx or a.config or a.checkpoint:
            p.error("TensorRT requires --onnx; --config/--checkpoint belong to --backend pytorch")
        from omg.pipeline.tensorrt import TensorRTPlanner

        planner = TensorRTPlanner(a.onnx, metadata=a.metadata, text_encoder=a.text_encoder,
                                  engine_cache=a.engine_cache, seed=a.seed)
    else:
        if not a.config or not a.checkpoint or any((a.onnx, a.metadata, a.text_encoder, a.engine_cache)):
            p.error("PyTorch requires --config and --checkpoint, without TensorRT arguments")
        from omg.pipeline.checkpoint import CheckpointPlanner

        planner = CheckpointPlanner(a.config, a.checkpoint, seed=a.seed)
    service = PlannerService(planner, schedule=schedule)
    with ZmqPlanServer(a.bind) as server:
        while True:
            request = server.recv_request()
            server.send_plan(service.plan(request))


if __name__ == "__main__":
    main()
