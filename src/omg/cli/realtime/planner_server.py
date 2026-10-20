import argparse

import torch

from omg.pipeline.checkpoint import CheckpointPlanner
from omg.realtime.native import PlannerService
from omg.realtime.transport import ZmqPlanServer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bind", default="tcp://0.0.0.0:5571")
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    torch.set_num_threads(1)
    service = PlannerService(CheckpointPlanner(a.config, a.checkpoint, seed=a.seed))
    with ZmqPlanServer(a.bind) as server:
        while True:
            request = server.recv_request()
            server.send_plan(service.plan(request))


if __name__ == "__main__":
    main()
