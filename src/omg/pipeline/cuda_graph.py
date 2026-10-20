"""Exact CUDA replay of a fixed-shape denoiser; no diffusion-step caching."""

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


class CUDAGraphDenoiser(torch.nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser
        self.graphs = {}

    def forward(self, x, timesteps, cond_tokens, valid_mask=None):
        leaves, spec = tree_flatten((x, timesteps, cond_tokens, valid_mask))
        if any(v is not None and not isinstance(v, torch.Tensor) for v in leaves):
            raise TypeError("CUDA replay inputs must be tensors or None")
        key = (
            str(spec),
            tuple(
                None if v is None else (tuple(v.shape), v.dtype, v.device)
                for v in leaves
            ),
        )
        if key not in self.graphs:
            static = [None if v is None else v.clone() for v in leaves]
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                args = tree_unflatten(static, spec)
                for _ in range(3):
                    self.denoiser(*args)
            torch.cuda.current_stream(x.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.denoiser(*args)
            self.graphs[key] = (static, graph, output)
        static, graph, output = self.graphs[key]
        for dst, src in zip(static, leaves):
            if dst is not None:
                dst.copy_(src)
        graph.replay()
        # CFG evaluates conditional and unconditional branches sequentially.
        # A fresh result is necessary so the second replay cannot overwrite the first.
        return output.clone()
