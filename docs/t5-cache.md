# Frozen T5 encoding cache

Opt in with:

```bash
python -m omg.cli.generation.train exp=base \
  model.text_encoder.frozen_eval=true \
  model.text_encoder.cache_max_entries=4096 \
  model.text_encoder.cache_device=encoder
```

Defaults remain `frozen_eval=false` and `cache_max_entries=0`, preserving existing
training semantics. The cache requires `frozen_eval=true`. Existing checkpoints
load without changes to weight keys; cache contents are transient.

## Important experiment difference

Historically the wrapper called `encoder.eval()` only during construction. The
parent model's `train()` recursively re-enabled T5 dropout, even though parameters
were frozen. The local T5-base configuration has dropout 0.1. Memoizing these
stochastic outputs would be incorrect. `frozen_eval=true` explicitly keeps T5 in
evaluation mode through parent train/eval transitions. This is a change to the
text-conditioning distribution compared with legacy training. Compare cache on
and off with **frozen_eval=true in both cases** to isolate caching performance.
Do not describe switching a running legacy experiment as a bitwise-equivalent
optimization. Checkpoints do not retain a cache or sampler/RNG continuity.

## Cache contract

Each DDP rank has a bounded LRU cache keyed by the exact caption string. It
stores the frozen encoder's unprojected hidden states and tokenizer attention mask.
Max-length padding, truncation, EOS and empty-caption behavior remain unchanged.
Duplicate misses are encoded once per call. The optional trainable projection and
per-call CFG/text mask are applied after lookup, so projection gradients are
preserved. Precision/autocast or device changes and weight reloads invalidate the
cache. The cache rejects an encoder in training mode.

`cache_device=cpu` stores entries in host RAM; `cache_device=encoder` stores them
on the encoder device, avoiding host-to-device copies on cache hits.

At max_length=50 and hidden_dim=768, 4096 FP32 entries occupy approximately 600 MiB
per rank plus masks and Python metadata (about 4.7 GiB across eight ranks). BF16
entries use half that storage if the encoder actually returns BF16. Bounded
capacity does not imply that the corpus fits or that hit rate is high; measure
hit rate on the actual sampler. Cache counters count a duplicate in the same
batch as a hit because it avoids an extra encoding.

Miss batching may change floating-point rounding relative to encoding a full
batch. Validate numerical agreement at the production precision, and measure
end-to-end throughput on isolated GPUs before making speedup claims. CPU cached
lookup latency is not a GPU training speedup measurement.
