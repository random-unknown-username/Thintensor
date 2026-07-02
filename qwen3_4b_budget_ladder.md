# Qwen3-4B GPU budget ladder

## Current status

The Qwen3-4B `.thin` archive is not present in the current workspace or `/tmp`,
so the requested 1/2/4/6 GB CPU-offload ladder could not be revalidated in this
run. No new speed, transfer, cache-hit, or memory numbers are claimed here.

Previously established behavior, retained only as historical context:

- HF Transformers BF16 OOMs on the 8 GB GPU near 7.96 GB allocated.
- ThinTensor cold BF16 streaming runs with the corrected runtime geometry
  (`32` query heads, `8` KV heads, `128` head dimension).
- The prior cold stream moved roughly 7.42 GB/token and ran around 1.45 tok/s.

Those historical values are not a CPU-offload budget ladder and must not be
presented as one.

| Mode | GPU budget | Current-run result | Reason |
|:---|---:|:---|:---|
| Cold stream | implicit | not rerun | archive absent |
| CPU offload | 1 GB | unavailable | archive absent |
| CPU offload | 2 GB | unavailable | archive absent |
| CPU offload | 4 GB | unavailable | archive absent |
| CPU offload | 6 GB | unavailable | archive absent |
| CPU offload + prefetch | 4 GB | unavailable | archive absent |
| CPU offload + prefetch | 6 GB | unavailable | archive absent |

A valid future ladder must record GPU resident/peak bytes, cache hits/misses,
evictions, prefetch waits, and H2D bytes per token for every row. A useful
budget result requires transfer per token to decrease as the budget increases.
