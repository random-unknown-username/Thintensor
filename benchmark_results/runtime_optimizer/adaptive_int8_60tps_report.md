# Adaptive body INT8 result

## Accepted speed result

The all-resident guarded-quality decode path now has an opt-in adaptive body
mode. It keeps the established FP8 gate/up, selective FP8 down projection, and
BF16 attention projections for the first 18 token positions. After that prefix,
the remaining down, QKV, and O-projection weights use row-scaled INT8. Exact
prefix weights remain resident.

Command:

```bash
python3 scripts/thin_runtime.py run /home/satvik/Projects/SmolLM3-3B.thin \
  --device cuda --dtype bf16 --residency all \
  --steps 500 --warmup-steps 10 \
  --kernel-backend triton --attention-mode causal_kv \
  --attention-backend triton_fused --kv-block-size 512 \
  --adaptive-body-int8-start-token 18 \
  --lm-head-fp8 --keep-bf16-lm-head --lm-head-topk-guard 64 --json
```

Two clean 500-token runs:

| Run | Steady tok/s | ms/token |
|---|---:|---:|
| 1 | 61.2313 | 16.3286 |
| 2 | 60.7522 | 16.4571 |
| Median | 60.9918 | 16.3928 |

The locked guarded-quality baseline was 50.4275 tok/s and 19.8217 ms/token.
Estimated projection traffic fell from 3,813,921,792 to 3,075,724,288 bytes per
token. Resident weights are 5,077,492,736 bytes, including 1,476,395,008 bytes
of exact-prefix sidecars. Peak CUDA allocation was 6,590,755,840 bytes.

## Correctness gate

The combined HF-referenced check used prefills 1 and 8 and checkpoints
1, 10, 32, 64, and 128. Relative to the locked guarded-quality profile:

- maximum cosine drop: 0.00001365
- requested maximum cosine drop: 0.00005000
- top-1 failures: 0
- minimum top-5 overlap with HF: 0.8

One prefill-8, step-32 top-5 set changed from full overlap to 0.8; top-1 stayed
unchanged and the cosine delta there was -0.00001365. This is disclosed rather
than described as bit-exact quality.

## Kernel decisions

- The adaptive INT8 down projection uses loop-256, block-M 32, four warps.
- The exact prefix uses the prior BF16/FP8 kernels and torch attention.
- The fused attention path starts with the adaptive body transition.
- Block-scaled matvec launch geometry now rounds the inferred scale block to a
  Triton-compatible power of two, fixing invalid 512/1024-scale launches for
  widths that are not evenly divisible by the number of scale blocks.
