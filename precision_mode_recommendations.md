# Precision mode recommendations

Measured on the NVIDIA GeForce RTX 5050 Laptop GPU with SmolLM3-3B. Throughput
is batch-1 causal-KV decode. FP8 modes use scaled row-wise FP8 weights and are
opt-in selective quantization, not BF16.

| Tier | Mode | Throughput | Worst tested cosine | Resident bytes saved | Recommendation |
|:---|:---|---:|---:|---:|:---|
| Safe default | BF16 causal KV | 31.44 tok/s | 0.99814 optimized; 0.99932 exact mode | 0 | Default runtime path |
| Exact reference | `--exact-hf-mode` | correctness-only | 0.99932 | 0 | Debug and HF parity checks; pins HF comparison to eager attention |
| Quality-first selective FP8 | gate/up all + down layers 8:28 | 46.41 tok/s over 500 tokens | 0.99804 required; 0.99379 stress subset | 2,070,749,184 | Best retained general speed/quality mode; opt-in |
| Faster selective FP8 | gate/up all + down layers 6:30 + O layers 4:32 | 49.88 tok/s over 500 tokens | 0.99650 required; 0.98471 stress matrix | 2,278,105,088 | Faster, but O FP8 causes materially larger chat-template drift |
| Ranking-stable selective FP8 | `--gate-up-fp8` | 43.37 tok/s | 0.99444 | 1,620,025,344 | Conservative quantized body option |
| Ranking-stable selective FP8 | gate/up all + down layers 6:30 | 47.00 tok/s | 0.99588 | 2,160,893,952 | Higher quality than full MLP FP8 |
| Experimental fast | `--mlp-fp8` | 50.61 tok/s | 0.98378 | 2,431,328,256 | Short tests preserve rankings, but logit-vector drift is material |
| Experimental fast | `--mlp-fp8 --o-proj-fp8` | 51.53 tok/s | 0.97096 | 2,582,028,288 | Maximum body-FP8 speed tested without Head8; larger drift |
| Rejected for safety | `--mlp-fp8 --lm-head-fp8` | 53.54 tok/s | 0.98303 | 2,431,328,256 body bytes | A tested top-1 differs; do not call ranking-equivalent |

The retained quality-first command is:

```bash
python3 scripts/thin_runtime.py run SmolLM3-3B.thin \
  --device cuda --dtype bf16 --steps 500 --warmup-steps 10 \
  --residency all --attention-mode causal_kv --kernel-backend triton \
  --gate-up-fp8 \
  --down-proj-fp8 --down-fp8-layers 8:28 \
  --lm-head-backend triton --json
```

Do not make any body-FP8 mode the default. The exact BF16 report uses a
full-recompute eager HF reference; the stress report separately tests cached HF
decode and must be interpreted as trajectory stability, not as the exact
equivalence gate. On deterministic sampling tests, the quality-first family
still showed trajectory divergence after small logit differences; use the
reported total-variation and Jensen-Shannon metrics rather than treating exact
sampled-token identity as distribution equivalence.
