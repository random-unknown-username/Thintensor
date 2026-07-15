# HF vs ThinTensor correctness report

- Model: `/home/satvik/.cache/thintensor/models/microsoft--phi-4`
- Archive: `/home/satvik/.cache/thintensor/models/microsoft--phi-4/phi-4-fresh.thin`
- Dtype: `bf16`
- Kernel backend: `triton`
- Weight residency: `stream`
- GPU weight budget: `7800000000` bytes
- Prompt: `The capital of France is`
- HF attention implementation: `sdpa`
- HF-equivalent: `false`
- Correctness tier: `experimental`
- Attention-equivalent: `true`
- Ranking-equivalent: `false`
- Quantized experimental: `true`
- LM-head FP8: `true`
- MLP FP8: `false`
- Config mismatches: `0`

| prefill | step | top1 | top5 overlap | cosine | mean abs err | max abs err | token match | attention | KV tokens | not HF equivalent |
|---:|---:|:---:|---:|---:|---:|---:|---:|:---|---:|:---:|
| 5 | 1 | True | 0.800 | 0.984758 | 0.449006 | 2.804688 | 1.000 | causal_kv | 5 | False |

HF top5 (5/1): `[{"token_id": 12366, "logit": 18.75}, {"token_id": 3967, "logit": 14.625}, {"token_id": 264, "logit": 14.25}, {"token_id": 3146, "logit": 14.0}, {"token_id": 539, "logit": 13.5625}]`

ThinTensor top5 (5/1): `[{"token_id": 12366, "logit": 17.75}, {"token_id": 264, "logit": 13.6875}, {"token_id": 3967, "logit": 13.625}, {"token_id": 1, "logit": 13.0625}, {"token_id": 3146, "logit": 13.0}]`

| 5 | 8 | True | 0.800 | 0.986204 | 0.376071 | 2.304688 | 0.875 | causal_kv | 12 | False |

HF top5 (5/8): `[{"token_id": 8085, "logit": 14.875}, {"token_id": 40, "logit": 14.5}, {"token_id": 3923, "logit": 14.0}, {"token_id": 644, "logit": 13.6875}, {"token_id": 4438, "logit": 13.625}]`

ThinTensor top5 (5/8): `[{"token_id": 8085, "logit": 16.0}, {"token_id": 40, "logit": 14.4375}, {"token_id": 3923, "logit": 14.125}, {"token_id": 644, "logit": 13.8125}, {"token_id": 32, "logit": 13.375}]`

| 5 | 32 | True | 1.000 | 0.990491 | 0.261207 | 2.250000 | 0.906 | causal_kv | 36 | False |

HF top5 (5/32): `[{"token_id": 524, "logit": 21.0}, {"token_id": 28293, "logit": 17.375}, {"token_id": 4005, "logit": 14.8125}, {"token_id": 9668, "logit": 14.375}, {"token_id": 612, "logit": 13.6875}]`

ThinTensor top5 (5/32): `[{"token_id": 524, "logit": 20.125}, {"token_id": 28293, "logit": 16.875}, {"token_id": 4005, "logit": 14.0}, {"token_id": 612, "logit": 13.75}, {"token_id": 9668, "logit": 13.3125}]`
