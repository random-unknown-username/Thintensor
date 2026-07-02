# HF vs ThinTensor correctness report

- Model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- Dtype: `bf16`
- Kernel backend: `triton`
- Weight residency: `all`
- GPU weight budget: `0` bytes
- Prompt: `Hello`
- HF attention implementation: `sdpa`
- HF-equivalent: `false`
- Correctness tier: `ranking_pass`
- Attention-equivalent: `true`
- Ranking-equivalent: `true`
- Quantized experimental: `true`
- LM-head FP8: `false`
- MLP FP8: `false`
- Config mismatches: `0`

| prefill | step | top1 | top5 overlap | cosine | mean abs err | max abs err | token match | attention | KV tokens | not HF equivalent |
|---:|---:|:---:|---:|---:|---:|---:|---:|:---|---:|:---:|
| 1 | 1 | True | 1.000 | 0.999391 | 0.081911 | 0.703125 | 1.000 | causal_kv | 1 | False |

HF top5 (1/1): `[{"token_id": 9125, "logit": 34.0}, {"token_id": 46654, "logit": 21.25}, {"token_id": 2374, "logit": 20.0}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.0}]`

ThinTensor top5 (1/1): `[{"token_id": 9125, "logit": 34.25}, {"token_id": 46654, "logit": 21.5}, {"token_id": 2374, "logit": 20.125}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.125}]`

| 1 | 10 | True | 1.000 | 0.999515 | 0.594768 | 1.593750 | 1.000 | causal_kv | 10 | False |

HF top5 (1/10): `[{"token_id": 25, "logit": 37.25}, {"token_id": 1473, "logit": 26.125}, {"token_id": 512, "logit": 25.125}, {"token_id": 31074, "logit": 23.375}, {"token_id": 49131, "logit": 23.0}]`

ThinTensor top5 (1/10): `[{"token_id": 25, "logit": 38.0}, {"token_id": 1473, "logit": 26.125}, {"token_id": 512, "logit": 25.125}, {"token_id": 49131, "logit": 22.875}, {"token_id": 31074, "logit": 22.75}]`

| 8 | 1 | True | 1.000 | 0.999618 | 0.065219 | 0.593750 | 1.000 | causal_kv | 8 | False |

HF top5 (8/1): `[{"token_id": 9125, "logit": 34.0}, {"token_id": 46654, "logit": 21.25}, {"token_id": 2374, "logit": 20.0}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.0}]`

ThinTensor top5 (8/1): `[{"token_id": 9125, "logit": 34.25}, {"token_id": 46654, "logit": 21.5}, {"token_id": 2374, "logit": 20.125}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.125}]`

| 8 | 10 | True | 1.000 | 0.999179 | 0.133115 | 1.062500 | 1.000 | causal_kv | 17 | False |

HF top5 (8/10): `[{"token_id": 13466, "logit": 20.125}, {"token_id": 47424, "logit": 19.625}, {"token_id": 76752, "logit": 19.125}, {"token_id": 13068, "logit": 19.0}, {"token_id": 35, "logit": 18.5}]`

ThinTensor top5 (8/10): `[{"token_id": 13466, "logit": 20.0}, {"token_id": 47424, "logit": 19.625}, {"token_id": 76752, "logit": 19.25}, {"token_id": 13068, "logit": 18.75}, {"token_id": 35, "logit": 18.375}]`

| 128 | 1 | True | 1.000 | 0.999537 | 0.071246 | 0.625000 | 1.000 | causal_kv | 128 | False |

HF top5 (128/1): `[{"token_id": 9125, "logit": 34.0}, {"token_id": 46654, "logit": 21.25}, {"token_id": 2374, "logit": 20.0}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.0}]`

ThinTensor top5 (128/1): `[{"token_id": 9125, "logit": 34.25}, {"token_id": 46654, "logit": 21.5}, {"token_id": 2374, "logit": 20.125}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.125}]`

| 128 | 10 | True | 1.000 | 0.998697 | 0.221091 | 1.484375 | 0.900 | causal_kv | 137 | False |

HF top5 (128/10): `[{"token_id": 433, "logit": 22.75}, {"token_id": 279, "logit": 22.375}, {"token_id": 358, "logit": 22.125}, {"token_id": 584, "logit": 21.0}, {"token_id": 539, "logit": 21.0}]`

ThinTensor top5 (128/10): `[{"token_id": 433, "logit": 22.5}, {"token_id": 279, "logit": 22.25}, {"token_id": 358, "logit": 22.0}, {"token_id": 539, "logit": 20.75}, {"token_id": 584, "logit": 20.625}]`
