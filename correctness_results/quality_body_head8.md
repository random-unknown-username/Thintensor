# HF vs ThinTensor correctness report

- Model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- Dtype: `bf16`
- Kernel backend: `triton`
- Prompt: `Hello`
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
| 1 | 1 | True | 1.000 | 0.998548 | 0.123760 | 1.527344 | 1.000 | causal_kv | 1 | False |

HF top5 (1/1): `[{"token_id": 9125, "logit": 34.0}, {"token_id": 46654, "logit": 21.25}, {"token_id": 2374, "logit": 20.0}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.0}]`

ThinTensor top5 (1/1): `[{"token_id": 9125, "logit": 34.25}, {"token_id": 46654, "logit": 21.625}, {"token_id": 2374, "logit": 20.125}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.375}]`

| 1 | 10 | True | 1.000 | 0.999084 | 0.732078 | 3.000000 | 1.000 | causal_kv | 10 | False |

HF top5 (1/10): `[{"token_id": 25, "logit": 37.5}, {"token_id": 1473, "logit": 26.25}, {"token_id": 512, "logit": 25.125}, {"token_id": 31074, "logit": 23.375}, {"token_id": 49131, "logit": 23.0}]`

ThinTensor top5 (1/10): `[{"token_id": 25, "logit": 37.75}, {"token_id": 1473, "logit": 26.0}, {"token_id": 512, "logit": 25.0}, {"token_id": 49131, "logit": 23.0}, {"token_id": 31074, "logit": 22.75}]`

| 8 | 1 | True | 1.000 | 0.998394 | 0.134563 | 1.390381 | 1.000 | causal_kv | 8 | False |

HF top5 (8/1): `[{"token_id": 9125, "logit": 34.5}, {"token_id": 46654, "logit": 21.375}, {"token_id": 2374, "logit": 20.125}, {"token_id": 744, "logit": 20.0}, {"token_id": 1887, "logit": 19.25}]`

ThinTensor top5 (8/1): `[{"token_id": 9125, "logit": 34.25}, {"token_id": 46654, "logit": 21.625}, {"token_id": 2374, "logit": 20.125}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.375}]`

| 8 | 10 | True | 1.000 | 0.997136 | 0.225324 | 3.687500 | 0.900 | causal_kv | 17 | False |

HF top5 (8/10): `[{"token_id": 13466, "logit": 20.25}, {"token_id": 47424, "logit": 19.75}, {"token_id": 76752, "logit": 19.375}, {"token_id": 13068, "logit": 19.0}, {"token_id": 35, "logit": 18.5}]`

ThinTensor top5 (8/10): `[{"token_id": 13466, "logit": 20.0}, {"token_id": 47424, "logit": 19.625}, {"token_id": 76752, "logit": 19.25}, {"token_id": 13068, "logit": 19.0}, {"token_id": 35, "logit": 18.5}]`

| 128 | 1 | True | 1.000 | 0.998567 | 0.123164 | 1.479492 | 1.000 | causal_kv | 128 | False |

HF top5 (128/1): `[{"token_id": 9125, "logit": 34.0}, {"token_id": 46654, "logit": 21.25}, {"token_id": 2374, "logit": 20.0}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.0}]`

ThinTensor top5 (128/1): `[{"token_id": 9125, "logit": 34.25}, {"token_id": 46654, "logit": 21.625}, {"token_id": 2374, "logit": 20.125}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.375}]`

| 128 | 10 | False | 1.000 | 0.996907 | 0.322944 | 2.937500 | 0.800 | causal_kv | 137 | False |

HF top5 (128/10): `[{"token_id": 433, "logit": 22.625}, {"token_id": 279, "logit": 22.375}, {"token_id": 358, "logit": 22.125}, {"token_id": 584, "logit": 20.875}, {"token_id": 539, "logit": 20.875}]`

ThinTensor top5 (128/10): `[{"token_id": 279, "logit": 22.75}, {"token_id": 433, "logit": 22.625}, {"token_id": 358, "logit": 21.75}, {"token_id": 539, "logit": 20.75}, {"token_id": 584, "logit": 20.375}]`
