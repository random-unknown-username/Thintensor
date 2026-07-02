# HF vs ThinTensor correctness report

- Model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- Dtype: `bf16`
- Kernel backend: `triton`
- Weight residency: `stream`
- GPU weight budget: `6000000000` bytes
- Prompt: `Hello`
- HF attention implementation: `sdpa`
- HF-equivalent: `false`
- Correctness tier: `ranking_pass`
- Attention-equivalent: `true`
- Ranking-equivalent: `true`
- Quantized experimental: `false`
- LM-head FP8: `false`
- MLP FP8: `false`
- Config mismatches: `0`

| prefill | step | top1 | top5 overlap | cosine | mean abs err | max abs err | token match | attention | KV tokens | not HF equivalent |
|---:|---:|:---:|---:|---:|---:|---:|---:|:---|---:|:---:|
| 1 | 1 | True | 1.000 | 0.999861 | 0.042603 | 0.250000 | 1.000 | causal_kv | 1 | False |

HF top5 (1/1): `[{"token_id": 9125, "logit": 34.0}, {"token_id": 46654, "logit": 21.25}, {"token_id": 2374, "logit": 20.0}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.0}]`

ThinTensor top5 (1/1): `[{"token_id": 9125, "logit": 33.75}, {"token_id": 46654, "logit": 21.125}, {"token_id": 2374, "logit": 19.875}, {"token_id": 744, "logit": 19.75}, {"token_id": 1887, "logit": 19.0}]`

| 1 | 10 | True | 1.000 | 0.999947 | 0.135227 | 0.570312 | 1.000 | causal_kv | 10 | False |

HF top5 (1/10): `[{"token_id": 25, "logit": 37.25}, {"token_id": 1473, "logit": 26.125}, {"token_id": 512, "logit": 25.125}, {"token_id": 31074, "logit": 23.375}, {"token_id": 49131, "logit": 23.0}]`

ThinTensor top5 (1/10): `[{"token_id": 25, "logit": 37.5}, {"token_id": 1473, "logit": 26.0}, {"token_id": 512, "logit": 25.125}, {"token_id": 31074, "logit": 23.5}, {"token_id": 49131, "logit": 23.0}]`

| 8 | 1 | True | 1.000 | 0.999876 | 0.039831 | 0.250000 | 1.000 | causal_kv | 8 | False |

HF top5 (8/1): `[{"token_id": 9125, "logit": 34.0}, {"token_id": 46654, "logit": 21.25}, {"token_id": 2374, "logit": 20.0}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.0}]`

ThinTensor top5 (8/1): `[{"token_id": 9125, "logit": 33.75}, {"token_id": 46654, "logit": 21.125}, {"token_id": 2374, "logit": 19.875}, {"token_id": 744, "logit": 19.75}, {"token_id": 1887, "logit": 19.0}]`

| 8 | 10 | True | 1.000 | 0.999241 | 0.193975 | 0.593750 | 1.000 | causal_kv | 17 | False |

HF top5 (8/10): `[{"token_id": 13466, "logit": 20.125}, {"token_id": 47424, "logit": 19.625}, {"token_id": 76752, "logit": 19.125}, {"token_id": 13068, "logit": 19.0}, {"token_id": 35, "logit": 18.5}]`

ThinTensor top5 (8/10): `[{"token_id": 13466, "logit": 20.25}, {"token_id": 47424, "logit": 19.875}, {"token_id": 76752, "logit": 19.375}, {"token_id": 13068, "logit": 19.125}, {"token_id": 35, "logit": 18.625}]`

| 128 | 1 | True | 1.000 | 0.999696 | 0.065890 | 0.289062 | 1.000 | causal_kv | 128 | False |

HF top5 (128/1): `[{"token_id": 9125, "logit": 34.0}, {"token_id": 46654, "logit": 21.25}, {"token_id": 2374, "logit": 20.0}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.0}]`

ThinTensor top5 (128/1): `[{"token_id": 9125, "logit": 33.75}, {"token_id": 46654, "logit": 21.125}, {"token_id": 2374, "logit": 19.875}, {"token_id": 744, "logit": 19.75}, {"token_id": 1887, "logit": 19.0}]`

| 128 | 10 | True | 1.000 | 0.997313 | 0.329332 | 2.093750 | 0.900 | causal_kv | 137 | False |

HF top5 (128/10): `[{"token_id": 433, "logit": 22.75}, {"token_id": 279, "logit": 22.375}, {"token_id": 358, "logit": 22.125}, {"token_id": 584, "logit": 21.0}, {"token_id": 539, "logit": 21.0}]`

ThinTensor top5 (128/10): `[{"token_id": 433, "logit": 23.0}, {"token_id": 279, "logit": 22.75}, {"token_id": 358, "logit": 22.625}, {"token_id": 584, "logit": 21.5}, {"token_id": 539, "logit": 21.375}]`
