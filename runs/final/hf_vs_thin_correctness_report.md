# HF vs ThinTensor correctness report

- Model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- Dtype: `bf16`
- Prompt: `Hello`
- HF-equivalent: `true`
- LM-head FP8: `false`
- MLP FP8: `true`

| prefill | step | top1 | top5 overlap | cosine | mean abs err | max abs err | token match | attention | KV tokens | not HF equivalent |
|---:|---:|:---:|---:|---:|---:|---:|---:|:---|---:|:---:|
| 1 | 1 | True | 1.000 | 0.987830 | 0.377633 | 3.375000 | 1.000 | causal_kv | 1 | False |

HF top5 (1/1): `[{"token_id": 9125, "logit": 34.0}, {"token_id": 46654, "logit": 21.25}, {"token_id": 2374, "logit": 20.0}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.0}]`

ThinTensor top5 (1/1): `[{"token_id": 9125, "logit": 30.625}, {"token_id": 46654, "logit": 19.5}, {"token_id": 2374, "logit": 18.625}, {"token_id": 744, "logit": 18.0}, {"token_id": 1887, "logit": 16.75}]`

| 1 | 10 | True | 1.000 | 0.999665 | 0.288996 | 1.375000 | 1.000 | causal_kv | 10 | False |

HF top5 (1/10): `[{"token_id": 25, "logit": 37.5}, {"token_id": 1473, "logit": 26.125}, {"token_id": 512, "logit": 25.125}, {"token_id": 31074, "logit": 23.375}, {"token_id": 49131, "logit": 23.0}]`

ThinTensor top5 (1/10): `[{"token_id": 25, "logit": 38.5}, {"token_id": 1473, "logit": 26.5}, {"token_id": 512, "logit": 25.625}, {"token_id": 49131, "logit": 23.0}, {"token_id": 31074, "logit": 22.625}]`

| 1 | 50 | True | 0.800 | 0.998463 | 0.321349 | 2.375000 | 1.000 | causal_kv | 50 | False |

HF top5 (1/50): `[{"token_id": 16572, "logit": 30.625}, {"token_id": 4967, "logit": 17.875}, {"token_id": 70024, "logit": 17.625}, {"token_id": 6319, "logit": 16.625}, {"token_id": 56168, "logit": 15.75}]`

ThinTensor top5 (1/50): `[{"token_id": 16572, "logit": 30.875}, {"token_id": 70024, "logit": 18.0}, {"token_id": 4967, "logit": 18.0}, {"token_id": 6319, "logit": 16.875}, {"token_id": 28175, "logit": 16.0}]`

| 8 | 1 | True | 1.000 | 0.983844 | 0.435752 | 3.875000 | 1.000 | causal_kv | 8 | False |

HF top5 (8/1): `[{"token_id": 9125, "logit": 34.5}, {"token_id": 46654, "logit": 21.375}, {"token_id": 2374, "logit": 20.125}, {"token_id": 744, "logit": 20.0}, {"token_id": 1887, "logit": 19.25}]`

ThinTensor top5 (8/1): `[{"token_id": 9125, "logit": 30.625}, {"token_id": 46654, "logit": 19.5}, {"token_id": 2374, "logit": 18.625}, {"token_id": 744, "logit": 18.0}, {"token_id": 1887, "logit": 16.75}]`

| 8 | 10 | True | 1.000 | 0.999047 | 0.160953 | 0.828125 | 1.000 | causal_kv | 17 | False |

HF top5 (8/10): `[{"token_id": 13466, "logit": 20.125}, {"token_id": 47424, "logit": 19.75}, {"token_id": 76752, "logit": 19.25}, {"token_id": 13068, "logit": 19.0}, {"token_id": 35, "logit": 18.5}]`

ThinTensor top5 (8/10): `[{"token_id": 13466, "logit": 20.25}, {"token_id": 47424, "logit": 19.625}, {"token_id": 76752, "logit": 19.5}, {"token_id": 13068, "logit": 18.875}, {"token_id": 35, "logit": 18.625}]`

| 8 | 50 | True | 1.000 | 0.996104 | 0.236643 | 1.287422 | 0.980 | causal_kv | 57 | False |

HF top5 (8/50): `[{"token_id": 271, "logit": 20.375}, {"token_id": 2754, "logit": 18.5}, {"token_id": 46508, "logit": 17.75}, {"token_id": 36, "logit": 17.5}, {"token_id": 320, "logit": 16.625}]`

ThinTensor top5 (8/50): `[{"token_id": 271, "logit": 20.25}, {"token_id": 2754, "logit": 18.25}, {"token_id": 46508, "logit": 17.25}, {"token_id": 36, "logit": 17.25}, {"token_id": 320, "logit": 16.625}]`

| 32 | 1 | True | 1.000 | 0.987752 | 0.378166 | 3.375000 | 1.000 | causal_kv | 32 | False |

HF top5 (32/1): `[{"token_id": 9125, "logit": 34.0}, {"token_id": 46654, "logit": 21.25}, {"token_id": 2374, "logit": 20.0}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.0}]`

ThinTensor top5 (32/1): `[{"token_id": 9125, "logit": 30.625}, {"token_id": 46654, "logit": 19.5}, {"token_id": 2374, "logit": 18.625}, {"token_id": 744, "logit": 18.0}, {"token_id": 1887, "logit": 16.875}]`

| 32 | 10 | True | 0.800 | 0.997434 | 0.470923 | 1.609375 | 0.900 | causal_kv | 41 | False |

HF top5 (32/10): `[{"token_id": 2846, "logit": 25.0}, {"token_id": 1097, "logit": 24.375}, {"token_id": 4344, "logit": 23.25}, {"token_id": 1053, "logit": 21.375}, {"token_id": 1120, "logit": 21.125}]`

ThinTensor top5 (32/10): `[{"token_id": 2846, "logit": 25.625}, {"token_id": 1097, "logit": 24.875}, {"token_id": 4344, "logit": 23.75}, {"token_id": 3077, "logit": 21.875}, {"token_id": 1053, "logit": 21.875}]`

| 32 | 50 | True | 1.000 | 0.999804 | 0.113913 | 0.812500 | 0.980 | causal_kv | 81 | False |

HF top5 (32/50): `[{"token_id": 13, "logit": 28.375}, {"token_id": 382, "logit": 27.875}, {"token_id": 1131, "logit": 23.75}, {"token_id": 11, "logit": 23.75}, {"token_id": 2195, "logit": 23.625}]`

ThinTensor top5 (32/50): `[{"token_id": 13, "logit": 28.25}, {"token_id": 382, "logit": 27.75}, {"token_id": 1131, "logit": 23.875}, {"token_id": 2195, "logit": 23.75}, {"token_id": 11, "logit": 23.625}]`

| 128 | 1 | True | 1.000 | 0.987808 | 0.377613 | 3.375000 | 1.000 | causal_kv | 128 | False |

HF top5 (128/1): `[{"token_id": 9125, "logit": 34.0}, {"token_id": 46654, "logit": 21.25}, {"token_id": 2374, "logit": 20.0}, {"token_id": 744, "logit": 19.875}, {"token_id": 1887, "logit": 19.0}]`

ThinTensor top5 (128/1): `[{"token_id": 9125, "logit": 30.625}, {"token_id": 46654, "logit": 19.5}, {"token_id": 2374, "logit": 18.625}, {"token_id": 744, "logit": 18.0}, {"token_id": 1887, "logit": 16.75}]`

| 128 | 10 | True | 1.000 | 0.994004 | 0.611234 | 3.250000 | 0.800 | causal_kv | 137 | False |

HF top5 (128/10): `[{"token_id": 433, "logit": 22.875}, {"token_id": 279, "logit": 22.625}, {"token_id": 358, "logit": 22.125}, {"token_id": 584, "logit": 21.25}, {"token_id": 539, "logit": 21.0}]`

ThinTensor top5 (128/10): `[{"token_id": 433, "logit": 23.0}, {"token_id": 279, "logit": 22.75}, {"token_id": 358, "logit": 22.625}, {"token_id": 539, "logit": 21.125}, {"token_id": 584, "logit": 21.0}]`

| 128 | 50 | True | 1.000 | 0.999628 | 0.252557 | 1.718750 | 0.920 | causal_kv | 177 | False |

HF top5 (128/50): `[{"token_id": 13, "logit": 34.0}, {"token_id": 3060, "logit": 29.75}, {"token_id": 382, "logit": 29.5}, {"token_id": 11, "logit": 28.125}, {"token_id": 1606, "logit": 25.875}]`

ThinTensor top5 (128/50): `[{"token_id": 13, "logit": 34.0}, {"token_id": 3060, "logit": 30.125}, {"token_id": 382, "logit": 29.375}, {"token_id": 11, "logit": 28.25}, {"token_id": 1606, "logit": 26.25}]`
