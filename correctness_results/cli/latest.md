# HF vs ThinTensor correctness report

- Model: `/home/satvik/.cache/thintensor/models/unsloth--Llama-3.2-11B-Vision-Instruct`
- Archive: `/home/satvik/.cache/thintensor/models/unsloth--Llama-3.2-11B-Vision-Instruct/llama-3.2-11b-vision-text.thin`
- Dtype: `bf16`
- Kernel backend: `triton`
- Weight residency: `all`
- GPU weight budget: `0` bytes
- Prompt: `Hello`
- HF attention implementation: `sdpa`
- HF-equivalent: `false`
- Correctness tier: `fail`
- Attention-equivalent: `true`
- Ranking-equivalent: `false`
- Quantized experimental: `false`
- LM-head FP8: `false`
- MLP FP8: `false`
- Config mismatches: `3`

| prefill | step | top1 | top5 overlap | cosine | mean abs err | max abs err | token match | attention | KV tokens | not HF equivalent |
|---:|---:|:---:|---:|---:|---:|---:|---:|:---|---:|:---:|
| 1 | 1 | True | 0.600 | 0.942747 | 0.615646 | 3.882812 | 1.000 | causal_kv | 1 | False |

HF top5 (1/1): `[{"token_id": 14924, "logit": 11.75}, {"token_id": 128006, "logit": 10.75}, {"token_id": 17297, "logit": 10.4375}, {"token_id": 755, "logit": 10.125}, {"token_id": 121648, "logit": 9.5625}]`

ThinTensor top5 (1/1): `[{"token_id": 14924, "logit": 11.1875}, {"token_id": 755, "logit": 10.5}, {"token_id": 17297, "logit": 9.5}, {"token_id": 475, "logit": 9.3125}, {"token_id": 111912, "logit": 9.1875}]`

| 1 | 10 | True | 0.800 | 0.992658 | 0.317683 | 3.062500 | 0.900 | causal_kv | 10 | False |

HF top5 (1/10): `[{"token_id": 374, "logit": 19.25}, {"token_id": 19, "logit": 17.5}, {"token_id": 16, "logit": 17.375}, {"token_id": 24, "logit": 17.375}, {"token_id": 18, "logit": 17.375}]`

ThinTensor top5 (1/10): `[{"token_id": 374, "logit": 19.875}, {"token_id": 19, "logit": 18.0}, {"token_id": 18, "logit": 17.75}, {"token_id": 22, "logit": 17.625}, {"token_id": 16, "logit": 17.625}]`

| 128 | 1 | True | 0.800 | 0.980700 | 0.575037 | 2.375000 | 1.000 | causal_kv | 128 | False |

HF top5 (128/1): `[{"token_id": 11, "logit": 12.4375}, {"token_id": 0, "logit": 12.125}, {"token_id": 323, "logit": 12.0}, {"token_id": 5127, "logit": 11.6875}, {"token_id": 505, "logit": 11.125}]`

ThinTensor top5 (128/1): `[{"token_id": 11, "logit": 13.375}, {"token_id": 0, "logit": 13.0}, {"token_id": 323, "logit": 12.8125}, {"token_id": 5127, "logit": 12.3125}, {"token_id": 1070, "logit": 11.9375}]`

| 128 | 10 | True | 0.800 | 0.987488 | 0.511413 | 3.898438 | 0.700 | causal_kv | 137 | False |

HF top5 (128/10): `[{"token_id": 662, "logit": 8.0625}, {"token_id": 22691, "logit": 8.0625}, {"token_id": 1174, "logit": 7.28125}, {"token_id": 578, "logit": 7.21875}, {"token_id": 220, "logit": 6.875}]`

ThinTensor top5 (128/10): `[{"token_id": 662, "logit": 8.375}, {"token_id": 1174, "logit": 7.75}, {"token_id": 22691, "logit": 7.6875}, {"token_id": 578, "logit": 7.46875}, {"token_id": 482, "logit": 7.25}]`
