# HF vs ThinTensor correctness report

- Model: `/home/satvik/.cache/thintensor/models/Qwen--Qwen3.5-0.8B`
- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.5-0.8B.thin`
- Dtype: `bf16`
- Kernel backend: `triton-matvec`
- Weight residency: `all`
- GPU weight budget: `0` bytes
- Prompt: `Hello`
- HF attention implementation: `sdpa`
- HF-equivalent: `false`
- Correctness tier: `experimental`
- Attention-equivalent: `true`
- Ranking-equivalent: `false`
- Quantized experimental: `true`
- LM-head FP8: `true`
- MLP FP8: `false`
- Config mismatches: `2`

| prefill | step | top1 | top5 overlap | cosine | mean abs err | max abs err | token match | attention | KV tokens | not HF equivalent |
|---:|---:|:---:|---:|---:|---:|---:|---:|:---|---:|:---:|
| 1 | 1 | True | 0.800 | 0.998687 | 0.227240 | 1.375000 | 1.000 | causal_kv | 1 | False |

HF top5 (1/1): `[{"token_id": 11, "logit": 12.625}, {"token_id": 13, "logit": 11.4375}, {"token_id": 0, "logit": 10.6875}, {"token_id": 198, "logit": 10.1875}, {"token_id": 4858, "logit": 9.625}]`

ThinTensor top5 (1/1): `[{"token_id": 11, "logit": 12.1875}, {"token_id": 13, "logit": 11.5}, {"token_id": 198, "logit": 10.4375}, {"token_id": 0, "logit": 10.4375}, {"token_id": 25, "logit": 9.5625}]`

| 1 | 10 | True | 1.000 | 0.998633 | 0.115476 | 0.875000 | 0.800 | causal_kv | 10 | False |

HF top5 (1/10): `[{"token_id": 421, "logit": 22.25}, {"token_id": 310, "logit": 19.875}, {"token_id": 321, "logit": 19.75}, {"token_id": 1332, "logit": 19.125}, {"token_id": 364, "logit": 19.0}]`

ThinTensor top5 (1/10): `[{"token_id": 421, "logit": 22.0}, {"token_id": 310, "logit": 19.75}, {"token_id": 321, "logit": 19.375}, {"token_id": 1332, "logit": 19.0}, {"token_id": 364, "logit": 19.0}]`

| 128 | 1 | True | 1.000 | 0.997080 | 0.196382 | 0.968750 | 1.000 | causal_kv | 128 | False |

HF top5 (128/1): `[{"token_id": 9419, "logit": 24.0}, {"token_id": 271, "logit": 16.5}, {"token_id": 21251, "logit": 16.125}, {"token_id": 198, "logit": 15.625}, {"token_id": 248046, "logit": 15.4375}]`

ThinTensor top5 (128/1): `[{"token_id": 9419, "logit": 23.75}, {"token_id": 271, "logit": 16.125}, {"token_id": 198, "logit": 15.9375}, {"token_id": 21251, "logit": 15.875}, {"token_id": 248046, "logit": 15.25}]`

| 128 | 10 | True | 1.000 | 0.997078 | 0.189529 | 1.019531 | 1.000 | causal_kv | 137 | False |

HF top5 (128/10): `[{"token_id": 9419, "logit": 23.875}, {"token_id": 271, "logit": 16.375}, {"token_id": 21251, "logit": 16.0}, {"token_id": 198, "logit": 15.5625}, {"token_id": 248046, "logit": 15.3125}]`

ThinTensor top5 (128/10): `[{"token_id": 9419, "logit": 23.625}, {"token_id": 271, "logit": 16.0}, {"token_id": 198, "logit": 15.9375}, {"token_id": 21251, "logit": 15.625}, {"token_id": 248046, "logit": 15.1875}]`
