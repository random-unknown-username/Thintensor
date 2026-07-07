# HF vs ThinTensor correctness report

- Model: `/home/satvik/.cache/thintensor/models/allenai--OLMoE-1B-7B-0924-Instruct`
- Archive: `/home/satvik/.cache/thintensor/archives/allenai--OLMoE-1B-7B-0924-Instruct.thin`
- Dtype: `bf16`
- Kernel backend: `triton-matvec`
- Weight residency: `stream`
- GPU weight budget: `7456253543` bytes
- Prompt: `Hello`
- HF attention implementation: `sdpa`
- HF-equivalent: `false`
- Correctness tier: `experimental`
- Attention-equivalent: `true`
- Ranking-equivalent: `false`
- Quantized experimental: `false`
- LM-head FP8: `false`
- MLP FP8: `false`
- Config mismatches: `0`

| prefill | step | top1 | top5 overlap | cosine | mean abs err | max abs err | token match | attention | KV tokens | not HF equivalent |
|---:|---:|:---:|---:|---:|---:|---:|---:|:---|---:|:---:|
| 1 | 1 | True | 1.000 | 0.999666 | 0.020310 | 0.113281 | 1.000 | causal_kv | 1 | False |

HF top5 (1/1): `[{"token_id": 47111, "logit": 6.625}, {"token_id": 20291, "logit": 6.4375}, {"token_id": 33007, "logit": 5.5625}, {"token_id": 16877, "logit": 5.46875}, {"token_id": 14026, "logit": 5.03125}]`

ThinTensor top5 (1/1): `[{"token_id": 47111, "logit": 6.65625}, {"token_id": 20291, "logit": 6.4375}, {"token_id": 33007, "logit": 5.5625}, {"token_id": 16877, "logit": 5.5}, {"token_id": 14026, "logit": 5.09375}]`

| 1 | 10 | True | 1.000 | 0.999993 | 0.020641 | 0.125000 | 0.900 | causal_kv | 10 | False |

HF top5 (1/10): `[{"token_id": 368, "logit": 15.0}, {"token_id": 3890, "logit": 14.4375}, {"token_id": 1642, "logit": 13.3125}, {"token_id": 2748, "logit": 13.125}, {"token_id": 2085, "logit": 12.75}]`

ThinTensor top5 (1/10): `[{"token_id": 368, "logit": 15.0625}, {"token_id": 3890, "logit": 14.4375}, {"token_id": 1642, "logit": 13.3125}, {"token_id": 2748, "logit": 13.125}, {"token_id": 2085, "logit": 12.75}]`

| 128 | 1 | True | 1.000 | 0.999855 | 0.013807 | 0.093750 | 1.000 | causal_kv | 128 | False |

HF top5 (128/1): `[{"token_id": 47111, "logit": 6.625}, {"token_id": 20291, "logit": 6.4375}, {"token_id": 33007, "logit": 5.5625}, {"token_id": 16877, "logit": 5.46875}, {"token_id": 14026, "logit": 5.03125}]`

ThinTensor top5 (128/1): `[{"token_id": 47111, "logit": 6.65625}, {"token_id": 20291, "logit": 6.4375}, {"token_id": 33007, "logit": 5.5625}, {"token_id": 16877, "logit": 5.5}, {"token_id": 14026, "logit": 5.09375}]`

| 128 | 10 | True | 0.800 | 0.999657 | 0.063313 | 0.375000 | 0.900 | causal_kv | 137 | False |

HF top5 (128/10): `[{"token_id": 5996, "logit": 9.75}, {"token_id": 936, "logit": 9.5625}, {"token_id": 783, "logit": 9.5625}, {"token_id": 1201, "logit": 9.3125}, {"token_id": 4064, "logit": 9.125}]`

ThinTensor top5 (128/10): `[{"token_id": 5996, "logit": 10.0}, {"token_id": 936, "logit": 9.625}, {"token_id": 783, "logit": 9.625}, {"token_id": 1201, "logit": 9.5}, {"token_id": 1542, "logit": 9.375}]`
