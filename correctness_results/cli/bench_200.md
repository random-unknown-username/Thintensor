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
- Correctness tier: `experimental`
- Attention-equivalent: `true`
- Ranking-equivalent: `false`
- Quantized experimental: `false`
- LM-head FP8: `false`
- MLP FP8: `false`
- Config mismatches: `3`

| prefill | step | top1 | top5 overlap | cosine | mean abs err | max abs err | token match | attention | KV tokens | not HF equivalent |
|---:|---:|:---:|---:|---:|---:|---:|---:|:---|---:|:---:|
| 1 | 200 | True | 0.800 | 0.939758 | 0.696231 | 4.726562 | 0.960 | causal_kv | 200 | False |

HF top5 (1/200): `[{"token_id": 1620, "logit": 27.875}, {"token_id": 1493, "logit": 15.5}, {"token_id": 12085, "logit": 15.125}, {"token_id": 4320, "logit": 14.875}, {"token_id": 13321, "logit": 14.375}]`

ThinTensor top5 (1/200): `[{"token_id": 1620, "logit": 27.875}, {"token_id": 12085, "logit": 15.6875}, {"token_id": 4320, "logit": 15.1875}, {"token_id": 1493, "logit": 15.1875}, {"token_id": 4495, "logit": 14.625}]`
