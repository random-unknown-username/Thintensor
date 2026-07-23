# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/qwen3-1.7b-smoke.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `-0.654856`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 28 | 1 | -0.654856 | False | 0.000 | 0.000 | 9.740758 | 32.062500 | 28/28 |
| 28 | 8 | -0.105241 | False | 0.000 | 0.000 | 6.118046 | 30.765625 | 35/35 |
| 28 | 32 | -0.590428 | False | 0.000 | 0.000 | 8.611145 | 33.437500 | 59/59 |
