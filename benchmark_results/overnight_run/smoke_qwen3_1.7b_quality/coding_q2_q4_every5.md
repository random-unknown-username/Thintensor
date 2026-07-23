# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/qwen3-1.7b-smoke.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `-0.647188`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | -0.606398 | False | 0.000 | 0.000 | 9.402751 | 34.375000 | 27/27 |
| 27 | 8 | -0.647188 | False | 0.200 | 0.000 | 9.717795 | 29.625000 | 34/34 |
| 27 | 32 | -0.584051 | False | 0.000 | 0.000 | 9.712632 | 31.437500 | 58/58 |
