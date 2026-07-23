# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/qwen3-1.7b-smoke.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `-0.630817`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 36 | 1 | -0.630817 | False | 0.000 | 0.000 | 9.019973 | 27.781250 | 36/36 |
| 36 | 8 | -0.494260 | False | 0.000 | 0.000 | 8.117990 | 30.375000 | 43/43 |
| 36 | 32 | -0.475914 | False | 0.000 | 0.000 | 7.805316 | 28.562500 | 67/67 |
