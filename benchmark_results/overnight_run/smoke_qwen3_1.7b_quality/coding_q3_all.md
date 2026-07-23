# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/qwen3-1.7b-smoke.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `-0.419290`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.031`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | -0.137617 | False | 0.000 | 0.031 | 5.090559 | 23.335938 | 27/27 |
| 27 | 8 | -0.043025 | False | 0.400 | 0.031 | 4.807048 | 19.343750 | 34/34 |
| 27 | 32 | -0.419290 | False | 0.200 | 0.031 | 7.601760 | 24.687500 | 58/58 |
