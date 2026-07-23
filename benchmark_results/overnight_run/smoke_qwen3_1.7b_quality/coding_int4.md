# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/qwen3-1.7b-smoke.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.993710`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `0.969`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.995107 | True | 0.800 | 0.969 | 0.382012 | 2.187500 | 27/27 |
| 27 | 8 | 0.994522 | True | 1.000 | 0.969 | 0.379856 | 2.328125 | 34/34 |
| 27 | 32 | 0.993710 | True | 1.000 | 0.969 | 0.371105 | 2.242188 | 58/58 |
