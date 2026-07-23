# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/qwen3-1.7b-smoke.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.999990`
- Minimum top-5 overlap: `1.000`
- Top-1 same for all records: `True`
- Minimum generated-token match: `1.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.999997 | True | 1.000 | 1.000 | 0.000000 | 0.000000 | 27/27 |
| 27 | 8 | 0.999990 | True | 1.000 | 1.000 | 0.000000 | 0.000000 | 34/34 |
| 27 | 32 | 0.999994 | True | 1.000 | 1.000 | 0.000000 | 0.000000 | 58/58 |
