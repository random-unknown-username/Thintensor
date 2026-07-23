# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/qwen3-1.7b-smoke.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `-0.632338`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | -0.610733 | False | 0.000 | 0.000 | 9.489192 | 32.218750 | 27/27 |
| 27 | 8 | -0.632338 | False | 0.000 | 0.000 | 9.453184 | 28.343750 | 34/34 |
| 27 | 32 | -0.547423 | False | 0.000 | 0.000 | 9.107662 | 27.625000 | 58/58 |
