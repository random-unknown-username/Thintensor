# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/qwen3-1.7b-smoke.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `-0.659333`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | -0.659333 | False | 0.000 | 0.000 | 9.714808 | 31.750000 | 27/27 |
| 27 | 8 | -0.590024 | False | 0.000 | 0.000 | 8.737046 | 28.875000 | 34/34 |
| 27 | 32 | -0.502849 | False | 0.000 | 0.000 | 8.268166 | 28.750000 | 58/58 |
