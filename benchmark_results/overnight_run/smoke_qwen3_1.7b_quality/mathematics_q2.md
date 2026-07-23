# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/qwen3-1.7b-smoke.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `-0.649616`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | -0.555754 | False | 0.000 | 0.000 | 8.233035 | 31.812500 | 27/27 |
| 27 | 8 | -0.649616 | False | 0.000 | 0.000 | 9.524976 | 30.250000 | 34/34 |
| 27 | 32 | -0.511992 | False | 0.000 | 0.000 | 8.004785 | 30.906250 | 58/58 |
