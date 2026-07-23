# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/qwen3-1.7b-smoke.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `-0.547874`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.031`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | -0.508251 | False | 0.000 | 0.031 | 7.098289 | 29.375000 | 27/27 |
| 27 | 8 | -0.547874 | False | 0.000 | 0.031 | 7.221553 | 28.187500 | 34/34 |
| 27 | 32 | -0.515595 | False | 0.000 | 0.031 | 6.895082 | 29.531250 | 58/58 |
