# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/Qwen--Qwen3.5-9B/qwen-9b-fresh.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.973334`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `0.969`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 5 | 1 | 0.988964 | True | 0.800 | 0.969 | 0.342512 | 2.235474 | 5/5 |
| 5 | 8 | 0.973334 | True | 0.800 | 0.969 | 0.451804 | 3.031250 | 12/12 |
| 5 | 32 | 0.981885 | True | 0.800 | 0.969 | 0.443756 | 2.906250 | 36/36 |
