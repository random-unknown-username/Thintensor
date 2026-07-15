# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/Qwen--Qwen3.5-9B/qwen-9b.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.983967`
- Minimum top-5 overlap: `0.600`
- Top-1 same for all records: `True`
- Minimum generated-token match: `1.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 5 | 1 | 0.988798 | True | 0.600 | 1.000 | 0.348678 | 2.348633 | 5/5 |
| 5 | 8 | 0.984856 | True | 0.800 | 1.000 | 0.342761 | 2.107422 | 12/12 |
| 5 | 32 | 0.983967 | True | 0.800 | 1.000 | 0.424698 | 2.570312 | 36/36 |
