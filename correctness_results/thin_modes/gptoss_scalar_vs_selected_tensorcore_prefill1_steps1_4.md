# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/openai--gpt-oss-20b/gpt-oss-20b.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.999961`
- Minimum top-5 overlap: `1.000`
- Top-1 same for all records: `True`
- Minimum generated-token match: `1.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.999981 | True | 1.000 | 1.000 | 0.018238 | 0.125000 | 1/1 |
| 1 | 4 | 0.999961 | True | 1.000 | 1.000 | 0.025925 | 0.187500 | 4/4 |
