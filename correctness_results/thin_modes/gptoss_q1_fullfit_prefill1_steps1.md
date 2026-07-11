# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/openai--gpt-oss-20b/gpt-oss-20b.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.991415`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `1.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.991415 | True | 0.800 | 1.000 | 0.397124 | 2.687500 | 1/1 |
