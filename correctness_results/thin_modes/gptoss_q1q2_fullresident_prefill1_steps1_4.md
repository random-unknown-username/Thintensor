# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/openai--gpt-oss-20b/gpt-oss-20b.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.983678`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `1.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.987381 | True | 0.800 | 1.000 | 0.489157 | 3.541962 | 1/1 |
| 1 | 4 | 0.983678 | True | 0.800 | 1.000 | 0.658597 | 6.000000 | 4/4 |
