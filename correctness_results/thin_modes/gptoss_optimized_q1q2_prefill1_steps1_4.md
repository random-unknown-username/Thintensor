# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/openai--gpt-oss-20b/gpt-oss-20b.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.983299`
- Minimum top-5 overlap: `0.600`
- Top-1 same for all records: `True`
- Minimum generated-token match: `0.750`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.987376 | True | 0.800 | 0.750 | 0.490946 | 3.526337 | 1/1 |
| 1 | 4 | 0.983299 | True | 0.600 | 0.750 | 0.678812 | 5.865723 | 4/4 |
