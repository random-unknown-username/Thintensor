# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/openai--gpt-oss-20b/gpt-oss-20b.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.933438`
- Minimum top-5 overlap: `0.200`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.750`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.933438 | False | 0.200 | 0.750 | 1.284258 | 8.187500 | 1/1 |
| 1 | 4 | 0.969199 | True | 0.800 | 0.750 | 0.911323 | 4.867188 | 4/4 |
| 8 | 1 | 0.966277 | True | 0.600 | 1.000 | 0.763842 | 3.859375 | 8/8 |
| 8 | 4 | 0.973037 | True | 0.600 | 1.000 | 0.635220 | 4.203125 | 11/11 |
