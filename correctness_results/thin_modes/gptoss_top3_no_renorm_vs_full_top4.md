# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/openai--gpt-oss-20b/gpt-oss-20b.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.923205`
- Minimum top-5 overlap: `0.200`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.500`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.962921 | True | 0.200 | 0.500 | 1.011518 | 7.500000 | 1/1 |
| 1 | 4 | 0.954177 | False | 0.200 | 0.500 | 1.121123 | 7.789062 | 4/4 |
| 8 | 1 | 0.940701 | True | 0.600 | 0.500 | 0.935729 | 7.250000 | 8/8 |
| 8 | 4 | 0.923205 | False | 0.800 | 0.500 | 1.040578 | 8.875000 | 11/11 |
