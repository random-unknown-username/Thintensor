# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/Qwen--Qwen3.5-9B/qwen-9b.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.997920`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `1.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.997920 | True | 1.000 | 1.000 | 0.092874 | 0.640625 | 1/1 |
| 1 | 4 | 0.998373 | True | 1.000 | 1.000 | 0.070016 | 0.429688 | 4/4 |
| 8 | 1 | 0.998859 | True | 0.800 | 1.000 | 0.050411 | 0.312500 | 8/8 |
| 8 | 4 | 0.999288 | True | 1.000 | 1.000 | 0.046170 | 0.281250 | 11/11 |
