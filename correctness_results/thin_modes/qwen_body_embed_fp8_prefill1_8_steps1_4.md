# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/Qwen--Qwen3.5-9B/qwen-9b.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.997576`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `1.000`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.997576 | True | 0.800 | 1.000 | 0.100173 | 0.648438 | 1/1 |
| 1 | 4 | 0.997773 | True | 1.000 | 1.000 | 0.080731 | 0.484375 | 4/4 |
| 8 | 1 | 0.998472 | True | 0.800 | 1.000 | 0.058409 | 0.367188 | 8/8 |
| 8 | 4 | 0.998956 | True | 1.000 | 1.000 | 0.056070 | 0.343262 | 11/11 |
