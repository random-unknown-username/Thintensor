# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/Qwen--Qwen3.5-9B/qwen-9b-fresh.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.783342`
- Minimum top-5 overlap: `0.600`
- Top-1 same for all records: `True`
- Minimum generated-token match: `0.844`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.997151 | True | 0.800 | 0.938 | 0.254490 | 1.593750 | 1/1 |
| 1 | 8 | 0.985826 | True | 0.800 | 0.938 | 0.571355 | 3.796875 | 8/8 |
| 1 | 32 | 0.990355 | True | 0.800 | 0.938 | 0.337603 | 2.246094 | 32/32 |
| 8 | 1 | 0.990978 | True | 1.000 | 0.844 | 0.366891 | 2.460938 | 8/8 |
| 8 | 8 | 0.956675 | True | 1.000 | 0.844 | 0.674145 | 3.681641 | 15/15 |
| 8 | 32 | 0.783342 | True | 0.600 | 0.844 | 1.258385 | 8.007812 | 39/39 |
