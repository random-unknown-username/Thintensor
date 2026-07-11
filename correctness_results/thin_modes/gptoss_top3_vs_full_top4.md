# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/openai--gpt-oss-20b/gpt-oss-20b.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.983904`
- Minimum top-5 overlap: `0.600`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.750`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.983904 | False | 0.600 | 0.750 | 0.647758 | 4.312500 | 1/1 |
| 1 | 4 | 0.995109 | True | 0.800 | 0.750 | 0.367720 | 2.625000 | 4/4 |
| 8 | 1 | 0.993891 | True | 0.800 | 1.000 | 0.440330 | 2.281250 | 8/8 |
| 8 | 4 | 0.993420 | True | 0.800 | 1.000 | 0.447992 | 2.484375 | 11/11 |
