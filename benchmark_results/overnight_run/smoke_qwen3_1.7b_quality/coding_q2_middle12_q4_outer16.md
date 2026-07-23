# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/qwen3-1.7b-smoke.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.819762`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `0.156`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.842596 | False | 0.000 | 0.156 | 2.088549 | 15.250000 | 27/27 |
| 27 | 8 | 0.903496 | True | 0.400 | 0.156 | 1.550665 | 11.500000 | 34/34 |
| 27 | 32 | 0.819762 | False | 0.400 | 0.156 | 2.120447 | 15.562500 | 58/58 |
