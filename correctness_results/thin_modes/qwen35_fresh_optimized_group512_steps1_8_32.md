# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/models/Qwen--Qwen3.5-9B/qwen-9b-fresh.thin`
- Reference: `ThinTensor BF16 streamed weights`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.896589`
- Minimum top-5 overlap: `0.400`
- Top-1 same for all records: `True`
- Minimum generated-token match: `0.906`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.996153 | True | 0.800 | 0.938 | 0.299940 | 1.875000 | 1/1 |
| 1 | 8 | 0.970073 | True | 0.600 | 0.938 | 0.817730 | 5.062500 | 8/8 |
| 1 | 32 | 0.990690 | True | 1.000 | 0.938 | 0.330595 | 2.105469 | 32/32 |
| 8 | 1 | 0.988259 | True | 1.000 | 0.906 | 0.397476 | 2.500000 | 8/8 |
| 8 | 8 | 0.974336 | True | 0.800 | 0.906 | 0.529724 | 3.328125 | 15/15 |
| 8 | 32 | 0.896589 | True | 0.400 | 0.906 | 1.263254 | 9.125000 | 39/39 |
