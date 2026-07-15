# ThinTensor mode comparison

- Archive: `/home/satvik/Projects/thintensor-qwen27b-work/output/qwen3.6-27b.thin`
- Reference: `saved full-vocabulary vectors: benchmarks/qwen27b/hf_reference/hf_bf16_fixed_teacher.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.993508`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 12 | 1 | 0.993508 | True | 1.000 | n/a | 0.154102 | 1.531250 | 12/12 |
| 12 | 8 | 0.994784 | True | 1.000 | n/a | 0.157863 | 1.281250 | 19/19 |
| 12 | 32 | 0.998853 | True | 0.800 | n/a | 0.124204 | 0.765625 | 43/43 |
