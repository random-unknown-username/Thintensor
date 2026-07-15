# ThinTensor mode comparison

- Archive: `/home/satvik/Projects/thintensor-qwen27b-work/output/qwen3.6-27b.thin`
- Reference: `saved full-vocabulary vectors: benchmarks/qwen27b/hf_reference/hf_bf16_fixed_teacher.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.957029`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 12 | 1 | 0.957029 | True | 1.000 | n/a | 0.402112 | 4.000000 | 12/12 |
| 12 | 8 | 0.962324 | True | 0.800 | n/a | 0.427990 | 2.929688 | 19/19 |
| 12 | 32 | 0.989311 | True | 0.800 | n/a | 0.370738 | 2.718750 | 43/43 |
