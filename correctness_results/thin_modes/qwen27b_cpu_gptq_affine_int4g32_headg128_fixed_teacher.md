# ThinTensor mode comparison

- Archive: `/home/satvik/Projects/thintensor-qwen27b-work/output/qwen3.6-27b.thin`
- Reference: `saved full-vocabulary vectors: benchmarks/qwen27b/hf_reference/hf_bf16_fixed_teacher.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.986787`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 12 | 1 | 0.986787 | True | 1.000 | n/a | 0.224881 | 2.875000 | 12/12 |
| 12 | 8 | 0.988847 | True | 1.000 | n/a | 0.234372 | 2.000000 | 19/19 |
| 12 | 32 | 0.996149 | True | 0.800 | n/a | 0.223037 | 1.671875 | 43/43 |
