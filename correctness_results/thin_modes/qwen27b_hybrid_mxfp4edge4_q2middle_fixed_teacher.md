# ThinTensor mode comparison

- Archive: `/home/satvik/Projects/thintensor-qwen27b-work/output/qwen3.6-27b.thin`
- Reference: `saved full-vocabulary vectors: benchmarks/qwen27b/hf_reference/hf_bf16_fixed_teacher.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.427380`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 12 | 1 | 0.591914 | False | 0.600 | n/a | 1.369060 | 6.687500 | 12/12 |
| 12 | 8 | 0.427380 | True | 0.800 | n/a | 1.667021 | 9.593750 | 19/19 |
| 12 | 32 | 0.870305 | False | 0.000 | n/a | 1.347369 | 10.015625 | 43/43 |
