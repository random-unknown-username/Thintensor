# ThinTensor mode comparison

- Archive: `/home/satvik/Projects/thintensor-qwen27b-work/output/qwen3.6-27b.thin`
- Reference: `saved full-vocabulary vectors: benchmarks/qwen27b/hf_reference/hf_bf16_fixed_teacher.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `-0.160552`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 12 | 1 | -0.160552 | False | 0.000 | n/a | 2.796158 | 15.171875 | 12/12 |
| 12 | 8 | 0.545284 | True | 0.200 | n/a | 1.757834 | 13.039062 | 19/19 |
| 12 | 32 | 0.701131 | False | 0.000 | n/a | 1.810042 | 16.007812 | 43/43 |
