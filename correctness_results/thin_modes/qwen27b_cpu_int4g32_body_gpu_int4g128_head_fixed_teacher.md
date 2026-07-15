# ThinTensor mode comparison

- Archive: `/home/satvik/Projects/thintensor-qwen27b-work/output/qwen3.6-27b.thin`
- Reference: `saved full-vocabulary vectors: benchmarks/qwen27b/hf_reference/hf_bf16_fixed_teacher.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.950113`
- Minimum top-5 overlap: `1.000`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 12 | 1 | 0.950113 | True | 1.000 | n/a | 0.435203 | 2.742188 | 12/12 |
| 12 | 8 | 0.963512 | True | 1.000 | n/a | 0.464794 | 2.847656 | 19/19 |
| 12 | 32 | 0.992258 | True | 1.000 | n/a | 0.315852 | 2.333984 | 43/43 |
