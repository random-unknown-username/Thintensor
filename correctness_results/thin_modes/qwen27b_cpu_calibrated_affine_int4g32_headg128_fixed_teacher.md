# ThinTensor mode comparison

- Archive: `/home/satvik/Projects/thintensor-qwen27b-work/output/qwen3.6-27b.thin`
- Reference: `saved full-vocabulary vectors: benchmarks/qwen27b/hf_reference/hf_bf16_fixed_teacher.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.980371`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 12 | 1 | 0.984045 | True | 1.000 | n/a | 0.248805 | 3.375000 | 12/12 |
| 12 | 8 | 0.980371 | True | 1.000 | n/a | 0.306856 | 1.965820 | 19/19 |
| 12 | 32 | 0.995344 | True | 0.800 | n/a | 0.244748 | 1.867188 | 43/43 |
