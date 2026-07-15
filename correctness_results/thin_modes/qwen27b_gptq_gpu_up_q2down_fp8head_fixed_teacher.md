# ThinTensor mode comparison

- Archive: `/home/satvik/Projects/thintensor-qwen27b-work/output/qwen3.6-27b.thin`
- Reference: `saved full-vocabulary vectors: benchmarks/qwen27b/hf_reference/hf_bf16_fixed_teacher.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `-0.162783`
- Minimum top-5 overlap: `0.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 12 | 1 | -0.162783 | False | 0.000 | n/a | 2.821681 | 14.742188 | 12/12 |
| 12 | 8 | 0.548638 | True | 0.200 | n/a | 1.769855 | 13.218750 | 19/19 |
| 12 | 32 | 0.706035 | False | 0.000 | n/a | 1.798292 | 15.960938 | 43/43 |
