# ThinTensor mode comparison

- Archive: `/home/satvik/Projects/thintensor-qwen27b-work/output/qwen3.6-27b.thin`
- Reference: `saved full-vocabulary vectors: benchmarks/qwen27b/hf_reference/hf_bf16_fixed_teacher.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.985256`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 12 | 1 | 0.985256 | True | 1.000 | n/a | 0.234799 | 3.125000 | 12/12 |
| 12 | 8 | 0.987099 | True | 0.800 | n/a | 0.248934 | 2.000000 | 19/19 |
| 12 | 32 | 0.996479 | True | 0.800 | n/a | 0.212467 | 1.507812 | 43/43 |
