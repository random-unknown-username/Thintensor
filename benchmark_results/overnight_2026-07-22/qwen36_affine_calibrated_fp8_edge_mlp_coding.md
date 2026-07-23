# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_coding.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.995524`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.995524 | True | 1.000 | n/a | 0.164917 | 1.046875 | 27/27 |
| 27 | 8 | 0.997427 | True | 1.000 | n/a | 0.111737 | 0.968750 | 34/34 |
| 27 | 32 | 0.996978 | True | 0.800 | n/a | 0.125845 | 0.906250 | 58/58 |
