# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_long_continuation.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.994478`
- Minimum top-5 overlap: `1.000`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 35 | 1 | 0.994478 | True | 1.000 | n/a | 0.254115 | 2.156250 | 35/35 |
| 35 | 8 | 0.996102 | True | 1.000 | n/a | 0.264365 | 1.482422 | 42/42 |
| 35 | 32 | 0.997934 | True | 1.000 | n/a | 0.152350 | 0.906250 | 66/66 |
