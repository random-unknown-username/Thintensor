# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_long_continuation.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.993784`
- Minimum top-5 overlap: `1.000`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 35 | 1 | 0.993784 | True | 1.000 | n/a | 0.269201 | 2.281250 | 35/35 |
| 35 | 8 | 0.995880 | True | 1.000 | n/a | 0.275749 | 1.555664 | 42/42 |
| 35 | 32 | 0.998131 | True | 1.000 | n/a | 0.139210 | 0.875000 | 66/66 |
