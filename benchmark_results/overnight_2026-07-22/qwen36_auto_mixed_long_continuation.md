# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_long_continuation.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.988391`
- Minimum top-5 overlap: `0.600`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 35 | 1 | 0.988391 | True | 0.600 | n/a | 0.393812 | 2.687500 | 35/35 |
| 35 | 8 | 0.989238 | True | 0.800 | n/a | 0.388806 | 2.601562 | 42/42 |
| 35 | 32 | 0.994489 | True | 1.000 | n/a | 0.249872 | 1.932617 | 66/66 |
