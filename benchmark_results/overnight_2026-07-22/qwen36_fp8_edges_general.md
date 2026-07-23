# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_general.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.985619`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 28 | 1 | 0.992669 | True | 0.800 | n/a | 0.283258 | 1.851562 | 28/28 |
| 28 | 8 | 0.985619 | True | 1.000 | n/a | 0.391955 | 2.078125 | 35/35 |
| 28 | 32 | 0.989403 | True | 1.000 | n/a | 0.226605 | 1.601562 | 59/59 |
