# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_coding.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.923347`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.923347 | True | 0.800 | n/a | 1.028886 | 3.234375 | 27/27 |
| 27 | 8 | 0.986524 | True | 1.000 | n/a | 0.261274 | 1.501709 | 34/34 |
| 27 | 32 | 0.989892 | True | 1.000 | n/a | 0.229672 | 1.746094 | 58/58 |
