# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_general.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.995586`
- Minimum top-5 overlap: `1.000`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 28 | 1 | 0.998847 | True | 1.000 | n/a | 0.110919 | 0.812500 | 28/28 |
| 28 | 8 | 0.998757 | True | 1.000 | n/a | 0.084773 | 0.601562 | 35/35 |
| 28 | 32 | 0.995586 | True | 1.000 | n/a | 0.143615 | 1.015625 | 59/59 |
