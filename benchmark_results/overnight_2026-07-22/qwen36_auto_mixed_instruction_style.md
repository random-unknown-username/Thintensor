# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_instruction_style.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.993319`
- Minimum top-5 overlap: `1.000`
- Top-1 same for all records: `False`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.995903 | True | 1.000 | n/a | 0.290509 | 2.371094 | 27/27 |
| 27 | 8 | 0.994443 | True | 1.000 | n/a | 0.207287 | 1.468750 | 34/34 |
| 27 | 32 | 0.993319 | False | 1.000 | n/a | 0.243860 | 1.750000 | 58/58 |
