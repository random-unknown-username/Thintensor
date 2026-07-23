# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_mathematics.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.995472`
- Minimum top-5 overlap: `1.000`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.995522 | True | 1.000 | n/a | 0.147711 | 1.218750 | 27/27 |
| 27 | 8 | 0.995472 | True | 1.000 | n/a | 0.147264 | 0.984375 | 34/34 |
| 27 | 32 | 0.996477 | True | 1.000 | n/a | 0.128821 | 1.125000 | 58/58 |
