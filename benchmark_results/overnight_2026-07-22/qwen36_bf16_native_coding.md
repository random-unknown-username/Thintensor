# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_coding.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.999883`
- Minimum top-5 overlap: `1.000`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.999883 | True | 1.000 | n/a | 0.023514 | 0.187500 | 27/27 |
| 27 | 8 | 0.999889 | True | 1.000 | n/a | 0.021373 | 0.187500 | 34/34 |
| 27 | 32 | 0.999918 | True | 1.000 | n/a | 0.018508 | 0.125977 | 58/58 |
