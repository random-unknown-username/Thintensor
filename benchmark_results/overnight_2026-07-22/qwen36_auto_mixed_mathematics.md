# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_mathematics.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.985385`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.985385 | True | 1.000 | n/a | 0.277501 | 1.531250 | 27/27 |
| 27 | 8 | 0.989036 | True | 1.000 | n/a | 0.227140 | 1.566406 | 34/34 |
| 27 | 32 | 0.985790 | True | 0.800 | n/a | 0.273688 | 1.477539 | 58/58 |
