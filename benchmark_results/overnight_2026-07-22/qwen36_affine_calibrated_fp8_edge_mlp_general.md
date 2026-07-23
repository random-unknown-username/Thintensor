# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_general.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.995185`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 28 | 1 | 0.998648 | True | 0.800 | n/a | 0.120351 | 0.796875 | 28/28 |
| 28 | 8 | 0.998663 | True | 1.000 | n/a | 0.088041 | 0.593750 | 35/35 |
| 28 | 32 | 0.995185 | True | 1.000 | n/a | 0.150568 | 1.101562 | 59/59 |
