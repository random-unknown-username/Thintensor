# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_general.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.995266`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `True`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 28 | 1 | 0.998491 | True | 0.800 | n/a | 0.127689 | 0.835938 | 28/28 |
| 28 | 8 | 0.998655 | True | 1.000 | n/a | 0.089325 | 0.593750 | 35/35 |
| 28 | 32 | 0.995266 | True | 1.000 | n/a | 0.149284 | 1.070312 | 59/59 |
