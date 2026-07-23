# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_long_continuation.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.988648`
- Minimum top-5 overlap: `0.600`
- Top-1 same for all records: `False`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 35 | 1 | 0.990797 | True | 0.600 | n/a | 0.338311 | 2.000000 | 35/35 |
| 35 | 8 | 0.988648 | False | 1.000 | n/a | 0.398653 | 2.539062 | 42/42 |
| 35 | 32 | 0.993724 | True | 1.000 | n/a | 0.275956 | 2.109375 | 66/66 |
