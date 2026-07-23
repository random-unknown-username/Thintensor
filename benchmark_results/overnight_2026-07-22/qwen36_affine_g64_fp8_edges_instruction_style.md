# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_instruction_style.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.996099`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `False`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.998469 | True | 1.000 | n/a | 0.182147 | 1.063477 | 27/27 |
| 27 | 8 | 0.996099 | True | 0.800 | n/a | 0.187907 | 1.265625 | 34/34 |
| 27 | 32 | 0.997608 | False | 0.800 | n/a | 0.142719 | 0.976562 | 58/58 |
