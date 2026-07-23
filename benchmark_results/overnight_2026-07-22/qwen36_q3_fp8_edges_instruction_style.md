# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_instruction_style.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.928329`
- Minimum top-5 overlap: `0.800`
- Top-1 same for all records: `False`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.974212 | True | 0.800 | n/a | 0.714781 | 5.820312 | 27/27 |
| 27 | 8 | 0.928329 | False | 0.800 | n/a | 0.884598 | 6.218750 | 34/34 |
| 27 | 32 | 0.936021 | False | 0.800 | n/a | 0.776930 | 6.687500 | 58/58 |
