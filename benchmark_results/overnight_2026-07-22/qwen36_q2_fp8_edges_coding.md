# ThinTensor mode comparison

- Archive: `/home/satvik/.cache/thintensor/archives/Qwen--Qwen3.6-27B.thin`
- Reference: `saved full-vocabulary vectors: benchmark_results/overnight_2026-07-22/qwen36_hf_reference_coding.npz`
- Candidate: `ThinTensor selected runtime flags`
- Minimum cosine: `0.219842`
- Minimum top-5 overlap: `0.200`
- Top-1 same for all records: `False`
- Minimum generated-token match: `not measured for saved fixed-teacher reference`

| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 27 | 1 | 0.585592 | False | 0.400 | n/a | 2.377056 | 15.554688 | 27/27 |
| 27 | 8 | 0.219842 | False | 0.400 | n/a | 2.656597 | 15.343750 | 34/34 |
| 27 | 32 | 0.351476 | False | 0.200 | n/a | 2.703216 | 15.500000 | 58/58 |
