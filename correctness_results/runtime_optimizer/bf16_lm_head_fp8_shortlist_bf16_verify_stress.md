# ThinTensor stress validation

- HF model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- HF reference: `eager_cached_decode`
- Cases: `13`

## bf16_triton_guarded

- Minimum cosine: `0.999109`
- Minimum top-5 overlap: `0.800`
- All checkpoint top-1 same: `False`
- Minimum teacher-forced token match: `0.953`
- Minimum free-running token match: `0.031`
- Maximum sampling total variation: `0.030315`
- Maximum sampling JS divergence: `0.003072`
- Maximum peak VRAM: `6694691328` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | max TV | max JS | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hello | prompt | 1 | 128 | greedy | 0.999109 | 1.000 | 1.000 | 0.992 | 0.820 | 104 | - | - | 128 | 9437184 | 6556076544 |
| factual | prompt | 5 | 32 | greedy | 0.999657 | 1.000 | 1.000 | 1.000 | 1.000 | 32 | - | - | 36 | 3538944 | 6544185856 |
| reasoning | prompt | 13 | 32 | greedy | 0.999918 | 1.000 | 1.000 | 0.969 | 0.344 | 11 | - | - | 44 | 3538944 | 6544194048 |
| code | prompt | 12 | 32 | greedy | 0.999866 | 1.000 | 1.000 | 0.969 | 0.844 | 23 | - | - | 43 | 3538944 | 6544193024 |
| unicode | prompt | 14 | 32 | greedy | 0.999951 | 1.000 | 1.000 | 1.000 | 1.000 | 32 | - | - | 45 | 3538944 | 6544195072 |
| repetition | prompt | 25 | 32 | greedy | 0.999968 | 1.000 | 0.800 | 1.000 | 1.000 | 32 | - | - | 56 | 4718592 | 6546565632 |
| weird_control | prompt | 21 | 32 | greedy | 0.999883 | 1.000 | 0.800 | 1.000 | 1.000 | 32 | - | - | 52 | 4718592 | 6546561536 |
| multi_turn_chat | chat_template | 83 | 64 | greedy | 0.999319 | 1.000 | 1.000 | 0.984 | 0.078 | 1 | - | - | 146 | 11796480 | 6560813568 |
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.999921 | 1.000 | 1.000 | 0.953 | 0.031 | 1 | 0.027234 | 0.000817 | 77 | 5898240 | 6548946432 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.999787 | 1.000 | 0.800 | 0.969 | 0.094 | 0 | 0.030315 | 0.000619 | 77 | 5898240 | 6548946432 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.999942 | 0.667 | 1.000 | 0.969 | 0.219 | 12 | 0.022187 | 0.003072 | 77 | 5898240 | 6548946432 |
| long_context_512 | long_context | 512 | 16 | greedy | 0.999647 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | - | - | 527 | 38928384 | 6616128000 |
| long_context_1024 | long_context | 1024 | 16 | greedy | 0.999955 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | - | - | 1039 | 76677120 | 6694691328 |
