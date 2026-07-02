# ThinTensor stress validation

- HF model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- HF reference: `eager_cached_decode`
- Cases: `13`

## retained_fp8

- Minimum cosine: `0.993787`
- Minimum top-5 overlap: `0.800`
- All checkpoint top-1 same: `False`
- Minimum teacher-forced token match: `0.906`
- Minimum free-running token match: `0.031`
- Maximum sampling total variation: `0.075521`
- Maximum sampling JS divergence: `0.015362`
- Maximum peak VRAM: `4311492608` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | max TV | max JS | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hello | prompt | 1 | 128 | greedy | 0.996249 | 1.000 | 0.800 | 0.984 | 0.680 | 84 | - | - | 128 | 9437184 | 4173386752 |
| factual | prompt | 5 | 32 | greedy | 0.998352 | 1.000 | 0.800 | 1.000 | 1.000 | 32 | - | - | 36 | 3538944 | 4161496064 |
| reasoning | prompt | 13 | 32 | greedy | 0.998986 | 1.000 | 0.800 | 1.000 | 1.000 | 32 | - | - | 44 | 3538944 | 4161504256 |
| code | prompt | 12 | 32 | greedy | 0.996153 | 1.000 | 1.000 | 0.938 | 0.250 | 6 | - | - | 43 | 3538944 | 4161503232 |
| unicode | prompt | 14 | 32 | greedy | 0.999694 | 1.000 | 0.800 | 0.969 | 0.562 | 18 | - | - | 45 | 3538944 | 4161505280 |
| repetition | prompt | 25 | 32 | greedy | 0.999883 | 1.000 | 0.800 | 1.000 | 1.000 | 32 | - | - | 56 | 4718592 | 4163875840 |
| weird_control | prompt | 21 | 32 | greedy | 0.998468 | 1.000 | 0.800 | 0.906 | 0.031 | 1 | - | - | 52 | 4718592 | 4163871744 |
| multi_turn_chat | chat_template | 83 | 64 | greedy | 0.993787 | 1.000 | 0.800 | 0.969 | 0.250 | 12 | - | - | 146 | 11796480 | 4178123776 |
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.998217 | 1.000 | 1.000 | 0.953 | 0.578 | 34 | 0.066252 | 0.015362 | 77 | 5898240 | 4166256640 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.998410 | 1.000 | 0.800 | 0.953 | 0.047 | 0 | 0.075521 | 0.004675 | 77 | 5898240 | 4166256640 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.998280 | 0.667 | 0.800 | 0.953 | 0.219 | 12 | 0.050551 | 0.007245 | 77 | 5898240 | 4166256640 |
| long_context_512 | long_context | 512 | 16 | greedy | 0.998790 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | - | - | 527 | 38928384 | 4233438208 |
| long_context_1024 | long_context | 1024 | 16 | greedy | 0.999741 | 1.000 | 0.800 | 1.000 | 1.000 | 16 | - | - | 1039 | 76677120 | 4311492608 |
