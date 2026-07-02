# ThinTensor stress validation

- HF model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- HF reference: `eager_cached_decode`
- Cases: `9`

## gate_up_fp8

- Minimum cosine: `0.992117`
- Minimum top-5 overlap: `0.800`
- All checkpoint top-1 same: `False`
- Minimum teacher-forced token match: `0.938`
- Minimum free-running token match: `0.031`
- Maximum sampling total variation: `0.104134`
- Maximum sampling JS divergence: `0.015362`
- Maximum peak VRAM: `4639332864` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | max TV | max JS | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hello | prompt | 1 | 128 | greedy | 0.997558 | 1.000 | 0.800 | 0.984 | 0.680 | 84 | - | - | 128 | 9437184 | 4634595840 |
| reasoning | prompt | 13 | 32 | greedy | 0.999344 | 1.000 | 1.000 | 1.000 | 1.000 | 32 | - | - | 44 | 3538944 | 4622713344 |
| code | prompt | 12 | 32 | greedy | 0.996845 | 1.000 | 1.000 | 0.969 | 0.844 | 23 | - | - | 43 | 3538944 | 4622712320 |
| unicode | prompt | 14 | 32 | greedy | 0.999836 | 1.000 | 1.000 | 0.938 | 0.562 | 18 | - | - | 45 | 3538944 | 4622714368 |
| weird_control | prompt | 21 | 32 | greedy | 0.998558 | 1.000 | 0.800 | 0.938 | 0.031 | 1 | - | - | 52 | 4718592 | 4625080832 |
| multi_turn_chat | chat_template | 83 | 64 | greedy | 0.992117 | 1.000 | 0.800 | 0.969 | 0.250 | 12 | - | - | 146 | 11796480 | 4639332864 |
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.999490 | 1.000 | 1.000 | 0.969 | 0.578 | 34 | 0.064074 | 0.015362 | 77 | 5898240 | 4627465728 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.998674 | 1.000 | 0.800 | 0.969 | 0.094 | 0 | 0.104134 | 0.007994 | 77 | 5898240 | 4627465728 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.999447 | 0.667 | 1.000 | 0.953 | 0.234 | 12 | 0.046855 | 0.006201 | 77 | 5898240 | 4627465728 |

## hybrid_mlp_fp8

- Minimum cosine: `0.992224`
- Minimum top-5 overlap: `0.800`
- All checkpoint top-1 same: `True`
- Minimum teacher-forced token match: `0.906`
- Minimum free-running token match: `0.031`
- Maximum sampling total variation: `0.086321`
- Maximum sampling JS divergence: `0.016457`
- Maximum peak VRAM: `4085881344` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | max TV | max JS | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hello | prompt | 1 | 128 | greedy | 0.996027 | 1.000 | 0.800 | 0.984 | 0.680 | 84 | - | - | 128 | 9437184 | 4081144320 |
| reasoning | prompt | 13 | 32 | greedy | 0.998999 | 1.000 | 1.000 | 1.000 | 1.000 | 32 | - | - | 44 | 3538944 | 4069261824 |
| code | prompt | 12 | 32 | greedy | 0.992918 | 1.000 | 1.000 | 0.969 | 0.844 | 23 | - | - | 43 | 3538944 | 4069260800 |
| unicode | prompt | 14 | 32 | greedy | 0.999598 | 1.000 | 1.000 | 0.969 | 0.562 | 18 | - | - | 45 | 3538944 | 4069262848 |
| weird_control | prompt | 21 | 32 | greedy | 0.997937 | 1.000 | 0.800 | 0.906 | 0.031 | 1 | - | - | 52 | 4718592 | 4071629312 |
| multi_turn_chat | chat_template | 83 | 64 | greedy | 0.992224 | 1.000 | 0.800 | 0.969 | 0.250 | 12 | - | - | 146 | 11796480 | 4085881344 |
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.998050 | 1.000 | 1.000 | 0.938 | 0.344 | 20 | 0.073184 | 0.016457 | 77 | 5898240 | 4074014208 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.998067 | 1.000 | 0.800 | 0.953 | 0.078 | 0 | 0.086321 | 0.005039 | 77 | 5898240 | 4074014208 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.997945 | 1.000 | 0.800 | 0.953 | 0.234 | 12 | 0.053958 | 0.010225 | 77 | 5898240 | 4074014208 |
