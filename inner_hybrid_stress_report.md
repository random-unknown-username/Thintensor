# ThinTensor stress validation

- HF model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- HF reference: `eager_cached_decode`
- Cases: `7`

## hybrid_mlp_8_28

- Minimum cosine: `0.993787`
- Minimum top-5 overlap: `0.800`
- All checkpoint top-1 same: `False`
- Minimum teacher-forced token match: `0.906`
- Minimum free-running token match: `0.031`
- Maximum sampling total variation: `0.075521`
- Maximum sampling JS divergence: `0.015362`
- Maximum peak VRAM: `4178123264` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | max TV | max JS | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hello | prompt | 1 | 128 | greedy | 0.996249 | 1.000 | 0.800 | 0.984 | 0.680 | 84 | - | - | 128 | 9437184 | 4173386240 |
| code | prompt | 12 | 32 | greedy | 0.996153 | 1.000 | 1.000 | 0.938 | 0.250 | 6 | - | - | 43 | 3538944 | 4161502720 |
| weird_control | prompt | 21 | 32 | greedy | 0.998468 | 1.000 | 0.800 | 0.906 | 0.031 | 1 | - | - | 52 | 4718592 | 4163871232 |
| multi_turn_chat | chat_template | 83 | 64 | greedy | 0.993787 | 1.000 | 0.800 | 0.969 | 0.250 | 12 | - | - | 146 | 11796480 | 4178123264 |
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.998217 | 1.000 | 1.000 | 0.953 | 0.578 | 34 | 0.066252 | 0.015362 | 77 | 5898240 | 4166256128 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.998410 | 1.000 | 0.800 | 0.953 | 0.047 | 0 | 0.075521 | 0.004675 | 77 | 5898240 | 4166256128 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.998280 | 0.667 | 0.800 | 0.953 | 0.219 | 12 | 0.050551 | 0.007245 | 77 | 5898240 | 4166256128 |

## hybrid_mlp_10_26

- Minimum cosine: `0.994296`
- Minimum top-5 overlap: `0.600`
- All checkpoint top-1 same: `False`
- Minimum teacher-forced token match: `0.906`
- Minimum free-running token match: `0.000`
- Maximum sampling total variation: `0.096167`
- Maximum sampling JS divergence: `0.016457`
- Maximum peak VRAM: `4270365184` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | max TV | max JS | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hello | prompt | 1 | 128 | greedy | 0.996115 | 1.000 | 0.800 | 0.984 | 0.680 | 84 | - | - | 128 | 9437184 | 4265628160 |
| code | prompt | 12 | 32 | greedy | 0.996893 | 1.000 | 1.000 | 0.938 | 0.250 | 6 | - | - | 43 | 3538944 | 4253744640 |
| weird_control | prompt | 21 | 32 | greedy | 0.998716 | 1.000 | 0.800 | 0.906 | 0.031 | 1 | - | - | 52 | 4718592 | 4256113152 |
| multi_turn_chat | chat_template | 83 | 64 | greedy | 0.994296 | 1.000 | 0.800 | 0.969 | 0.250 | 12 | - | - | 146 | 11796480 | 4270365184 |
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.997670 | 1.000 | 1.000 | 0.984 | 0.578 | 34 | 0.073596 | 0.016457 | 77 | 5898240 | 4258498048 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.998255 | 1.000 | 0.600 | 0.969 | 0.000 | 0 | 0.096167 | 0.006331 | 77 | 5898240 | 4258498048 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.998989 | 0.667 | 0.800 | 0.938 | 0.031 | 1 | 0.055345 | 0.009077 | 77 | 5898240 | 4258498048 |
