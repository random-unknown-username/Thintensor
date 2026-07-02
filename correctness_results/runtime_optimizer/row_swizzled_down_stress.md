# ThinTensor stress validation

- HF model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- HF reference: `eager_cached_decode`
- Cases: `13`

## quality_8_28_rowswizzle

- Minimum cosine: `0.992827`
- Minimum top-5 overlap: `0.800`
- All checkpoint top-1 same: `False`
- Minimum teacher-forced token match: `0.906`
- Minimum free-running token match: `0.031`
- Maximum sampling total variation: `0.076197`
- Maximum sampling JS divergence: `0.015925`
- Maximum peak VRAM: `4311493120` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | max TV | max JS | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hello | prompt | 1 | 128 | greedy | 0.996060 | 1.000 | 0.800 | 0.977 | 0.680 | 84 | - | - | 128 | 9437184 | 4173387264 |
| factual | prompt | 5 | 32 | greedy | 0.998627 | 1.000 | 0.800 | 1.000 | 1.000 | 32 | - | - | 36 | 3538944 | 4161496576 |
| reasoning | prompt | 13 | 32 | greedy | 0.999151 | 1.000 | 1.000 | 1.000 | 1.000 | 32 | - | - | 44 | 3538944 | 4161504768 |
| code | prompt | 12 | 32 | greedy | 0.995926 | 1.000 | 1.000 | 0.938 | 0.250 | 6 | - | - | 43 | 3538944 | 4161503744 |
| unicode | prompt | 14 | 32 | greedy | 0.999740 | 1.000 | 1.000 | 0.969 | 0.562 | 18 | - | - | 45 | 3538944 | 4161505792 |
| repetition | prompt | 25 | 32 | greedy | 0.999871 | 1.000 | 0.800 | 0.969 | 0.531 | 2 | - | - | 56 | 4718592 | 4163876352 |
| weird_control | prompt | 21 | 32 | greedy | 0.998479 | 1.000 | 0.800 | 0.906 | 0.031 | 1 | - | - | 52 | 4718592 | 4163872256 |
| multi_turn_chat | chat_template | 83 | 64 | greedy | 0.992827 | 1.000 | 0.800 | 0.953 | 0.031 | 1 | - | - | 146 | 11796480 | 4178124288 |
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.997895 | 1.000 | 1.000 | 0.953 | 0.578 | 34 | 0.054607 | 0.015925 | 77 | 5898240 | 4166257152 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.998217 | 1.000 | 0.800 | 0.969 | 0.031 | 0 | 0.076197 | 0.004263 | 77 | 5898240 | 4166257152 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.998080 | 0.333 | 1.000 | 0.953 | 0.219 | 12 | 0.066927 | 0.007239 | 77 | 5898240 | 4166257152 |
| long_context_512 | long_context | 512 | 16 | greedy | 0.998860 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | - | - | 527 | 38928384 | 4233438720 |
| long_context_1024 | long_context | 1024 | 16 | greedy | 0.999711 | 1.000 | 0.800 | 1.000 | 1.000 | 16 | - | - | 1039 | 76677120 | 4311493120 |
