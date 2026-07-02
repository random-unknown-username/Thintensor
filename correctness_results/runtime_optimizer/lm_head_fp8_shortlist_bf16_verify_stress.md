# ThinTensor stress validation

- HF model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- HF reference: `eager_cached_decode`
- Cases: `13`

## quality_8_28_head8_topk_guard

- Minimum cosine: `0.991255`
- Minimum top-5 overlap: `0.800`
- All checkpoint top-1 same: `True`
- Minimum teacher-forced token match: `0.906`
- Minimum free-running token match: `0.000`
- Maximum sampling total variation: `0.086904`
- Maximum sampling JS divergence: `0.016176`
- Maximum peak VRAM: `4575183360` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | max TV | max JS | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hello | prompt | 1 | 128 | greedy | 0.995828 | 1.000 | 0.800 | 0.977 | 0.680 | 84 | - | - | 128 | 9437184 | 4436568576 |
| factual | prompt | 5 | 32 | greedy | 0.997681 | 1.000 | 1.000 | 1.000 | 1.000 | 32 | - | - | 36 | 3538944 | 4424677888 |
| reasoning | prompt | 13 | 32 | greedy | 0.998807 | 1.000 | 1.000 | 1.000 | 1.000 | 32 | - | - | 44 | 3538944 | 4424686080 |
| code | prompt | 12 | 32 | greedy | 0.995156 | 1.000 | 1.000 | 0.969 | 0.844 | 23 | - | - | 43 | 3538944 | 4424685056 |
| unicode | prompt | 14 | 32 | greedy | 0.999659 | 1.000 | 1.000 | 0.969 | 0.562 | 18 | - | - | 45 | 3538944 | 4424687104 |
| repetition | prompt | 25 | 32 | greedy | 0.999886 | 1.000 | 0.800 | 1.000 | 1.000 | 32 | - | - | 56 | 4718592 | 4427057664 |
| weird_control | prompt | 21 | 32 | greedy | 0.998415 | 1.000 | 0.800 | 0.906 | 0.031 | 1 | - | - | 52 | 4718592 | 4427053568 |
| multi_turn_chat | chat_template | 83 | 64 | greedy | 0.991255 | 1.000 | 0.800 | 0.969 | 0.031 | 1 | - | - | 146 | 11796480 | 4441305600 |
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.998112 | 1.000 | 1.000 | 0.953 | 0.578 | 34 | 0.064495 | 0.016176 | 77 | 5898240 | 4429438464 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.998301 | 1.000 | 0.800 | 0.969 | 0.000 | 0 | 0.086904 | 0.005755 | 77 | 5898240 | 4429438464 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.998634 | 1.000 | 1.000 | 0.953 | 0.234 | 12 | 0.039781 | 0.005964 | 77 | 5898240 | 4429438464 |
| long_context_512 | long_context | 512 | 16 | greedy | 0.998749 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | - | - | 527 | 38928384 | 4496620032 |
| long_context_1024 | long_context | 1024 | 16 | greedy | 0.999861 | 1.000 | 0.800 | 1.000 | 1.000 | 16 | - | - | 1039 | 76677120 | 4575183360 |
