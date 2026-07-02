# ThinTensor stress validation

- HF model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- HF reference: `eager_cached_decode`
- Cases: `3`

## bf16_exact

- Minimum cosine: `0.999425`
- Minimum top-5 overlap: `1.000`
- All checkpoint top-1 same: `False`
- Minimum teacher-forced token match: `0.953`
- Minimum free-running token match: `0.016`
- Maximum sampling total variation: `0.042918`
- Maximum sampling JS divergence: `0.007773`
- Maximum peak VRAM: `6285665280` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | max TV | max JS | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.999863 | 1.000 | 1.000 | 0.953 | 0.016 | 1 | 0.042918 | 0.007773 | 77 | 5898240 | 6285665280 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.999425 | 1.000 | 1.000 | 1.000 | 1.000 | 64 | 0.037770 | 0.000897 | 77 | 5898240 | 6285665280 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.999863 | 0.667 | 1.000 | 0.969 | 0.219 | 12 | 0.027880 | 0.003948 | 77 | 5898240 | 6285665280 |

## retained_fp8

- Minimum cosine: `0.997608`
- Minimum top-5 overlap: `0.800`
- All checkpoint top-1 same: `False`
- Minimum teacher-forced token match: `0.953`
- Minimum free-running token match: `0.047`
- Maximum sampling total variation: `0.091750`
- Maximum sampling JS divergence: `0.016176`
- Maximum peak VRAM: `3956803072` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | max TV | max JS | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.997608 | 1.000 | 1.000 | 0.953 | 0.484 | 31 | 0.064495 | 0.016176 | 77 | 5898240 | 3956803072 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.997775 | 1.000 | 0.800 | 0.969 | 0.047 | 0 | 0.091750 | 0.005965 | 77 | 5898240 | 3956803072 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.997809 | 0.667 | 1.000 | 0.969 | 0.219 | 12 | 0.050078 | 0.009592 | 77 | 5898240 | 3956803072 |
