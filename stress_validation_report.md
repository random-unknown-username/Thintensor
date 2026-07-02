# ThinTensor stress validation

- HF model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- HF reference: `eager_cached_decode`
- Cases: `13`

## bf16_exact

- Minimum cosine: `0.997886`
- Minimum top-5 overlap: `0.800`
- All checkpoint top-1 same: `False`
- Minimum teacher-forced token match: `0.953`
- Minimum free-running token match: `0.016`
- Maximum peak VRAM: `6430984704` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hello | prompt | 1 | 128 | greedy | 0.997886 | 1.000 | 1.000 | 1.000 | 1.000 | 128 | 128 | 9437184 | 6292795392 |
| factual | prompt | 5 | 32 | greedy | 0.999775 | 1.000 | 1.000 | 1.000 | 1.000 | 32 | 36 | 3538944 | 6280904704 |
| reasoning | prompt | 13 | 32 | greedy | 0.999917 | 1.000 | 1.000 | 0.969 | 0.344 | 11 | 44 | 3538944 | 6280912896 |
| code | prompt | 12 | 32 | greedy | 0.999876 | 1.000 | 1.000 | 0.969 | 0.844 | 23 | 43 | 3538944 | 6280911872 |
| unicode | prompt | 14 | 32 | greedy | 0.999954 | 1.000 | 1.000 | 0.969 | 0.562 | 18 | 45 | 3538944 | 6280913920 |
| repetition | prompt | 25 | 32 | greedy | 0.999941 | 1.000 | 1.000 | 1.000 | 1.000 | 32 | 56 | 4718592 | 6283284480 |
| weird_control | prompt | 21 | 32 | greedy | 0.999863 | 1.000 | 0.800 | 0.969 | 0.281 | 9 | 52 | 4718592 | 6283280384 |
| multi_turn_chat | chat_template | 83 | 64 | greedy | 0.999308 | 1.000 | 1.000 | 0.984 | 0.734 | 47 | 146 | 11796480 | 6297532416 |
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.999863 | 1.000 | 1.000 | 0.953 | 0.016 | 1 | 77 | 5898240 | 6285665280 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.999425 | 1.000 | 1.000 | 1.000 | 1.000 | 64 | 77 | 5898240 | 6285665280 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.999863 | 0.667 | 1.000 | 0.969 | 0.219 | 12 | 77 | 5898240 | 6285665280 |
| long_context_512 | long_context | 512 | 16 | greedy | 0.999693 | 1.000 | 0.800 | 1.000 | 1.000 | 16 | 527 | 38928384 | 6352930304 |
| long_context_1024 | long_context | 1024 | 16 | greedy | 0.999960 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 1039 | 76677120 | 6430984704 |

## retained_fp8

- Minimum cosine: `0.984710`
- Minimum top-5 overlap: `0.800`
- All checkpoint top-1 same: `False`
- Minimum teacher-forced token match: `0.938`
- Minimum free-running token match: `0.047`
- Maximum peak VRAM: `4103072256` bytes

| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | KV tokens | KV bytes | peak VRAM |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hello | prompt | 1 | 128 | greedy | 0.995673 | 1.000 | 0.800 | 0.984 | 0.680 | 84 | 128 | 9437184 | 3963933184 |
| factual | prompt | 5 | 32 | greedy | 0.998614 | 1.000 | 1.000 | 1.000 | 1.000 | 32 | 36 | 3538944 | 3952042496 |
| reasoning | prompt | 13 | 32 | greedy | 0.998803 | 1.000 | 0.800 | 1.000 | 1.000 | 32 | 44 | 3538944 | 3952050688 |
| code | prompt | 12 | 32 | greedy | 0.994525 | 1.000 | 1.000 | 0.969 | 0.844 | 23 | 43 | 3538944 | 3952049664 |
| unicode | prompt | 14 | 32 | greedy | 0.999543 | 1.000 | 1.000 | 0.969 | 0.562 | 18 | 45 | 3538944 | 3952051712 |
| repetition | prompt | 25 | 32 | greedy | 0.999838 | 1.000 | 0.800 | 1.000 | 1.000 | 32 | 56 | 4718592 | 3954422272 |
| weird_control | prompt | 21 | 32 | greedy | 0.998553 | 1.000 | 0.800 | 0.938 | 0.438 | 6 | 52 | 4718592 | 3954418176 |
| multi_turn_chat | chat_template | 83 | 64 | greedy | 0.984710 | 1.000 | 0.800 | 0.969 | 0.250 | 12 | 146 | 11796480 | 3968670208 |
| sampling_1 | sampling | 14 | 64 | temp=0.7,top_p=0.9,top_k=50 | 0.997608 | 1.000 | 1.000 | 0.953 | 0.484 | 31 | 77 | 5898240 | 3956803072 |
| sampling_2 | sampling | 14 | 64 | temp=1,top_p=1,top_k=0 | 0.997775 | 1.000 | 0.800 | 0.969 | 0.047 | 0 | 77 | 5898240 | 3956803072 |
| sampling_3 | sampling | 14 | 64 | temp=1.3,top_p=0.95,top_k=100 | 0.997809 | 0.667 | 1.000 | 0.969 | 0.219 | 12 | 77 | 5898240 | 3956803072 |
| long_context_512 | long_context | 512 | 16 | greedy | 0.998990 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 527 | 38928384 | 4023984640 |
| long_context_1024 | long_context | 1024 | 16 | greedy | 0.999738 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 1039 | 76677120 | 4103072256 |
