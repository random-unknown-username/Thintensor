# SmolLM3 Q8 quality baseline

- HF model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- Reference: `HF BF16 eager cached decode`
- Stress cases: `13`

| Mode | Min cosine | Max cosine loss | Top1 all pass | Min top5 | Max JS distance | Max TV | Peak VRAM | Cases |
|:---|---:|---:|:---:|---:|---:|---:|---:|---:|
| HF BF16 | 1.000000 | 0.000000 | True | 1.000 | 0.000000 | 0.000000 | 6506062336 | 13 |
| ThinTensor quality FP8 10:26 | 0.993908 | 0.006092 | False | 0.800 | 0.148174 | 0.146480 | 4259034624 | 13 |

HF bitsandbytes int8 is a Q8-ish quality baseline. It is not llama.cpp GGUF Q8_0.

## HF BF16

- `hello`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `factual`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `reasoning`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `code`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `unicode`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `repetition`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `weird_control`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `multi_turn_chat`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `sampling_1`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `sampling_2`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `sampling_3`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `long_context_512`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000
- `long_context_1024`: cosine 1.000000, top1=True, top5=1.000, JS distance=0.000000, TV=0.000000

## ThinTensor quality FP8 10:26

- `hello`: cosine 0.996189, top1=True, top5=0.800, JS distance=0.000642, TV=0.000008
- `factual`: cosine 0.998911, top1=True, top5=1.000, JS distance=0.020835, TV=0.025822
- `reasoning`: cosine 0.998881, top1=True, top5=1.000, JS distance=0.071804, TV=0.076626
- `code`: cosine 0.996444, top1=True, top5=1.000, JS distance=0.129197, TV=0.124905
- `unicode`: cosine 0.999649, top1=True, top5=0.800, JS distance=0.037525, TV=0.041664
- `repetition`: cosine 0.999885, top1=True, top5=0.800, JS distance=0.035904, TV=0.037354
- `weird_control`: cosine 0.998669, top1=True, top5=0.800, JS distance=0.148174, TV=0.146480
- `multi_turn_chat`: cosine 0.993908, top1=True, top5=0.800, JS distance=0.054005, TV=0.033640
- `sampling_1`: cosine 0.997868, top1=True, top5=1.000, JS distance=0.043812, TV=0.051947
- `sampling_2`: cosine 0.998313, top1=True, top5=0.800, JS distance=0.076701, TV=0.085654
- `sampling_3`: cosine 0.999161, top1=False, top5=0.800, JS distance=0.049028, TV=0.063499
- `long_context_512`: cosine 0.999025, top1=True, top5=1.000, JS distance=0.008947, TV=0.003646
- `long_context_1024`: cosine 0.999829, top1=True, top5=0.800, JS distance=0.019298, TV=0.012087
