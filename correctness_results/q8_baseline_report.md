# SmolLM3 Q8 quality baseline

- HF model: `SmolLM3-3B`
- Archive: `SmolLM3-3B.thin`
- Reference: `HF BF16 eager cached decode`
- Stress cases: `13`

| Mode | Min cosine | Max cosine loss | Top1 all pass | Min top5 | Max JS distance | Max TV | Peak VRAM | Cases |
|:---|---:|---:|:---:|---:|---:|---:|---:|---:|
| HF BF16 | 1.000000 | 0.000000 | True | 1.000 | 0.000000 | 0.000000 | 6506062336 | 13 |
| ThinTensor BF16 | 0.997820 | 0.002180 | False | 0.800 | 0.041240 | 0.044123 | 6239639040 | 13 |
| ThinTensor quality FP8 | 0.993787 | 0.006213 | False | 0.800 | 0.154445 | 0.155155 | 4168889856 | 13 |
| HF bitsandbytes int8/Q8-ish | 0.393246 | 0.606754 | False | 0.600 | 0.203726 | 0.257416 | 3699070464 | 13 |

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

## ThinTensor BF16

- `hello`: cosine 0.998647, top1=True, top5=1.000, JS distance=0.000111, TV=0.000001
- `factual`: cosine 0.999868, top1=True, top5=1.000, JS distance=0.012424, TV=0.009894
- `reasoning`: cosine 0.999938, top1=True, top5=1.000, JS distance=0.028685, TV=0.031798
- `code`: cosine 0.999873, top1=True, top5=1.000, JS distance=0.036073, TV=0.038077
- `unicode`: cosine 0.999952, top1=True, top5=1.000, JS distance=0.021943, TV=0.029341
- `repetition`: cosine 0.999970, top1=True, top5=1.000, JS distance=0.018809, TV=0.018194
- `weird_control`: cosine 0.999889, top1=True, top5=0.800, JS distance=0.041240, TV=0.044123
- `multi_turn_chat`: cosine 0.997820, top1=True, top5=1.000, JS distance=0.018488, TV=0.009837
- `sampling_1`: cosine 0.999850, top1=True, top5=1.000, JS distance=0.021783, TV=0.030354
- `sampling_2`: cosine 0.999777, top1=True, top5=1.000, JS distance=0.024557, TV=0.032662
- `sampling_3`: cosine 0.999824, top1=False, top5=1.000, JS distance=0.024206, TV=0.023819
- `long_context_512`: cosine 0.999842, top1=True, top5=1.000, JS distance=0.005513, TV=0.001366
- `long_context_1024`: cosine 0.999948, top1=True, top5=1.000, JS distance=0.010105, TV=0.004371

## ThinTensor quality FP8

- `hello`: cosine 0.996249, top1=True, top5=0.800, JS distance=0.000717, TV=0.000009
- `factual`: cosine 0.998352, top1=True, top5=0.800, JS distance=0.022585, TV=0.014562
- `reasoning`: cosine 0.998986, top1=True, top5=0.800, JS distance=0.069059, TV=0.073833
- `code`: cosine 0.996153, top1=True, top5=1.000, JS distance=0.152999, TV=0.145121
- `unicode`: cosine 0.999694, top1=True, top5=0.800, JS distance=0.035197, TV=0.034229
- `repetition`: cosine 0.999883, top1=True, top5=0.800, JS distance=0.038641, TV=0.046764
- `weird_control`: cosine 0.998468, top1=True, top5=0.800, JS distance=0.154445, TV=0.155155
- `multi_turn_chat`: cosine 0.993787, top1=True, top5=0.800, JS distance=0.052821, TV=0.032343
- `sampling_1`: cosine 0.998217, top1=True, top5=1.000, JS distance=0.040471, TV=0.046625
- `sampling_2`: cosine 0.998410, top1=True, top5=0.800, JS distance=0.068376, TV=0.075521
- `sampling_3`: cosine 0.998280, top1=False, top5=0.800, JS distance=0.049363, TV=0.063557
- `long_context_512`: cosine 0.998790, top1=True, top5=1.000, JS distance=0.008989, TV=0.003753
- `long_context_1024`: cosine 0.999741, top1=True, top5=0.800, JS distance=0.020300, TV=0.013437

## HF bitsandbytes int8/Q8-ish

- `hello`: cosine 0.393246, top1=True, top5=0.600, JS distance=0.007581, TV=0.000170
- `factual`: cosine 0.975897, top1=True, top5=0.800, JS distance=0.063070, TV=0.035505
- `reasoning`: cosine 0.995715, top1=True, top5=0.800, JS distance=0.085580, TV=0.070751
- `code`: cosine 0.993369, top1=False, top5=0.800, JS distance=0.198838, TV=0.257416
- `unicode`: cosine 0.999184, top1=True, top5=0.800, JS distance=0.059380, TV=0.058375
- `repetition`: cosine 0.999368, top1=True, top5=0.800, JS distance=0.069638, TV=0.077089
- `weird_control`: cosine 0.997118, top1=True, top5=0.600, JS distance=0.203726, TV=0.232333
- `multi_turn_chat`: cosine 0.952472, top1=True, top5=0.800, JS distance=0.050497, TV=0.025710
- `sampling_1`: cosine 0.997955, top1=True, top5=1.000, JS distance=0.060611, TV=0.059372
- `sampling_2`: cosine 0.991564, top1=True, top5=0.600, JS distance=0.137296, TV=0.148911
- `sampling_3`: cosine 0.997955, top1=False, top5=1.000, JS distance=0.121394, TV=0.164224
- `long_context_512`: cosine 0.990131, top1=True, top5=0.800, JS distance=0.017588, TV=0.004072
- `long_context_1024`: cosine 0.999494, top1=True, top5=0.800, JS distance=0.041298, TV=0.032384
