# Full-logit llama.cpp Q8_0 comparison

All modes use identical HF token IDs and HF BF16 teacher trajectories.

| Mode | Min centered-logit cosine | Min raw-logit cosine | Top1 all pass | Min top5 | Max JS distance | Max TV |
|:---|---:|---:|:---:|---:|---:|---:|
| hf_bf16 | 1.000000 | 1.000000 | True | 1.000 | 0.000000 | 0.000000 |
| thin_bf16 | 0.999277 | 0.997820 | False | 0.800 | 0.041239 | 0.044124 |
| thin_quality_8_28 | 0.992888 | 0.993787 | False | 0.800 | 0.154444 | 0.155157 |
| thin_quality_10_26 | 0.992646 | 0.993413 | False | 0.800 | 0.166408 | 0.173962 |
| llamacpp_q8_0 | 0.952666 | - | True | 0.800 | 0.059670 | 0.066013 |

llama.cpp returns complete log-probabilities, not raw logits. Centered-logit cosine removes the unknown additive log-softmax constant and is therefore the direct cross-runtime cosine.
