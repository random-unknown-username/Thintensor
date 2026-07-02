# Q8 vs ThinTensor summary

Decision: **The measured modes have mixed results; see the full table.**

| Mode | Speed tok/s | Min cosine vs HF BF16 | Max cosine loss | Top1 all pass | Min top5 | Peak VRAM | Resident weight saved | Verdict |
|:---|---:|---:|---:|:---:|---:|---:|---:|:---|
| HF BF16 | 17.797 | 1.000000 | 0.000000 | True | 1.000 | 6201513984 | - | reference |
| HF bitsandbytes int8/Q8-ish | 15.068 | 0.393246 | 0.606754 | False | 0.600 | 3421783040 | - | measured |
| ThinTensor BF16 | 32.256 | 0.997820 | 0.002180 | False | 0.800 | 6186137600 | 0 | measured |
| ThinTensor quality FP8 | 45.936 | 0.993787 | 0.006213 | False | 0.800 | 6189679104 | 2070749184 | measured |
| ThinTensor quality FP8 10:26 | 46.380 | 0.993908 | 0.006092 | False | 0.800 | 6189679104 | 1980604416 | measured |
| ThinTensor quality FP8 + Head8 experimental | 50.438 | 0.996907 | 0.003093 | False | 1.000 | 6567678464 | - | experimental: long-case top1 changed |
| Ollama/llama.cpp Q8_0 | 44.133 | - | - | - | - | - | - | speed-only; full logits unavailable |

HF bitsandbytes int8 is used as a Q8-ish quality baseline; it is not llama.cpp Q8_0.

Ollama Q8_0 is reported separately as a GGUF deployment-speed baseline. Ollama exposes top log-probability shortlists but not the complete raw logit vector, so a full-vector cosine is not claimed.

ThinTensor quality FP8 is selective quantization, not BF16-equivalent.

Ollama note: Ollama Q8_0 is a speed-only GGUF deployment baseline; logits were not extracted.
