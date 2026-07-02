# SmolLM3 Q8 speed baseline

- Prompt tokens: `128`
- Decode tokens: `200`
- Warmup tokens: `10`
- llama.cpp: llama.cpp baseline disabled by --skip-llamacpp
- Ollama: Ollama Q8_0 is a speed-only GGUF deployment baseline; logits were not extracted

| Mode | tok/s | ms/token | Peak allocated | Peak reserved | Resident weights | Saved weights |
|:---|---:|---:|---:|---:|---:|---:|
| ThinTensor optimized BF16 causal-KV decode | 32.691 | 30.589 | 6186137600 | 6194987008 | 6150197248 | 0 |
| ThinTensor selective scaled FP8 gate/up all + down layers 8:28; BF16 O projection and LM head | 47.403 | 21.096 | 6189679104 | 6239027200 | 4079448064 | 2070749184 |
| ThinTensor selective scaled FP8 gate/up all + down layers 10:26; BF16 O projection and LM head | 44.461 | 22.492 | 6189679104 | 6239027200 | 4169592832 | 1980604416 |
| Ollama/llama.cpp GGUF Q8_0 deployment baseline | 45.713 | 21.876 | - | - | 3519389696 | - |

HF bitsandbytes int8 is a Q8-ish baseline. llama.cpp Q8_0 is a separate GGUF deployment baseline.
