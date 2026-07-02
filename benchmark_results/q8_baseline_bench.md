# SmolLM3 Q8 speed baseline

- Prompt tokens: `128`
- Decode tokens: `200`
- Warmup tokens: `10`
- llama.cpp: llama.cpp GGUF baseline not run: missing model or binary
- Ollama: Ollama Q8_0 is a speed-only GGUF deployment baseline; logits were not extracted

| Mode | tok/s | ms/token | Peak allocated | Peak reserved | Resident weights | Saved weights |
|:---|---:|---:|---:|---:|---:|---:|
| HF Transformers BF16 cached greedy decode | 17.797 | 56.190 | 6201513984 | 6239027200 | 6150197248 | - |
| HF Transformers bitsandbytes int8/Q8-ish cached greedy decode | 15.068 | 66.368 | 3421783040 | 3531603968 | 3337916416 | - |
| ThinTensor optimized BF16 causal-KV decode | 32.256 | 31.002 | 6186137600 | 6194987008 | 6150197248 | 0 |
| ThinTensor selective scaled FP8 gate/up all + down layers 8:28; BF16 O projection and LM head | 45.936 | 21.770 | 6189679104 | 6239027200 | 4079448064 | 2070749184 |
| Ollama/llama.cpp GGUF Q8_0 deployment baseline | 44.133 | 22.659 | - | - | 3519389696 | - |

HF bitsandbytes int8 is a Q8-ish baseline. llama.cpp Q8_0 is a separate GGUF deployment baseline.
