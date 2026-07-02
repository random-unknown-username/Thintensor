# SmolLM3 Q8 speed baseline

- Prompt tokens: `128`
- Decode tokens: `200`
- Warmup tokens: `10`
- llama.cpp: llama.cpp baseline disabled by --skip-llamacpp
- Ollama: Ollama baseline disabled by --skip-ollama

| Mode | tok/s | ms/token | Peak allocated | Peak reserved | Resident weights | Saved weights |
|:---|---:|---:|---:|---:|---:|---:|
| ThinTensor selective scaled FP8 gate/up all + down layers 10:26; BF16 O projection and LM head | 46.380 | 21.561 | 6189679104 | 6239027200 | 4169592832 | 1980604416 |

HF bitsandbytes int8 is a Q8-ish baseline. llama.cpp Q8_0 is a separate GGUF deployment baseline.
