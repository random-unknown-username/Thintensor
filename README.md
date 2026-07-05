# ThinTensor

`thintensor` is one command for pulling, converting, running, chatting with,
benchmarking, and validating causal language models.

The CLI routes by model capability rather than repository name:

- compatible RMSNorm + gated-SiLU decoder models use the native ThinTensor
  runtime;
- other Transformers causal-LM architectures remain runnable through a
  clearly labelled compatibility engine;
- compatibility-engine measurements are never reported as ThinTensor speedups.

## Install

From a platform wheel:

```bash
python -m pip install 'thintensor[all]'
thintensor doctor --strict
```

From a Git clone:

```bash
git clone https://github.com/random-unknown-username/Thintensor
cd Thintensor
python -m pip install -e '.[all]'
thintensor --help
```

The repository launcher also works directly:

```bash
./thintensor --help
```

Platform wheels bundle the internal Rust archive core. Users invoke
`thintensor`; `thintensor-core` is an implementation detail.

## Run any Transformers causal LM

```bash
thintensor run Qwen/Qwen3-0.6B --prompt "Explain paged KV caches."
thintensor run meta-llama/Llama-3.2-1B --profile balanced
thintensor run ./local-model --engine transformers
```

`--engine auto` is the default. It downloads the model once, inspects its
semantics and tensor layout, converts native-compatible models to `.thin`, and
uses the Transformers compatibility engine otherwise. Remote custom code is
disabled unless `--trust-remote-code` is explicitly supplied.

## Convert

```bash
thintensor pull Qwen/Qwen3-0.6B
thintensor convert ~/.cache/thintensor/models/Qwen--Qwen3-0.6B \
  --out Qwen3-0.6B.thin
thintensor inspect Qwen3-0.6B.thin --verify
```

Conversion is deterministic and followed by archive verification unless
`--no-verify` is explicitly selected.

If an archive uses semantics outside the native engine, retain its original HF
config and run:

```bash
thintensor run MODEL.thin --hf-source ./original-model
```

## Profiles

Profiles are intent-based and shared by run, chat, bench, and validate.

```bash
thintensor profiles list
thintensor explain
thintensor explain max-performance --model ./local-model
thintensor architectures list
thintensor architectures audit ./local-model
```

- `auto`: stays on non-quantized balanced kernels for compatible dense
  decoders and safe BF16 for unsupported optimization capabilities.
- `safe`: BF16 weights, full BF16 KV history, broadest compatibility.
- `balanced`: BF16 weights with native ThinTensor matvec and causal-attention
  kernels. No approximate weight storage.
- `max-performance`: opt-in adaptive INT8/tensor-core profile. Quality and speed must be
  validated for every model and GPU.
- `lab`: explicit experimental overrides without quality or speed claims.

Legacy names such as `bf16`, `quality`, `fast-90`, and `experimental` remain
accepted as aliases.

Profile names are not universal throughput promises. `thintensor explain max-performance` includes
the hardware, context length, cosine, top-1, top-5, and KV-retention scope of
retained measurements.

Architecture status is stricter than load compatibility:

- `verified`: native correctness is recorded and native decode beat the matched
  Transformers loop.
- `candidate`: native semantic/tensor support exists, but correctness and speed
  gates are incomplete.
- `fallback`: the model runs through Transformers while its missing native
  operators are implemented.

Current matched verified results on the development RTX 5050 Laptop GPU:

- SmolLM3-3B: 94.00 tok/s versus Transformers 47.54 tok/s at 500 tokens.
- Phi-4-mini-instruct: 53.80 tok/s versus Transformers 39.09 tok/s at
  200 tokens, with fused QKV and fused gate/up native dispatch.
- StableLM-3B-4E1T: 55.76 tok/s versus Transformers 48.28 tok/s at
  200 tokens.

## Benchmark and validate

```bash
thintensor bench MODEL.thin \
  --profiles safe,balanced,max-performance \
  --warmup 10 --steps 500 \
  --hf-model ./original-model \
  --require-faster-than-hf \
  --out benchmark_results/cli.json

thintensor validate MODEL.thin \
  --hf-model ./original-model \
  --profile max-performance \
  --suite required \
  --require-tier ranking
```

Benchmarks use fresh processes and full causal KV history. Runs shorter than 10
warmup plus 200 measured tokens are marked smoke-only. Launch count and isolated
microkernel time are diagnostics, not end-to-end wins.

## Explain profiles and routing

The CLI explains why auto selected the native or compatibility engine, what
each profile changes, its quality and retention contract, retained measurements,
compatibility blockers, and the equivalent command:

```bash
thintensor explain max-performance --model ./local-model --json
```

## Residency

Full VRAM weight and KV residency are the defaults. Explicit bounded streaming:

```bash
thintensor run MODEL.thin \
  --residency stream \
  --gpu-weight-budget 6GiB \
  --cpu-offload --pin-cpu-pages
```

Exact KV modes are `gpu_full`, `hybrid_recent`, and `cpu_exact`. CPU-backed
modes preserve values but are memory-pressure options, not claimed speedups.

## Operations

```bash
thintensor cache list
thintensor cache path
thintensor doctor --json
thintensor inspect MODEL.thin --verify --json
thintensor core --help
```
