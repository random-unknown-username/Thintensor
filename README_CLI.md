# ThinTensor CLI

`thin` is the supported command surface for archive conversion and inspection,
greedy causal inference, real decode benchmarking, correctness validation, and
one-candidate-at-a-time optimization.

## Install

```bash
cargo build --release
python -m pip install -e '.[all]'
thin doctor --strict
```

For archive-only commands, `python -m pip install -e .` is sufficient. The
repository wrapper `scripts/thin` also works without installation.

## Start here

```bash
thin pull HuggingFaceTB/SmolLM3-3B
thin convert ~/.cache/thintensor/models/HuggingFaceTB--SmolLM3-3B \
  --out SmolLM3-3B.thin
thin inspect SmolLM3-3B.thin --verify
thin run SmolLM3-3B.thin --prompt "Explain paged KV caches."
```

Text generation is currently greedy. Non-greedy sampling flags are rejected
instead of being ignored. If tokenizer files are not beside the archive, pass
`--tokenizer MODEL_OR_DIRECTORY`.

## Profiles

```bash
thin profiles list
thin profiles show quality-guarded
```

- `auto` selects `quality-guarded` only for the validated SmolLM3-3B geometry,
  and otherwise selects universal BF16.
- `bf16` keeps body weights, exact-value KV storage, attention, and LM head in
  BF16. This describes storage precision, not bitwise equivalence to every HF
  execution backend.
- `quality` uses the validated SmolLM3-3B gate/up FP8 and middle-layer
  down-projection FP8 profile while retaining BF16 attention, O-projection, KV,
  and LM head.
- `quality-guarded` adds the FP8 shortlist head with exact BF16 top-64
  verification.
- `experimental` starts from BF16 and permits explicit overrides. It does not
  silently enable rejected optimizations.

Model-specific profiles refuse incompatible archive geometry unless
`--force-profile` is supplied.

## Real decode benchmark

```bash
thin bench SmolLM3-3B.thin \
  --profiles bf16,quality,quality-guarded \
  --warmup 10 --steps 200 \
  --out benchmark_results/cli.json
```

Each profile runs in a fresh process through `scripts/thin_runtime.py` with
`attention_mode=causal_kv`. Results include end-to-end decode latency,
throughput, estimated bytes moved per token, bandwidth, and memory telemetry.
This command does not treat launch count or isolated microkernel speed as an
end-to-end win.

Runs shorter than 10 warmup plus 200 measured steps are marked smoke-only. Use
`--dry-run` to inspect the exact commands without loading the model.

## Correctness

```bash
thin validate SmolLM3-3B.thin \
  --hf-model ./SmolLM3-3B \
  --profile quality-guarded \
  --suite quick

thin validate SmolLM3-3B.thin \
  --hf-model ./SmolLM3-3B \
  --profile quality-guarded \
  --suite required --require-tier ranking --json
```

Validation compares identical HF and ThinTensor causal trajectories. `quick`
uses the short development gate; `required` covers prefill lengths
`1,8,32,128` and decode checkpoints `1,10,50`. Use `--require-tier exact` when
HF-equivalent cosine, rather than ranking equivalence, is mandatory.

## Correctness-gated optimization

```bash
thin optimize --init my-plan.json \
  --archive SmolLM3-3B.thin \
  --hf-model ./SmolLM3-3B
thin optimize runtime_optimizer_plan.json --resume
```

This wraps `scripts/optimize_runtime_one_by_one.py`: every candidate is
benchmarked in isolation and stacked only after its configured correctness and
performance gates pass. The generated plan is ordinary JSON; edit its
hypotheses and candidate flags before starting a new investigation.

## Residency

Full VRAM residency is the default:

```bash
thin run MODEL.thin --residency all
```

Bounded streaming requires an explicit budget:

```bash
thin run MODEL.thin \
  --residency stream \
  --gpu-weight-budget 6GiB \
  --prefetch-layers 0 \
  --cpu-offload --pin-cpu-pages
```

Exact KV modes are `gpu_full`, `hybrid_recent`, and `cpu_exact`. CPU-backed
modes are memory-pressure options, not claimed speed optimizations.

## Operations

```bash
thin cache list
thin cache path
thin doctor --json
thin inspect MODEL.thin --verify --json
thin core --help
thin tui
```

`thin core` passes arguments to the Rust archive CLI for low-level commands not
yet promoted into the unified interface.
