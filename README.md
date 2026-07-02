# ThinTensor

ThinTensor v0 is a `.thin` archive, schema compiler, CUDA decode runtime, and
runtime budget planner.

The manifest says what the model means. The page table says where bytes live.
Verification compares both and then checks the blobs.

## Quick Run

```bash
cargo run -- pack examples/fake-llama.json examples/pages target/fake.thin
cargo run -- inspect target/fake.thin
cargo run -- stats target/fake.thin --json
cargo run -- verify target/fake.thin
cargo run -- plan target/fake.thin --backend cuda --vram 8GB --ctx 8192 --kv-dtype q4
cargo run -- extract target/fake.thin examples/out
cargo run -- convert-hf ./Qwen3-0.6B ./qwen3-0.6b.thin
python3 scripts/inspect_hf_compat.py --hf-model ./Qwen3-0.6B --json
python3 scripts/auto_optimize.py --archive ./qwen3-0.6b.thin --hf-model ./Qwen3-0.6B --modes auto
python3 scripts/first_run_pipeline.py --hf-dir ./Qwen3-0.6B --device cuda --trials 2
```

## CLI

```text
thintensor pack manifest.json pages_dir/ out.thin
thintensor inspect out.thin
thintensor stats out.thin --json
thintensor verify out.thin
thintensor extract out.thin out_dir/
thintensor plan out.thin --backend cuda --vram 8GB --ctx 8192 --batch 1 --kv-dtype q4
thintensor plan out.thin --backend cuda --vram 1GB --ctx 8192 --weight-residency stream
thintensor profile out.thin --target-vram 4GB --backend cuda --ctx 8192
thintensor simulate-load out.thin --backend cuda --vram 4GB --ctx 8192 --weight-residency stream --prefetch-pages 4
thintensor convert-hf ./Qwen-or-Llama-HF ./out.thin
thintensor convert-hf ./Qwen-or-Llama-HF ./out.thin --arch llama
thintensor convert-hf ./Qwen-or-Llama-HF ./out.thin --no-tokenizer
thintensor bench-baseline ./Qwen-or-Llama-HF --tokens 128 --device cuda
thintensor bench-archive ./out.thin --hf-dir ./Qwen-or-Llama-HF --tokens 128 --device cuda
thintensor bench-plan ./out.thin --backend cuda --vram 4GB --ctx 8192 --json
thintensor repack ./out.thin ./out.exec.thin --layout execution_ordered_v1
thintensor repack ./out.thin ./out.hot.thin --layout hot_stream_v1
thintensor repack ./out.thin ./out.fused.thin --layout fused_decode_v1
python3 scripts/thin_runtime.py run ./out.thin --residency all --kernel-backend triton-matvec --steps 500 --warmup-steps 10 --lm-head-fp8 --json
```

`pack` does not repair metadata. Every manifest page must already declare the
exact file size and BLAKE3 checksum for `pages_dir/<PAGE_ID>.bin`.

`convert-hf` reads `config.json`, optional tokenizer metadata, and all
`*.safetensors` shards. It discovers common decoder schemas from tensor roles,
preserves quantization metadata, and writes physical pages in execution order.
Dense separate/fused projections and packed sparse-MoE expert tensors are
represented explicitly in the execution tape.

`bench-baseline` and `bench-archive` run actual Transformers generation. The
ThinTensor path hydrates model weights from `.thin` pages, ties shared weights,
uses a thermal stop guard, and then runs the same generation call.

`stats` verifies the archive and emits the model anatomy: largest pages,
per-layer totals, per-op totals, dtype/layout distributions, execution tape
coverage, and unreferenced pages.

`plan` is the v0 VRAM budget compiler. It supports `--ctx`, `--batch`,
`--kv-dtype fp16|bf16|fp8|q8|q4|q3|q2`, `--weight-residency
all|stream|offload-last-n|offload-first-n`, `--offload-layers`,
`--gpu-fraction`, and `--json`. It accounts for scratch, KV cache, metadata,
resident weights, streamed/offloaded weights, and exact duplicate shared
weights such as tied embeddings.

`profile` writes a sidecar JSON profile for a target VRAM budget. It evaluates
all-resident, offload, and streamed candidates, then records always-hot pages,
streamable pages, CPU-offload candidates, per-layer costs, biggest memory
offenders, the KV codec recommendation, and the recommended repack layout.

`simulate-load` estimates the execution load order for a plan. It reports
resident and streamed pages, evicted pages, sequential versus random page-table
movement, stream read groups, prefetch IO ops, recommended staging memory, peak
memory, and estimated bytes moved per generated token. This is the main v0 tool
for proving whether a repack/profile improves runtime locality.

`repack --layout hot_stream_v1` is a streaming-oriented layout: always-hot
global pages are placed first, then layer pages follow execution order. It is
useful when profiles choose streaming/offload because hot pages can be loaded in
one contiguous startup group while streamed layer pages remain sequential.

`repack --layout fused_decode_v1` stores each layer's Q/K/V and gate/up matrices
once in stable row-concatenated physical pages. The original tensor IDs remain
logical aliases, so normal tensor loading still works without runtime
`torch.cat` or duplicated GPU weights.

The native CUDA runtime keeps BF16 as the default. `--lm-head-fp8` is an
explicit speed/memory tradeoff: it keeps embeddings, layers, activations, and KV
in BF16, but creates an FP8 LM-head copy. This cuts estimated per-token weight
reads by about 156 MB for Qwen3 0.6B and adds about 156 MB of resident VRAM.
Always compare fixed-token logits before using it for a model.

Packed MXFP4 expert pages execute directly through a selected-expert Triton
kernel; they are not expanded to full BF16 matrices. Integer Q8/Q4/Q2,
FP4/MXFP4, and FP8 are distinct formats. ThinTensor does not silently lower or
cross-convert them. See `quantization_support.md`.

Model labels do not select hardcoded runtime branches. The schema compiler
recognizes common causal-decoder traits including separate/fused QKV,
separate/fused gated MLPs, packed top-k MoE, MHA/GQA/MQA, per-layer sliding
attention, attention sinks, biases, and standard RoPE variants. See
`model_schema_support.md` for validated and planned coverage.

`scripts/first_run_pipeline.py` is the first-timer path: build, download or
reuse HF files, convert, verify, collect stats, benchmark HF versus ThinTensor,
compare, optionally repack, and append results to `BENCHMARKS.md`. Use
`--trials N` to run subprocess-level repeated trials and record median/min/max
tokens/sec.

## Structure

- `src/main.rs`: CLI dispatch
- `src/manifest.rs`: serde structs and manifest validation
- `src/archive.rs`: fixed header, page table, raw blob read/write
- `src/convert_hf.rs`: HF safetensors converter
- `src/verify.rs`: strict archive verifier
- `src/plan.rs`: backend/VRAM/context planner
- `src/profile.rs`: target-VRAM residency profile generator
- `src/repack.rs`: raw-byte preserving page-layout repacker
- `src/simulate.rs`: load-order and bytes-moved simulator
- `src/stats.rs`: archive introspection and optimization summaries
- `src/units.rs`: `4GB`, `8GiB`, byte parsing and formatting
- `src/error.rs`: shared report/error types
- `docs/FORMAT.md`: binary layout and invariants
- `scripts/bench_runtime.py`: HF vs ThinTensor runtime generation benchmark
- `scripts/first_run_pipeline.py`: download/convert/verify/bench/compare pipeline
- `scripts/inspect_hf_compat.py`: schema/quantization/capability compiler report
- `scripts/auto_optimize.py`: bounded descriptor-driven real-decode optimizer
