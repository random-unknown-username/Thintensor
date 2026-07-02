You are Codex working inside my ThinTensor repo.

Goal:
Build ThinTensor into a serious runtime-first model format and benchmark system that can ingest HuggingFace safetensors, run Qwen3 0.6B or the smallest available Qwen/Llama-style model, measure baseline tokens/sec, convert it into ThinTensor, run equivalent benchmark paths, then repeatedly optimize until there are no obvious improvements left.

Do not stop after one feature. Keep working in a loop. Each loop must:

1. Inspect the repo state.
2. Identify the highest-impact missing piece.
3. Implement it cleanly.
4. Run formatting, clippy, tests, and real commands.
5. Benchmark before/after where possible.
6. Record results in BENCHMARKS.md and ROADMAP.md.
7. Commit or at least leave a clear changelog-style summary.
8. Continue to the next improvement.

Current project state:

* ThinTensor archive format exists.
* CLI has pack, inspect, verify, extract, plan.
* Archive layout uses THINv0\0\0 fixed header, manifest bytes, page table, raw page blobs.
* Manifest validation and strict verifier exist.
* Planner exists with --backend, --vram, --ctx.
* Fake llama fixture exists.
* No safetensors/zstd/mmap dependency currently remains unless needed.

Main mission:
Make ThinTensor prove a real claim:

"ThinTensor can convert HuggingFace safetensors into a verified execution-ordered archive, benchmark baseline model loading/inference paths, and progressively improve memory efficiency and runtime behavior."

Phase 1: HF converter
Implement:

thintensor convert-hf <hf_dir> <out.thin> [--arch auto|llama|qwen|generic] [--no-tokenizer]

Requirements:

* Read hf_dir/config.json.
* Read one or more *.safetensors files.
* Support model.safetensors.index.json if present, but do not require it.
* Preserve original HF tensor names as page IDs.
* Copy raw safetensors tensor bytes exactly as page blobs.
* Do not deserialize tensors into floats.
* Set backend_layout = "hf_raw".
* Build execution tape from Llama/Qwen tensor names.
* Hard error on missing required layer tensors.
* Warn on missing lm_head/tokenizer/unknown extras.
* Run existing archive verification after writing.
* Add tests and tiny generated HF fixture if possible.

Supported tensor names:

* model.embed_tokens.weight
* model.layers.N.input_layernorm.weight
* model.layers.N.self_attn.q_proj.weight
* model.layers.N.self_attn.k_proj.weight
* model.layers.N.self_attn.v_proj.weight
* model.layers.N.self_attn.o_proj.weight
* model.layers.N.self_attn.q_norm.weight optional
* model.layers.N.self_attn.k_norm.weight optional
* model.layers.N.post_attention_layernorm.weight
* model.layers.N.mlp.gate_proj.weight
* model.layers.N.mlp.up_proj.weight
* model.layers.N.mlp.down_proj.weight
* model.norm.weight
* lm_head.weight optional

Phase 2: baseline benchmark
Create a benchmark system.

Add commands or scripts:

thintensor bench-baseline <hf_dir> --prompt "..." --tokens 128 --ctx 2048
thintensor bench-archive <thin_file> --tokens 128 --ctx 2048
thintensor bench-load <hf_dir|thin_file>
thintensor bench-plan <thin_file> --backend cuda --vram 4GB --ctx 2048

If full inference is too much inside Rust, create scripts/bench_baseline.py using Python transformers or llama.cpp if available. Be pragmatic.

Benchmark Qwen3 0.6B if available locally or downloadable. If not, benchmark the smallest available Qwen/Llama-style HF safetensors model and clearly record the fallback.

Measure:

* model load time
* archive conversion time
* archive verify time
* peak RSS if possible
* tokens/sec for baseline path if possible
* estimated VRAM from ThinTensor plan
* archive size
* page count
* largest pages
* execution-stage count

Write results to BENCHMARKS.md in a table.

Phase 3: ThinTensor introspection
Improve:

thintensor inspect

Add:

* top largest pages
* per-layer byte totals
* per-op byte totals
* dtype distribution
* backend_layout distribution
* execution tape summary
* unknown/extra tensors list
* model metadata summary

Add:

thintensor stats <thin_file>

Output machine-readable JSON too:

thintensor stats <thin_file> --json

Phase 4: memory-efficiency planning
Improve planner so it becomes a real VRAM budget compiler.

Add:

* --ctx
* --batch
* --kv-dtype fp16|bf16|fp8|q8|q4|q3|q2
* --weight-residency all|stream|offload-last-n|offload-first-n
* --gpu-fraction
* --json

Planner should estimate:

* weight bytes
* scratch bytes
* KV cache bytes
* metadata overhead
* total bytes
* whether it fits
* suggestions if not:

  * reduce ctx
  * reduce batch
  * use KV q4/q3/q2
  * stream layers
  * offload layers
  * use lower-bit profile

Phase 5: execution-ordered repack
Add:

thintensor repack <input.thin> <output.thin> --layout execution_ordered_v1

This should not change tensor bytes yet. It should reorder page blobs in archive storage order according to execution tape, so sequential reading follows runtime order.

Benchmark:

* verify still passes
* extract equality still passes
* inspect shows new layout
* compare scan/read time before and after

Phase 6: real page profiles
Add profile generation:

thintensor profile <thin_file> --target-vram 4GB
thintensor profile <thin_file> --target-vram 6GB
thintensor profile <thin_file> --target-vram 8GB

Profiles should mark:

* always_hot pages
* streamable pages
* CPU-offload candidates
* KV codec recommendation
* per-layer byte cost
* biggest memory offenders

Store in manifest if schema supports it, or emit sidecar JSON:

model.thin.profile-4gb.json

Phase 7: optional compression
Only after the above works, add optional page compression:

thintensor repack <input.thin> <output.thin> --compress zstd

Rules:

* Do not break raw v0 archives.
* Add flags in page table cleanly.
* Verifier must hash decompressed raw bytes or clearly define checksum semantics.
* Extract must restore original raw page bytes.
* Bench compression ratio and verify time.

Phase 8: optional mmap/load simulator
Add a load simulator:

thintensor simulate-load <thin_file> --backend cuda --vram 4GB --ctx 4096

It should simulate:

* page load order
* resident pages
* evicted pages
* peak memory
* sequential vs random reads
* bytes moved per generated token estimate

This is important. ThinTensor’s revolutionary claim should be "fewer bytes moved per token", so make the tool estimate that.

Phase 9: optimization loop
Repeat forever or until truly blocked.

At the start of every loop, choose one of:

* correctness improvement
* benchmark improvement
* planner improvement
* converter improvement
* archive layout improvement
* memory efficiency improvement
* documentation/demo improvement

Never do cosmetic-only work unless docs are blocking usability.

Every loop must end with:

* cargo fmt --check
* cargo clippy --all-targets --all-features -- -D warnings
* cargo test
* at least one real CLI command
* update BENCHMARKS.md or ROADMAP.md

Do not fake benchmark results.
Do not claim speedups unless measured.
Do not remove strict verification.
Do not weaken checksum/offset/overlap validation.
Do not introduce unsafe Rust unless there is a measurable reason and it is isolated/commented.
Do not make the format dependent on Python.
Python scripts are allowed only for benchmarking/conversion helpers if Rust implementation is not practical yet.

If something fails:

* Debug it.
* Fix it.
* Re-run.
* If still blocked, write BLOCKERS.md with exact error/output and move to the next useful task.

Target demo:
By the end, I want this flow to work:

cargo build --release
target/release/thintensor convert-hf ./models/Qwen3-0.6B ./target/qwen3-0.6b.thin
target/release/thintensor verify ./target/qwen3-0.6b.thin
target/release/thintensor inspect ./target/qwen3-0.6b.thin
target/release/thintensor stats ./target/qwen3-0.6b.thin --json
target/release/thintensor plan ./target/qwen3-0.6b.thin --backend cuda --vram 4GB --ctx 4096 --kv-dtype q4
target/release/thintensor repack ./target/qwen3-0.6b.thin ./target/qwen3-0.6b.exec.thin --layout execution_ordered_v1
target/release/thintensor simulate-load ./target/qwen3-0.6b.exec.thin --backend cuda --vram 4GB --ctx 4096
python scripts/bench_baseline.py ./models/Qwen3-0.6B --tokens 128
target/release/thintensor bench-load ./target/qwen3-0.6b.exec.thin

Work continuously. Prefer measurable, compounding improvements over big rewrites. Keep the code clean. Keep ThinTensor strict. Keep going.
