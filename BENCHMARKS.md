# ThinTensor Benchmarks

## Final current-source SmolLM3 checks (2026-07-02)

Standalone 500-token runs, all with 10 warmup tokens and causal KV:

| Path | Steady tok/s | Resident weights | Read bytes/token | Peak allocated |
|:---|---:|---:|---:|---:|
| BF16 | 34.310 | 6,150,197,248 | 6,149,898,240 | 6,313,040,896 |
| BF16 + guarded head | 35.206 | 6,413,378,560 | 5,888,005,120 | 6,576,220,672 |
| Quality FP8 | 48.625 | 4,076,113,920 | 4,075,814,912 | 6,304,497,664 |
| Quality FP8 + guarded head | 50.884 | 4,339,295,232 | 3,813,921,792 | 6,567,678,976 |

Balanced ABBA, not these standalone rates, controls retention. The guarded
head previously passed both 200- and 500-token ABBA plus the required and
13-case correctness gates.

## Exact BF16 weight-residency curve (2026-07-02)

The streamed-weight path now dispatches reloaded tensors by stable
shape/page-role keys instead of transient Python tensor IDs. A recycled ID
previously selected an incompatible Triton configuration and caused an illegal
memory access after layer eviction. Demanded CUDA pages are also recorded on
the consumer stream so allocator reuse cannot race queued kernels.

The budget policy retains a deterministic whole-layer prefix and reserves the
current plus next-prefetched layer. This avoids cyclic LRU thrashing, which
previously reloaded almost the entire model per token even with 6 GB available.

Same-command 10-token capacity curve:

| Budget | Steady tok/s | GPU resident weights | CPU exact weights | H2D bytes/token | Peak allocated | Peak reserved |
|:---|---:|---:|---:|---:|---:|---:|
| 1 GB | 2.624 | 681,586,688 | 6,150,197,248 | 5,499,591,773 | 1,016,142,848 | 6,769,606,656 |
| 2 GB | 3.146 | 1,619,062,784 | 6,150,197,248 | 4,604,728,227 | 1,972,493,312 | 6,769,606,656 |
| 4 GB | 5.658 | 3,650,260,992 | 6,150,197,248 | 2,665,857,210 | 4,044,585,984 | 6,769,606,656 |
| 6 GB | 28.139 | 5,681,459,200 | 6,150,197,248 | 726,986,193 | 6,116,678,656 | 6,748,635,136 |
| all | 45.347 | 6,150,197,248 | 0 | 0 | 6,276,070,400 | 6,289,358,848 |

These short runs characterize capacity, not the warmed 500-token throughput
claim. CPU offload is slower and is not presented as a speed optimization.
All four budgets passed the full 1/8/128-prefill by 1/10-step HF comparison
with zero top-1 failures, complete top-5 overlap, minimum cosine
0.9973128438, and minimum generated-token match 0.9—the same metrics as the
all-resident BF16 kernel path.

## SmolLM3 large-matvec bandwidth follow-up (2026-07-02)

GPU: NVIDIA GeForce RTX 5050 Laptop GPU. All results are batch-1 causal-KV
decode with 10 warmup tokens and the guarded 64-candidate BF16-verified
LM-head path.

| 500-token balanced ABBA | tok/s | Weight reads/token | Resident weights |
|:---|---:|---:|---:|
| Gate/up FP8 all, down FP8 8:28 | 48.811 | 3,813,921,792 | 4,339,295,232 |
| Gate/up FP8 all, down FP8 4:32, O-proj FP8 4:32 | 51.376 | 3,516,126,208 | 4,041,499,648 |

The experimental opt-in MLP/O profile is `+5.26%` faster while reading
297,795,584 fewer weight bytes/token. It preserved all required top-1/top-5
checks and every checkpoint top-1 in the 13-case stress suite. Worst required
and stress cosine were 0.998410 and 0.988403 respectively.

QKV FP8, lossless net-zero QKV packing (`+0.41%`), split-K down projection,
cache-policy variants, tensor-core skinny GEMM, interleaved gate/up storage,
and launch-only fusions were rejected.
Notably, QKV+O FP8 12:24 passed stress but was only `+0.05%` versus the simpler
O-only profile in direct 500-token ABBA.

Exact-kernel follow-up on the retained quality path:

| Candidate | Isolated result | Real decode | Correctness | Decision |
|:---|:---|:---|:---|:---|
| Row-swizzled/regrouped FP8 down | up to 1.88x | +2.84% at 500 tokens | new sampling_3 top-1 failures at steps 16/64 | reject |
| Original reduction groups | exact | +1.35% at 200 tokens | no arithmetic change | reject: speed |
| Paired 128-column loads | 1.76x, bit-exact | +0.50% at 200 tokens | bit-exact microkernel | reject: speed |
| FP16 products, FP32 accumulation | 1.35x, cosine 0.9999997 | -0.80% at 200 tokens | lower product precision | reject |
| On-chip K-chunk MLP pipeline | exact at best tile | kernel remained 28% slower | exact at best tile | reject |
| Static column-tiled BF16 head | +525,336,576 bytes | 1.7490 to 2.0017 ms | cosine about 1.0 | reject |

Live steady-decode telemetry showed 100% SM utilization, 47-50% memory
controller utilization, maximum 12,001 MHz memory clock, and the enforced
35 W laptop power ceiling. The 32 MB L2 can cache one 22.5 MB projection in a
repeated-weight microbenchmark, but cannot retain the roughly 4 GB/token model
working set. This is why repeated-layer kernel timings overstated every
scheduler candidate.

## Schema compiler and packed-MoE kernels (2026-07-01)

The converter/runtime is now driven by tensor roles and semantic traits rather
than a model-name allowlist. Synthetic archives are used only for operator
integration tests; they are not model throughput claims.

Validated integration cases:

- separate and fused QKV;
- separate and fused gate/up;
- packed top-k sparse MoE;
- packed MXFP4 expert blocks and E8M0 scales;
- MHA/GQA geometry, QKV/O/router/expert biases;
- learned attention sinks;
- per-layer sliding/full causal KV;
- default and YaRN RoPE;
- all-resident and CPU-offload/stream residency.

The packed MXFP4 kernel directly computes selected experts without expanding
their weights:

| Shape | Experts selected | Direct packed | Expand then matvec | Kernel speedup | Cosine |
|:---|---:|---:|---:|---:|---:|
| 5760x2880 gate/up | 4 of 32 | 0.634 ms | 18.183 ms | 28.69x | ~1.0 |
| 2880x2880 down | 4 of 32 | 0.292 ms | 8.422 ms | 28.86x | 1.0 |

These are isolated kernel results. No GPT-OSS-20B weights were downloaded or
benchmarked, so no 20B tokens/sec or end-to-end speedup is claimed.

Post-generalization regressions:

| SmolLM3-3B mode | Steps | tok/s | Correctness |
|:---|---:|---:|:---|
| BF16 causal KV | 200 | 32.38 | exact reference remains `exact_pass`, minimum cosine 0.999316 |
| gate/up FP8 + down FP8 8:28 | 300 | 48.03 | `ranking_pass`, minimum cosine 0.998042 |

Hot-path follow-up:

- Quality FP8 body plus Head8 reached 50.44 tok/s over 300 tokens, but remains
  experimental: prefill 128 / decode step 10 changed top-1, cosine was 0.99691,
  and generated-token match was 0.8.
- Two-stage BF16 head reduction and fused single-token attention were rejected
  because neither improved the real decode loop.

New `.thin` conversions place physical pages in execution-tape order and mark
row-major, expert-major, MXFP4 block, and scale layouts in each page record.

## SmolLM3-3B parity and quality-first selective FP8 (2026-07-01)

GPU: NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB, CUDA 12.8, PyTorch
2.11. All throughput below is real batch-1 causal-KV decode; no model work is
skipped.

The strict BF16 correctness reference now pins the HF model to eager attention.
This removes a validator-side comparison between different HF attention
backends. The required 1/8/128 prefill by 1/10 decode matrix reaches
`exact_pass`, with minimum cosine 0.999316 and complete top-1, top-5, and greedy
generation agreement.

| Mode | Steps | tok/s | Worst required-test cosine | Weight bytes saved | Status |
|:---|---:|---:|---:|---:|:---|
| Optimized BF16 | 200 | 31.44 | 0.99814 | 0 | Safe default |
| Gate/up FP8 | 100 | 43.37 | 0.99444 | 1,620,025,344 | Opt-in ranking-stable |
| Full MLP FP8 | 100 | 50.61 | 0.98378 | 2,431,328,256 | Experimental fast |
| Full MLP + O FP8 | 100 | 51.53 | 0.97096 | 2,582,028,288 | Experimental fast, larger drift |
| Gate/up all + down 8:28 | 500 | 46.41 | 0.99804 | 2,070,749,184 | Retained quality-first FP8 |
| Gate/up all + down 6:30 + O 4:32 | 500 | 49.88 | 0.99650 | 2,278,105,088 | Faster experimental FP8 |
| Full MLP + Head8 | 100 | 53.54 | 0.98303 | 2,431,328,256 body bytes | Rejected: tested top-1 mismatch |

The faster O-FP8 mode also reached 49.69 tok/s in a paired 300-token run and
51.01 tok/s in a separate cooler 300-token run. Its broader multi-turn
checkpoint cosine fell to 0.98471. Narrowing the quality-first mode to gate/up
plus down layers 8:28 measured 46.41 tok/s over 500 tokens, reached 0.99804 on
the required matrix and 0.99379 on the difficult cached-HF stress subset. Both
modes are selective quantization, not BF16 and not fully HF-equivalent.

After the retained mode was found, the following real-decode experiments failed
the 2% retention gate and were not enabled by default:

- fused FP8 gate/up plus SiLU: about 0.6%;
- SDPA attention: slower at both 300 and 1000 tokens;
- fused residual plus RMSNorm: slower;
- partial QKV FP8: no throughput improvement;
- role-specific loop/backend overrides: no repeatable 2% improvement;
- alternate LM-head backends: best median delta below 2%.

The faster 500-token profile reports about 3.87 GB estimated weight reads per
token, 192.9 GB/s effective bandwidth, 434 launches/token, 0.324 ms average
attention time per layer, 0.281 ms MLP time per layer, 0.242 ms O-projection
time per layer, and 1.502 ms for LM-head plus argmax. The quality-first profile
reads about 4.08 GB/token at 189.2 GB/s and spends 1.512 ms in LM-head plus
argmax. Remaining work is
bandwidth- and launch-sensitive; the straightforward fusion/backend candidates
above have been exhausted without a qualifying win.

The post-optimization behavioral matrix covers 13 cases: ordinary, reasoning,
code, Unicode, repetition and control-character prompts; a multi-turn chat
template; three deterministic temperature/top-p/top-k policies; a 128-token
generation; and 512/1024-token contexts. Long-context KV allocation grew
linearly and stayed below the 8 GB GPU limit. Stochastic free-running sequences
can diverge early from small logit changes, including on BF16 cached-HF
comparisons, so sampled-token equality is reported separately from
teacher-forced logit parity and distribution distance.

## Native Single-Token CUDA/BF16 Decode Optimization (2026-06-30)

GPU: NVIDIA GeForce RTX 5050 Laptop GPU, CUDA 12.8, PyTorch 2.11.
Archive: `/tmp/thintensor-first-run/qwen3-0.6b.thin`.

The retained BF16 hot-path changes are static direct-reference layer plans,
multi-pointer QKV/gate-up Triton launches, direct grouped-V indexing in
`o_proj`, shape launch rules, and one-sync all-resident loading.

| Mode | Baseline | Retained result |
| --- | ---: | ---: |
| Raw forward, no KV/head | 4.511 ms | 4.275 ms |
| Greedy BF16, 500 steps | 172.70 tok/s | 173.05 tok/s |
| Greedy BF16 KV append | 173.15 tok/s | 174.48 tok/s |

The raw-forward gain is measurable. End-to-end BF16 moved only slightly and
varied down to 171.60 tok/s in the standalone CLI under laptop power/thermal
behavior, so it is not claimed as a material greedy speedup.

Opt-in `--lm-head-fp8` reduced the head section from roughly 1.59 ms to
0.68-0.75 ms and estimated weight reads from 1,191,968,768 to 1,036,386,304
bytes/token. The final 500-step CLI run reached 202.53 tok/s (4.938 ms/token);
repeated 500-step event runs ranged from 187.66 to 201.31 tok/s as the laptop
throttled before launch-plan metadata caching was added.
The fixed token stayed `9`; mean/max absolute logit deltas were
0.09348/0.546875, and four of five top IDs matched. The mode adds 155,582,464
bytes for the FP8 head and remains opt-in.

Rejected experiments:

- Per-shape `torch.mv` switching: isolated 1024x1024 wins regressed full raw
  forward, so the guarded autotuner keeps Triton for current shapes.
- Runtime QKV/gate-up concatenation: improved raw speed but roughly doubled
  resident weights and did not improve greedy decode.
- Layer-matrix FP8 and scalar-dequant INT8: changed outputs and/or regressed to
  79-97 tok/s.
- Matvec+residual and Triton elementwise fusion: regressed full greedy decode.

Environment: local Codex workspace, Linux, Rust debug build.

## Qwen3 0.6B HF Conversion

Source: `Qwen/Qwen3-0.6B` downloaded to `/tmp/thintensor-qwen3-0.6b`.

| Metric | Value |
| --- | ---: |
| HF safetensors files | 1 |
| HF tensor count | 311 |
| HF tensor bytes | 1,503,264,768 |
| ThinTensor pages | 311 |
| ThinTensor archive bytes | 1,503,445,832 |
| Manifest bytes | 149,848 |
| Data offset | 181,064 |
| Convert command | `thintensor convert-hf /tmp/thintensor-qwen3-0.6b /tmp/qwen3-0.6b.thin` |
| Convert elapsed, warm cache | 1.10 s |
| Verify result | ok |

## Load / Plan Probes

These are metadata/path probes, not full inference benchmarks yet.

| Probe | Input | Time | RSS | Peak RSS | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| HF metadata load | `/tmp/thintensor-qwen3-0.6b` | 2.34 ms | 5.62 MiB | 7.46 MiB | mmap safetensors metadata and tensor views |
| Thin archive open | `/tmp/qwen3-0.6b.thin` | 3.08 ms | 5.68 MiB | 5.68 MiB | read header, manifest, page table |
| Thin verify | `/tmp/qwen3-0.6b.thin` | 469.28 ms | 12.10 MiB | 302.67 MiB | hashes all raw page blobs |
| Thin plan | `/tmp/qwen3-0.6b.thin --backend cuda --vram 8GB --ctx 8192` | 0.05 ms | 12.10 MiB | 302.67 MiB | after verified archive open |

## Plan Estimate

| Metric | Value |
| --- | ---: |
| Backend | cuda |
| VRAM budget | 8.00 GiB |
| Context | 8192 |
| Weight bytes | 1.40 GiB |
| Scratch bytes | 64.00 MiB |
| KV cache bytes | 122.50 MiB |
| Total estimated | 1.58 GiB |
| Status | OK |

## Largest Pages

| Page | Bytes |
| --- | ---: |
| `lm_head.weight` | 311,164,928 |
| `model.embed_tokens.weight` | 311,164,928 |
| `model.layers.0.mlp.down_proj.weight` | 6,291,456 |
| `model.layers.0.mlp.gate_proj.weight` | 6,291,456 |
| `model.layers.0.mlp.up_proj.weight` | 6,291,456 |

## Missing Measurements

- Cold-cache load timing.
- Repack/layout scan timing.

## Runtime Generation

Command shape:

```bash
thintensor bench-baseline /tmp/thintensor-qwen3-0.6b --tokens 128 --device cuda
thintensor bench-archive /tmp/thintensor-first-run/qwen3-0.6b.thin --hf-dir /tmp/thintensor-qwen3-0.6b --tokens 128 --device cuda
```

The ThinTensor runtime path hydrates a Transformers model from `.thin` page
records. It uses mmap-backed CPU tensor views, tied-weight aliasing, meta-device
initialization, `malloc_trim`, and a GPU thermal stop guard.

| Run | Tokens/sec | Load time | GPU peak | RSS delta | Peak temp | Thermal stop |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| HF baseline | 68.75 | 0.753 s | 1.13 GiB | 1.09 GiB | 44 C | false |
| ThinTensor archive | 67.34 | 0.825 s | 1.13 GiB | 1.11 GiB | 45 C | false |

This run was mixed: ThinTensor matched GPU memory but was slower. That selected
execution-ordered repack as the next optimization.

## Execution-Ordered Repack

Command:

```bash
thintensor repack /tmp/thintensor-first-run/qwen3-0.6b.thin /tmp/thintensor-first-run/qwen3-0.6b.exec.thin --layout execution_ordered_v1
```

Verification: `verify: ok`.

The repacked page table starts with `model.embed_tokens.weight`, then layer-0
attention/MLP pages in execution order, instead of lexicographic tensor order.

Sequential runtime comparison:

| Archive | Tokens/sec | Load time | GPU peak | RSS delta | Peak temp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original `.thin` | 68.58 | 0.828 s | 1.13 GiB | 1.08 GiB | 44 C |
| Execution-ordered `.thin` | 68.85 | 0.865 s | 1.13 GiB | 1.08 GiB | 45 C |

Repack is a small win here: same GPU peak, slightly higher load time, slightly
better generation throughput. Larger cold-cache and streaming tests are still
needed.


## First-Run Runtime Pipeline

| Metric | HF baseline | ThinTensor archive |
| --- | ---: | ---: |
| Load time | 0.757 s | 0.830 s |
| Generate time | 1.852 s | 1.837 s |
| Tokens/sec | 69.13 | 69.66 |
| GPU peak allocated | 1.13 GiB | 1.13 GiB |
| RSS delta | 1.08 GiB | 1.14 GiB |
| Peak GPU temp | 42 C | 43 C |

Convert time: 1.091 s.
TPS ratio: 1.008.
Next optimization: reduce ThinTensor host RSS by closing mmaps and avoiding duplicate state dict storage.


## First-Run Runtime Pipeline

| Metric | HF baseline | ThinTensor archive |
| --- | ---: | ---: |
| Load time | 0.753 s | 0.825 s |
| Generate time | 1.862 s | 1.901 s |
| Tokens/sec | 68.75 | 67.34 |
| GPU peak allocated | 1.13 GiB | 1.13 GiB |
| RSS delta | 1.09 GiB | 1.11 GiB |
| Peak GPU temp | 44 C | 45 C |

Convert time: 1.117 s.
TPS ratio: 0.980.
Next optimization: repack pages in execution order and benchmark scan/read locality.


## First-Run Runtime Pipeline

| Metric | HF baseline | ThinTensor archive |
| --- | ---: | ---: |
| Load time | 0.745 s | 0.835 s |
| Generate time | 1.850 s | 1.842 s |
| Tokens/sec | 69.19 | 69.50 |
| GPU peak allocated | 1.13 GiB | 1.13 GiB |
| RSS delta | 1.08 GiB | 1.11 GiB |
| Peak GPU temp | 43 C | 44 C |

Convert time: 1.085 s.
Stats: 311 pages, 199 execution stages, 0 unreferenced pages.
TPS ratio: 1.004.
Next optimization: move to execution-ordered repack and hot/cold page profiles.

Optimization loop 1: execution-ordered repack

| Metric | Value |
| --- | ---: |
| Tokens/sec | 67.39 |
| Load time | 0.842 s |
| GPU peak allocated | 1.13 GiB |
| RSS delta | 1.09 GiB |
| TPS delta vs previous | -2.11 |
| TPS ratio vs previous | 0.970 |

## Stats / Budget Compiler Probe

Command shape:

```bash
thintensor stats /tmp/thintensor-first-run/qwen3-0.6b.pipeline.thin
thintensor plan /tmp/thintensor-first-run/qwen3-0.6b.pipeline.thin --backend cuda --vram 1GB --ctx 8192 --kv-dtype q4 --weight-residency stream
thintensor plan /tmp/thintensor-first-run/qwen3-0.6b.pipeline.thin --backend cuda --vram 512MiB --ctx 8192 --kv-dtype fp16 --batch 2 --weight-residency all --gpu-fraction 0.9
```

Stats output on Qwen3 0.6B:

| Metric | Value |
| --- | ---: |
| Pages | 311 |
| Execution stages | 199 |
| Referenced pages | 311 |
| Empty stages | 28 |
| Unreferenced pages | 0 |
| Raw/stored bytes | 1.40 GiB |
| Per-layer bytes | 30.00 MiB |
| `embed_tokens` | 296.75 MiB |
| `lm_head` | 296.75 MiB |
| MLP down/gate/up total | 504.00 MiB |
| Attention q/o total | 224.00 MiB |
| Attention k/v total | 112.00 MiB |

Planner probes:

| Plan | Status | Total | Notes |
| --- | --- | ---: | --- |
| 1GB, ctx 8192, q4 KV, streamed weights | OK | 483.30 MiB | keeps globals resident, streams 840.12 MiB layer weights |
| 512MiB, usable 460.80 MiB, ctx 8192, fp16 KV, batch 2, all weights | NOT ENOUGH VRAM | 2.11 GiB | suggests reducing batch, q4 KV, streaming/offload |
| 1GB, ctx 8192, q2 KV, offload last 14 layers | OK | 849.11 MiB | resident 716.81 MiB, streamed 420.06 MiB |

The planner now counts exact duplicate shared weights once for runtime memory:
Qwen3 0.6B has 1.40 GiB physical archive weights, 1.11 GiB unique runtime
weights, and 296.75 MiB exact shared-weight savings from the tied
embedding/lm_head pair.

## Target-VRAM Profiles

Commands:

```bash
thintensor profile /tmp/thintensor-first-run/qwen3-0.6b.pipeline.thin --target-vram 1GB --ctx 8192 --backend cuda --out /tmp/thintensor-first-run-pipeline/qwen3.profile-1gb.json
thintensor profile /tmp/thintensor-first-run/qwen3-0.6b.pipeline.thin --target-vram 4GB --ctx 8192 --backend cuda --out /tmp/thintensor-first-run-pipeline/qwen3.profile-4gb.json
```

| Target | Selected candidate | Recommended layout | Status | Total | Resident weights | Streamed/offloaded | KV |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1GB | `offload_last_half_q4` | `hot_stream_v1` | OK | 903.36 MiB | 716.81 MiB | 420.06 MiB | q4 |
| 4GB | `all_q4` | `execution_ordered_v1` | OK | 1.29 GiB | 1.11 GiB | 0 B | q4 |

Profile anatomy for Qwen3 0.6B:

| Metric | Value |
| --- | ---: |
| Always-hot pages | 3 |
| Streamable pages | 308 |
| CPU offload candidates | 14 |
| Per-layer cost | 30.00 MiB |
| Shared-weight savings | 296.75 MiB |

The 1GB profile chooses a less aggressive offload-half plan before full
streaming because it is the first fitting low-disruption candidate. The sidecar
also includes `stream_q4` and `stream_q2`, which fit at lower estimated totals
but imply higher per-token host/GPU weight movement.

## Load Simulation / Execution Locality

Command shape:

```bash
thintensor simulate-load /tmp/thintensor-first-run/qwen3-0.6b.pipeline.thin --backend cuda --vram 1GB --ctx 8192 --kv-dtype q4 --weight-residency stream --prefetch-pages 4
thintensor simulate-load /tmp/thintensor-first-run-pipeline/model.exec1.thin --backend cuda --vram 1GB --ctx 8192 --kv-dtype q4 --weight-residency stream --prefetch-pages 4
```

| Archive | Sequential ratio | Sequential transitions | Non-sequential | Backward jumps |
| --- | ---: | ---: | ---: | ---: |
| Original `.thin` | 0.094 | 29 | 281 | 170 |
| Execution-ordered `.thin` | 1.000 | 310 | 0 | 0 |
| Hot-stream `.thin` | 0.990 | 307 | 3 | 2 |

Prefetch estimate with `--prefetch-pages 4`:

| Archive | Resident groups | Stream read groups | Prefetch IO ops/token | Max stream group | Recommended staging |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original `.thin` | 2 | 280 | 280 | 2 pages / 12.00 MiB | 18.00 MiB |
| Execution-ordered `.thin` | 2 | 1 | 77 | 308 pages / 840.12 MiB | 18.00 MiB |
| Hot-stream `.thin` | 1 | 1 | 77 | 308 pages / 840.12 MiB | 18.00 MiB |

Streaming profile estimate for both archives:

| Metric | Value |
| --- | ---: |
| Resident pages | 3 |
| Streamed/evicted pages | 308 |
| Initial archive read | 593.50 MiB |
| Peak memory | 483.30 MiB |
| Streamed weights per generated token | 840.12 MiB |
| KV writes per generated token | 15.31 KiB |
| Total bytes moved estimate per generated token | 904.14 MiB |

This confirms the execution-ordered repack did what the format is supposed to
do: it turns the runtime tape into sequential archive traversal. The runtime
tokens/sec benchmark was noisy, but the layout metric is now unambiguous.

Hot-stream layout command:

```bash
thintensor repack /tmp/thintensor-first-run/qwen3-0.6b.pipeline.thin /tmp/thintensor-first-run-pipeline/model.hot.thin --layout hot_stream_v1
```

Hot-stream runtime smoke:

| Metric | Value |
| --- | ---: |
| Tokens/sec, 128 tokens | 68.61 |
| Load time | 0.831 s |
| GPU peak allocated | 1.13 GiB |
| RSS delta | 1.08 GiB |
| Peak GPU temp | 44 C |

Hot-stream is not a claimed tokens/sec win yet; its measured improvement is
layout-level: one contiguous hot resident group plus one contiguous streamed
layer group.

## Release CLI Demo

Commands run with `target/release/thintensor`:

```bash
target/release/thintensor verify /tmp/thintensor-first-run/qwen3-0.6b.pipeline.thin
target/release/thintensor stats /tmp/thintensor-first-run/qwen3-0.6b.pipeline.thin --json
target/release/thintensor plan /tmp/thintensor-first-run/qwen3-0.6b.pipeline.thin --backend cuda --vram 4GB --ctx 4096 --kv-dtype q4 --json
target/release/thintensor profile /tmp/thintensor-first-run/qwen3-0.6b.pipeline.thin --target-vram 1GB --backend cuda --ctx 8192
target/release/thintensor simulate-load /tmp/thintensor-first-run-pipeline/model.hot.thin --backend cuda --vram 1GB --ctx 8192 --kv-dtype q4 --weight-residency stream --prefetch-pages 4
```

Result: all release commands succeeded. The release profile selected
`offload_last_half_q4`, recommended `hot_stream_v1`, and the hot-stream release
simulation matched the debug metrics: resident groups `1`, stream groups `1`,
and prefetch IO ops/token `77`.


## First-Run Runtime Pipeline

| Metric | HF baseline | ThinTensor archive |
| --- | ---: | ---: |
| Load time | 0.772 s | 0.848 s |
| Generate time | 1.046 s | 1.039 s |
| Tokens/sec | 61.19 | 61.62 |
| GPU peak allocated | 1.13 GiB | 1.13 GiB |
| RSS delta | 1.08 GiB | 1.07 GiB |
| Peak GPU temp | 45.5 C | 45.0 C |

Convert time: 1.104 s.
Tokens/context: 64/2048.
Stats: 311 pages, 199 execution stages, 0 unreferenced pages.
Trials: HF=2, ThinTensor=2.
TPS ratio: 1.007.
Next optimization: move to execution-ordered repack and hot/cold page profiles.

TPS range:

| Run | Min | Median | Max |
| --- | ---: | ---: | ---: |
| HF baseline | 60.91 | 61.19 | 61.48 |
| ThinTensor archive | 61.51 | 61.62 | 61.73 |

## SmolLM3 Runtime Hot-Path Optimizer

The retained LM-head optimization uses a row-scaled FP8 execution copy only
to select 64 candidate vocabulary IDs. It then reads those rows from the tied
BF16 head and performs the final BF16 argmax with baseline tie-breaking. Full
logits remain BF16.

Balanced `B-A-A-B` results on the RTX 5050 Laptop GPU:

| Path | Tokens | Baseline steady tok/s | Guarded steady tok/s | Delta |
| --- | ---: | ---: | ---: | ---: |
| Full BF16 | 200 | 32.717 | 33.714 | +3.05% |
| Full BF16 | 500 | 32.440 | 33.718 | +3.94% |
| Quality FP8 | 200 | 48.486 | 51.026 | +5.24% |
| Quality FP8 | 500 | 47.608 | 49.798 | +4.60% |

Memory and byte tradeoff:

| Path | Baseline resident weights | Guarded resident weights | Baseline read bytes/token | Guarded read bytes/token |
| --- | ---: | ---: | ---: | ---: |
| Full BF16 | 6,150,197,248 | 6,413,378,560 | 6,149,898,240 | 5,888,005,120 |
| Quality FP8 | 4,076,113,920 | 4,339,295,232 | 4,075,814,912 | 3,813,921,792 |

The required 1/8/128-by-1/10 matrix and 13-case stress grid were unchanged
against each path's own baseline. This remains an explicit profile because the
extra execution head costs 263,181,312 resident bytes and shortlist recall is
model/profile specific.

Quality command additions:

```bash
--lm-head-fp8 --keep-bf16-lm-head --lm-head-topk-guard 64
```

Rejected controls included fused residual+RMSNorm, fused scaled gate/up+SiLU,
split-K down projection, interleaved gate/up storage, two-stream gate/up, and
pipeline-stage tuning.
Their microkernel or launch-count changes did not clear the 2% full-decode
gate. See `runtime_optimization_report.md` and the machine-readable optimizer
artifacts under `benchmark_results/runtime_optimizer/`.
