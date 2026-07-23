# Qwen3.6-27B full-model result

## Scope and model identity

- Model: `Qwen/Qwen3.6-27B`, source revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Native Qwen3.5 text path: 64 text layers, 48 linear-attention/DeltaNet layers and 16 full-attention layers (every fourth layer), 248,320-token vocabulary.
- Archive: `Qwen--Qwen3.6-27B.thin`; verified streaming conversion size 53,792,526,418 bytes (50.10 GiB). The source shards were consumed only after verified archive writes.

## Selected quality profile

The automatic profile is archive-specific: all dense text projections use input-RMS weighted **affine INT4, group 32**; `gate_up`, `down_proj`, `qkv`, and `o_proj` in edge layers 0 and 63 use FP8; the LM head remains BF16. It is tied to the paired calibration JSON/NPZ and their SHA-256 values in the archive sidecar. A normal command has no internal flags:

```bash
thintensor run Qwen/Qwen3.6-27B --prompt 'Answer with exactly one word: ready.' --max-new-tokens 2 --context 256 --json
```

The final smoke used the selected plan, generated two tokens, had 6,544,000,960 resident-weight bytes, a 7,000,000,000-byte effective page-pool budget, and a 7,132,409,344-byte peak allocation. The 1.549 tok/s interactive value is deliberately not compared with benchmark throughput.

Official-HF sequential full-vocabulary controls were taken at the fixed trajectory points for coding, general explanation, mathematics, long continuation, and instruction style. The selected native configuration preserved top-1 at every record. Minimum centered-logit cosine was respectively 0.995623, 0.995586, 0.995472, 0.994478, and 0.998062. Raw controls and `qwen36_quality_summary.json` retain the exact records.

## Decode measurement

Public command for every sample:

```bash
thintensor bench Qwen/Qwen3.6-27B --profiles auto --steps 200 --warmup 10 --json
```

The default safe-envelope baseline five-run median was **0.810440 tok/s** (1233.898 ms/token), with 5,978,051,904 resident bytes and 14.643 GB/token H2D movement. The selected archive calibration records the highest stable 7 GB page-pool budget. Its five independent fresh-process measurements were 0.841375, 0.843761, 0.847764, 0.840098, and 0.835650 tok/s; median **0.841375 tok/s** (1188.530 ms/token), a **3.817%** gain.

At 7 GB it maintained 6,544,000,960 persistent bytes across 263 pages, moved 14.080 GB/token H2D, and reported 15.188 GB/s effective transfer bandwidth. Every sample used 10 warmup and 200 measured causal-KV tokens and is marked steady-state eligible in its raw JSON.

## Honest speed verdict

The `>=10 tok/s` stretch target was **not met**: the accepted result is 11.89x below it. This is a bandwidth/residency limit on the 8 GiB device, not a benchmark shortcut: the high-quality configuration reads 18.051 GB of weights per token, has 210–221 seconds of pinned CPU staging over a 200-token decode, and still transfers about 14 GB/token after persistence. Raising the page pool to 7.5 GB OOMed during warmup while allocating an 86 MiB FP8 projection.

The alternatives that could reduce transfers failed the full-vocabulary quality gate: symmetric G32 INT4 (0.750267 coding cosine), affine G64 (instruction top-1 mismatch), Q3 (instruction top-1 mismatch), and Q2 (0.219842 coding cosine and top-1 mismatches). Disabling pinned staging also regressed to 0.691389 tok/s. These are retained as rejections in `optimization_history.json`, not presented as speed wins.

## Reproduction artifacts

- Quality: `qwen36_hf_reference_*.json/.npz`, `qwen36_quality_trajectories/`, and `qwen36_affine_calibrated_fp8_edges_*.json`.
- Baseline: `qwen36_public_bench_run1.json` through `run5.json`.
- Selected five-run result: `qwen36_candidate_7gb_budget_run1.json` plus `qwen36_public_bench_7gb_run2.json` through `run5.json`.
- Normal automatic-profile proof: `qwen36_plan_calibrated_7gb.json`, `qwen36_bench_calibrated_7gb_dry_run.json`, and `qwen36_normal_cli_calibrated_7gb_smoke.json`.

## Cleanup before GPT-OSS

After preserving the artifacts above, the Qwen archive (53,792,526,418 bytes), its source metadata directory, sidecars, and 784 precisely matched Qwen-derived cache entries (21,859,493,584 bytes) were removed. The shared 1.7B smoke cache was not touched. Disk free space increased from 12,808,536,064 to 88,486,363,136 bytes. The exact inventory is `qwen36_cleanup.json`.
