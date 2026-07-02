#!/usr/bin/env python3
"""First-run ThinTensor pipeline.

Builds the CLI, gets a Qwen/Llama-style HF model, converts it, verifies it,
benchmarks HF vs ThinTensor runtime paths, compares results, and records them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    work_dir = args.work_dir.resolve()
    hf_dir = args.hf_dir.resolve() if args.hf_dir else work_dir / model_dir_name(args.model)
    thin = args.thin.resolve() if args.thin else work_dir / "model.thin"
    work_dir.mkdir(parents=True, exist_ok=True)

    run(["cargo", "build"], cwd=repo)
    if not hf_dir.exists() or not any(hf_dir.glob("*.safetensors")):
        run(["hf", "download", args.model, "--local-dir", str(hf_dir)], cwd=repo)

    convert_start = time.perf_counter()
    run([bin_path(repo), "convert-hf", str(hf_dir), str(thin)], cwd=repo)
    convert_s = time.perf_counter() - convert_start
    run([bin_path(repo), "verify", str(thin)], cwd=repo)
    stats = write_stats(repo, thin, work_dir / "model.stats.json")

    baseline = run_baseline_bench(repo, hf_dir, args)
    archive = run_archive_bench(repo, thin, hf_dir, args)

    comparison = compare(baseline, archive)
    iterations = optimize_loop(
        repo=repo,
        hf_dir=hf_dir,
        start_thin=thin,
        baseline=baseline,
        start_archive=archive,
        start_comparison=comparison,
        work_dir=work_dir,
        args=args,
    )
    result = {
        "model": args.model,
        "hf_dir": str(hf_dir),
        "thin": str(thin),
        "tokens": args.tokens,
        "ctx": args.ctx,
        "convert_s": convert_s,
        "stats": stats,
        "baseline": baseline,
        "archive": archive,
        "comparison": comparison,
        "optimization_iterations": iterations,
    }

    out_json = work_dir / "first-run-result.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    append_benchmarks(repo / "BENCHMARKS.md", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--hf-dir", type=Path)
    parser.add_argument("--thin", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/thintensor-first-run"))
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-gpu-temp", type=int, default=87)
    parser.add_argument("--opt-loops", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    return parser.parse_args()


def compare(baseline: dict[str, Any], archive: dict[str, Any]) -> dict[str, Any]:
    base_tps = baseline["tokens_per_s"]
    thin_tps = archive["tokens_per_s"]
    base_gpu = baseline.get("gpu_peak_allocated_bytes") or 0
    thin_gpu = archive.get("gpu_peak_allocated_bytes") or 0
    base_rss = baseline.get("rss_delta_bytes") or 0
    thin_rss = archive.get("rss_delta_bytes") or 0
    gpu_slack = 32 * 1024 * 1024
    rss_slack = 128 * 1024 * 1024
    return {
        "tokens_per_s_delta": thin_tps - base_tps,
        "tokens_per_s_ratio": thin_tps / base_tps if base_tps else None,
        "gpu_peak_allocated_delta_bytes": thin_gpu - base_gpu,
        "rss_delta_bytes": thin_rss - base_rss,
        "winner": "thin" if thin_tps > base_tps and thin_gpu <= base_gpu + gpu_slack else "mixed",
        "next_optimization": next_optimization(
            thin_tps,
            base_tps,
            thin_gpu,
            base_gpu,
            thin_rss,
            base_rss,
            gpu_slack,
            rss_slack,
        ),
    }


def next_optimization(
    thin_tps: float,
    base_tps: float,
    thin_gpu: int,
    base_gpu: int,
    thin_rss: int,
    base_rss: int,
    gpu_slack: int,
    rss_slack: int,
) -> str:
    if thin_gpu > base_gpu + gpu_slack:
        return "reduce ThinTensor GPU peak via tied weights, streaming, or residency planner"
    if thin_rss > base_rss + rss_slack:
        return "reduce ThinTensor host RSS by closing mmaps and avoiding duplicate state dict storage"
    if thin_tps <= base_tps:
        return "repack pages in execution order and benchmark scan/read locality"
    return "move to execution-ordered repack and hot/cold page profiles"


def optimize_loop(
    repo: Path,
    hf_dir: Path,
    start_thin: Path,
    baseline: dict[str, Any],
    start_archive: dict[str, Any],
    start_comparison: dict[str, Any],
    work_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    iterations: list[dict[str, Any]] = []
    current_thin = start_thin
    current_archive = start_archive
    current_comparison = start_comparison

    for index in range(args.opt_loops):
        action = choose_action(current_comparison)
        if action is None:
            iterations.append(
                {
                    "index": index,
                    "action": "stop",
                    "reason": "no automatic optimization action selected",
                }
            )
            break

        if action == "execution_ordered_repack":
            out_thin = work_dir / f"model.exec{index + 1}.thin"
            try:
                run(
                    [
                        bin_path(repo),
                        "repack",
                        str(current_thin),
                        str(out_thin),
                        "--layout",
                        "execution_ordered_v1",
                    ],
                    cwd=repo,
                )
                run([bin_path(repo), "verify", str(out_thin)], cwd=repo)
                stats = write_stats(repo, out_thin, work_dir / f"model.exec{index + 1}.stats.json")
                archive = run_archive_bench(repo, out_thin, hf_dir, args)
            except subprocess.CalledProcessError as exc:
                iterations.append(
                    {
                        "index": index,
                        "action": action,
                        "status": "blocked",
                        "reason": f"command failed with exit {exc.returncode}: {exc.cmd}",
                    }
                )
                break
            comparison = compare(baseline, archive)
            iterations.append(
                {
                    "index": index,
                    "action": action,
                    "status": "ok",
                    "thin": str(out_thin),
                    "stats": stats,
                    "archive": archive,
                    "comparison_to_baseline": comparison,
                    "comparison_to_previous": compare_runtime(current_archive, archive),
                }
            )
            current_thin = out_thin
            current_archive = archive
            current_comparison = comparison
            continue

        iterations.append(
            {
                "index": index,
                "action": "stop",
                "reason": f"action {action} is not implemented",
            }
        )
        break

    return iterations


def choose_action(comparison: dict[str, Any]) -> str | None:
    next_step = str(comparison.get("next_optimization", ""))
    if "repack" in next_step:
        return "execution_ordered_repack"
    return None


def run_archive_bench(
    repo: Path,
    thin: Path,
    hf_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return run_runtime_trials(
        "thin_archive",
        args.trials,
        lambda: run_json(
            [
                bin_path(repo),
                "bench-archive",
                str(thin),
                "--hf-dir",
                str(hf_dir),
                "--tokens",
                str(args.tokens),
                "--ctx",
                str(args.ctx),
                "--device",
                args.device,
                "--max-gpu-temp",
                str(args.max_gpu_temp),
                "--json",
            ],
            cwd=repo,
        ),
    )


def run_baseline_bench(repo: Path, hf_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return run_runtime_trials(
        "hf_baseline",
        args.trials,
        lambda: run_json(
            [
                bin_path(repo),
                "bench-baseline",
                str(hf_dir),
                "--tokens",
                str(args.tokens),
                "--ctx",
                str(args.ctx),
                "--device",
                args.device,
                "--max-gpu-temp",
                str(args.max_gpu_temp),
                "--json",
            ],
            cwd=repo,
        ),
    )


def run_runtime_trials(
    label: str,
    trials: int,
    run_one: Any,
) -> dict[str, Any]:
    results = []
    for index in range(max(1, trials)):
        result = run_one()
        result["trial_index"] = index
        results.append(result)
    aggregate = aggregate_trials(results)
    aggregate["label"] = label
    aggregate["trial_count"] = len(results)
    aggregate["trials"] = results
    return aggregate


def aggregate_trials(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = dict(results[0])
    for key in [
        "load_s",
        "generate_s",
        "tokens_per_s",
        "rss_delta_bytes",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
        "gpu_peak_temp_c",
    ]:
        values = [
            result.get(key)
            for result in results
            if isinstance(result.get(key), (int, float)) and not isinstance(result.get(key), bool)
        ]
        if not values:
            continue
        aggregate[key] = statistics.median(values)
        aggregate[f"{key}_min"] = min(values)
        aggregate[f"{key}_max"] = max(values)
        aggregate[f"{key}_mean"] = statistics.mean(values)
    aggregate["thermal_stop"] = any(bool(result.get("thermal_stop")) for result in results)
    return aggregate


def write_stats(repo: Path, thin: Path, out_json: Path) -> dict[str, Any]:
    stats = run_json([bin_path(repo), "stats", str(thin), "--json"], cwd=repo)
    out_json.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "json_path": str(out_json),
        "page_count": stats["page_count"],
        "execution_stages": stats["execution"]["stage_count"],
        "referenced_pages": stats["execution"]["referenced_page_count"],
        "unreferenced_pages": len(stats["execution"]["unreferenced_pages"]),
        "total_raw_bytes": stats["total_raw_bytes"],
        "largest_pages": stats["largest_pages"][:5],
        "per_op": stats["per_op"][:8],
    }


def compare_runtime(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_tps = previous["tokens_per_s"]
    current_tps = current["tokens_per_s"]
    return {
        "tokens_per_s_delta": current_tps - previous_tps,
        "tokens_per_s_ratio": current_tps / previous_tps if previous_tps else None,
        "load_s_delta": current["load_s"] - previous["load_s"],
        "generate_s_delta": current["generate_s"] - previous["generate_s"],
        "gpu_peak_allocated_delta_bytes": (current.get("gpu_peak_allocated_bytes") or 0)
        - (previous.get("gpu_peak_allocated_bytes") or 0),
        "rss_delta_bytes": (current.get("rss_delta_bytes") or 0)
        - (previous.get("rss_delta_bytes") or 0),
    }


def append_benchmarks(path: Path, result: dict[str, Any]) -> None:
    baseline = result["baseline"]
    archive = result["archive"]
    comparison = result["comparison"]
    stats = result["stats"]
    block = f"""

## First-Run Runtime Pipeline

| Metric | HF baseline | ThinTensor archive |
| --- | ---: | ---: |
| Load time | {baseline["load_s"]:.3f} s | {archive["load_s"]:.3f} s |
| Generate time | {baseline["generate_s"]:.3f} s | {archive["generate_s"]:.3f} s |
| Tokens/sec | {baseline["tokens_per_s"]:.2f} | {archive["tokens_per_s"]:.2f} |
| GPU peak allocated | {fmt_bytes(baseline.get("gpu_peak_allocated_bytes"))} | {fmt_bytes(archive.get("gpu_peak_allocated_bytes"))} |
| RSS delta | {fmt_bytes(baseline.get("rss_delta_bytes"))} | {fmt_bytes(archive.get("rss_delta_bytes"))} |
| Peak GPU temp | {baseline.get("gpu_peak_temp_c")} C | {archive.get("gpu_peak_temp_c")} C |

Convert time: {result["convert_s"]:.3f} s.
Tokens/context: {result["tokens"]}/{result["ctx"]}.
Stats: {stats["page_count"]} pages, {stats["execution_stages"]} execution stages, {stats["unreferenced_pages"]} unreferenced pages.
Trials: HF={baseline.get("trial_count", 1)}, ThinTensor={archive.get("trial_count", 1)}.
TPS ratio: {comparison["tokens_per_s_ratio"]:.3f}.
Next optimization: {comparison["next_optimization"]}.
"""
    if baseline.get("trial_count", 1) > 1 or archive.get("trial_count", 1) > 1:
        block += f"""
TPS range:

| Run | Min | Median | Max |
| --- | ---: | ---: | ---: |
| HF baseline | {baseline["tokens_per_s_min"]:.2f} | {baseline["tokens_per_s"]:.2f} | {baseline["tokens_per_s_max"]:.2f} |
| ThinTensor archive | {archive["tokens_per_s_min"]:.2f} | {archive["tokens_per_s"]:.2f} | {archive["tokens_per_s_max"]:.2f} |
"""
    for item in result["optimization_iterations"]:
        if item.get("status") == "blocked":
            block += f"""
Optimization loop {item["index"] + 1} blocked during {item["action"]}: {item["reason"]}.
"""
        elif item.get("action") == "execution_ordered_repack":
            current = item["archive"]
            previous = item["comparison_to_previous"]
            block += f"""
Optimization loop {item["index"] + 1}: execution-ordered repack

| Metric | Value |
| --- | ---: |
| Tokens/sec | {current["tokens_per_s"]:.2f} |
| Load time | {current["load_s"]:.3f} s |
| GPU peak allocated | {fmt_bytes(current.get("gpu_peak_allocated_bytes"))} |
| RSS delta | {fmt_bytes(current.get("rss_delta_bytes"))} |
| TPS delta vs previous | {previous["tokens_per_s_delta"]:.2f} |
| TPS ratio vs previous | {previous["tokens_per_s_ratio"]:.3f} |
"""
        elif item.get("action") == "stop":
            block += f"""
Optimization loop stopped: {item["reason"]}.
"""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(block)


def run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def run_json(args: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE)
    return json.loads(proc.stdout)


def bin_path(repo: Path) -> str:
    return str(repo / "target/debug/thintensor")


def model_dir_name(model: str) -> str:
    return model.replace("/", "--")


def fmt_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    units = [("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)]
    for name, scale in units:
        if value >= scale:
            return f"{value / scale:.2f} {name}"
    return f"{value} B"


if __name__ == "__main__":
    main()
