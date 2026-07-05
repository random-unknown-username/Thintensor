#!/usr/bin/env python3
"""ThinTensor CLI — unified command-line interface.

Usage:
    thintensor convert ./SmolLM3-3B --out SmolLM3-3B.thin
    thintensor run SmolLM3-3B.thin --prompt "Hello" --profile balanced
    thintensor explain max-performance
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Rich / plain-text output helpers
# ---------------------------------------------------------------------------

_RICH_AVAILABLE = False
_console = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from rich.console import Console
    from rich.table import Table

    _RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    pass


def _print(*args: Any, **kwargs: Any) -> None:
    if _RICH_AVAILABLE and _console:
        _console.print(*args, **kwargs)
    else:
        print(*[str(a) for a in args])


def _format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} {unit}"
    return f"{amount:.2f} {unit}"


def _header(title: str) -> None:
    if _RICH_AVAILABLE and _console:
        _console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")


def _kv(key: str, value: Any) -> None:
    if _RICH_AVAILABLE:
        _print(f"  [bold]{key}:[/bold] {value}")
    else:
        print(f"  {key}: {value}")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thintensor",
        description=(
            "Convert, run, benchmark, and validate .thin models from one CLI"
        ),
        epilog=(
            "Start: thintensor doctor | thintensor profiles list | "
            "thintensor explain --goal balanced"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_convert = sub.add_parser(
        "convert",
        help="Convert a Hugging Face directory to a verified .thin archive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_convert.add_argument("hf_dir", type=str, help="Path to HF model directory")
    p_convert.add_argument("--out", type=str, required=True, help="Output .thin path")
    p_convert.add_argument("--arch", type=str, help="Override architecture")
    p_convert.add_argument(
        "--include-tokenizer-hashes",
        action="store_true",
        default=True,
        help="Include tokenizer hashes (default: true)",
    )
    p_convert.add_argument(
        "--no-tokenizer-hashes",
        action="store_true",
        help="Exclude tokenizer hashes",
    )
    p_convert.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output"
    )
    p_convert.add_argument(
        "--no-verify", action="store_true", help="Skip post-conversion verification"
    )

    p_pull = sub.add_parser(
        "pull",
        help="Download a Hugging Face model into the ThinTensor cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_pull.add_argument("model_id", type=str, help="HF model id (e.g. Qwen/Qwen3-0.6B)")
    p_pull.add_argument("--dir", type=str, help="Target directory")
    p_pull.add_argument("--revision", type=str, default="main", help="Git revision")
    p_pull.add_argument("--token", type=str, help="HF API token")

    p_run = sub.add_parser(
        "run",
        help="Generate text with greedy causal decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_run.add_argument("model", type=str, help=".thin archive, HF dir, or HF model id")
    p_run.add_argument("--prompt", type=str, default="Hello", help="Input prompt")
    p_run.add_argument("--chat", action="store_true", help="Start in chat mode")
    p_run.add_argument("--context", type=int, default=512, help="Context length")
    p_run.add_argument(
        "--max-new-tokens", "--steps", dest="max_new_tokens", type=int,
        default=200, help="Maximum generated tokens",
    )
    p_run.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature; only 0 (greedy) is currently supported",
    )
    p_run.add_argument("--top-p", type=float, default=1.0, help="Must be 1 for greedy decode")
    p_run.add_argument("--top-k", type=int, default=0, help="Must be 0 for greedy decode")
    p_run.add_argument("--seed", type=int, default=0, help="Random seed")
    p_run.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p_run.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p_run.add_argument(
        "--engine",
        choices=["auto", "native", "transformers"],
        default="auto",
        help=(
            "Execution engine; auto uses native ThinTensor when semantic and "
            "tensor capabilities match, otherwise Transformers"
        ),
    )
    p_run.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom Hugging Face model code in Transformers fallback",
    )
    p_run.add_argument(
        "--profile", default="auto",
        help="Runtime profile (auto/safe/balanced/max-performance/lab)",
    )
    p_run.add_argument(
        "--force-profile", action="store_true",
        help="Bypass semantic capability checks for an explicit experiment",
    )
    p_run.add_argument(
        "--residency", choices=["all", "stream"], default="all",
        help="Weight residency policy",
    )
    p_run.add_argument(
        "--gpu-weight-budget", default="0",
        help="Streaming GPU weight budget such as 6GiB; required for bounded residency",
    )
    p_run.add_argument("--prefetch-layers", type=int, default=0)
    p_run.add_argument("--cpu-offload", action="store_true")
    p_run.add_argument("--pin-cpu-pages", action="store_true")
    p_run.add_argument(
        "--kv-residency", choices=["gpu_full", "hybrid_recent", "cpu_exact"],
        default="gpu_full", help="Exact KV cache residency",
    )
    p_run.add_argument("--kv-gpu-recent-tokens", type=int, default=256)
    p_run.add_argument("--tokenizer", help="Tokenizer directory or Hugging Face model id")
    p_run.add_argument(
        "--hf-source",
        help=(
            "Original HF directory/config for running a non-native .thin "
            "archive through Transformers"
        ),
    )
    p_run.add_argument("--head8", action="store_true", help="[experimental] head8 mode")
    p_run.add_argument(
        "--o-proj-fp8", action="store_true", help="[experimental] FP8 O-projection"
    )
    p_run.add_argument(
        "--fused-scaled-mlp", action="store_true", help="[experimental] Fused scaled MLP"
    )
    p_run.add_argument(
        "--fused-residual-norm",
        action="store_true",
        help="[experimental] Fused residual+norm",
    )
    p_run.add_argument(
        "--allow-experimental",
        action="store_true",
        help="Allow experimental flags",
    )
    p_run.add_argument("--json", action="store_true", help="JSON output")

    p_chat = sub.add_parser(
        "chat",
        help="Start an interactive greedy-decoding session",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_chat.add_argument("model", type=str, help=".thin archive, HF dir, or HF model id")
    p_chat.add_argument("--context", type=int, default=2048, help="Context length")
    p_chat.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    p_chat.add_argument("--temperature", type=float, default=0.0)
    p_chat.add_argument("--top-p", type=float, default=1.0)
    p_chat.add_argument("--top-k", type=int, default=0)
    p_chat.add_argument("--seed", type=int, default=0)
    p_chat.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p_chat.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p_chat.add_argument(
        "--engine",
        choices=["auto", "native", "transformers"],
        default="auto",
    )
    p_chat.add_argument("--trust-remote-code", action="store_true")
    p_chat.add_argument("--profile", default="auto")
    p_chat.add_argument("--force-profile", action="store_true")
    p_chat.add_argument("--tokenizer")
    p_chat.add_argument("--hf-source")
    p_chat.add_argument("--allow-experimental", action="store_true")

    p_bench = sub.add_parser(
        "bench",
        help="Benchmark real causal-KV decode in isolated processes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_bench.add_argument("model", type=str, help=".thin archive path")
    p_bench.add_argument(
        "--profiles", default="safe,balanced",
        help="Comma-separated profiles to benchmark",
    )
    p_bench.add_argument("--steps", type=int, default=200, help="Steps per profile")
    p_bench.add_argument("--warmup", type=int, default=10, help="Warmup decode steps")
    p_bench.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p_bench.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p_bench.add_argument("--force-profile", action="store_true")
    p_bench.add_argument("--max-gpu-temp", type=int, default=87)
    p_bench.add_argument("--json", action="store_true")
    p_bench.add_argument("--out", type=str, help="Write JSON results to this file or directory")
    p_bench.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p_bench.add_argument(
        "--hf-model",
        help="Also benchmark this original HF model/directory as the baseline",
    )
    p_bench.add_argument(
        "--require-faster-than-hf",
        action="store_true",
        help="Exit nonzero unless at least one ThinTensor profile beats HF",
    )
    p_bench.add_argument("--trust-remote-code", action="store_true")

    p_validate = sub.add_parser(
        "validate",
        help="Compare ThinTensor logits with a Hugging Face reference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_validate.add_argument("archive", help=".thin archive")
    p_validate.add_argument("--hf-model", required=True, help="HF model id or local directory")
    p_validate.add_argument("--profile", default="auto")
    p_validate.add_argument("--force-profile", action="store_true")
    p_validate.add_argument("--suite", choices=["quick", "required"], default="quick")
    p_validate.add_argument(
        "--require-tier",
        choices=["ranking", "exact"],
        default="ranking",
        help="Minimum accepted correctness tier",
    )
    p_validate.add_argument("--prompt", default="Hello")
    p_validate.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p_validate.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p_validate.add_argument(
        "--out",
        default="correctness_results/cli/latest.md",
        help="Markdown report path; a JSON sibling is written too",
    )
    p_validate.add_argument("--json", action="store_true")
    p_validate.add_argument("--trust-remote-code", action="store_true")
    p_validate.add_argument("--dry-run", action="store_true")

    p_optimize = sub.add_parser(
        "optimize",
        help="Run a correctness-gated one-candidate-at-a-time experiment plan",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_optimize.add_argument("plan", type=Path, nargs="?", help="JSON or YAML optimizer plan")
    p_optimize.add_argument(
        "--init", type=Path, metavar="PATH",
        help="Write a documented starter JSON plan instead of running",
    )
    p_optimize.add_argument("--archive", help="Archive path for --init")
    p_optimize.add_argument("--hf-model", help="HF model ID or directory for --init")
    p_optimize.add_argument("--benchmark-dir", type=Path)
    p_optimize.add_argument("--correctness-dir", type=Path)
    p_optimize.add_argument("--report", type=Path)
    p_optimize.add_argument("--resume", action="store_true")
    p_optimize.add_argument("--dry-run", action="store_true")

    p_profiles = sub.add_parser("profiles", help="List or inspect runtime profiles")
    profile_sub = p_profiles.add_subparsers(dest="profile_command", metavar="ACTION")
    p_profiles_list = profile_sub.add_parser("list", help="List profiles")
    p_profiles_list.add_argument("--json", action="store_true")
    p_profiles_show = profile_sub.add_parser("show", help="Show one profile")
    p_profiles_show.add_argument("name")
    p_profiles_show.add_argument(
        "--archive",
        help="Resolve auto and validate model-specific profiles for this archive",
    )
    p_profiles_show.add_argument("--json", action="store_true")

    p_explain = sub.add_parser(
        "explain",
        help="Explain profile quality, retention, speed, and tradeoffs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_explain.add_argument(
        "profile",
        nargs="?",
        help="Profile to explain; omit to compare all profiles",
    )
    p_explain.add_argument(
        "--goal",
        choices=["quality", "balanced", "speed"],
        help="Recommend a profile for this priority",
    )
    p_explain.add_argument(
        "--model",
        "--archive",
        dest="model",
        help="Analyze an HF directory or .thin archive",
    )
    p_explain.add_argument("--json", action="store_true")

    p_architectures = sub.add_parser(
        "architectures",
        help="Show native architecture correctness/performance coverage",
    )
    architecture_sub = p_architectures.add_subparsers(
        dest="architecture_command",
        metavar="ACTION",
    )
    p_arch_list = architecture_sub.add_parser("list", help="List coverage")
    p_arch_list.add_argument("--json", action="store_true")
    p_arch_show = architecture_sub.add_parser("show", help="Show one family")
    p_arch_show.add_argument("name")
    p_arch_show.add_argument("--json", action="store_true")
    p_arch_audit = architecture_sub.add_parser(
        "audit",
        help="Audit an HF directory or .thin archive",
    )
    p_arch_audit.add_argument("model")
    p_arch_audit.add_argument("--json", action="store_true")

    p_inspect = sub.add_parser("inspect", help="Inspect a .thin archive")
    p_inspect.add_argument("archive", type=str, help=".thin archive path")
    p_inspect.add_argument("--verify", action="store_true", help="Verify archive hashes first")
    p_inspect.add_argument("--json", action="store_true")

    p_doctor = sub.add_parser("doctor", help="Check environment and dependencies")
    p_doctor.add_argument("--json", action="store_true")
    p_doctor.add_argument(
        "--strict", action="store_true",
        help="Exit nonzero if a required runtime component is unavailable",
    )

    p_cache = sub.add_parser("cache", help="Inspect the local model/archive cache")
    cache_sub = p_cache.add_subparsers(dest="cache_command", metavar="ACTION")
    p_cache_list = cache_sub.add_parser("list", help="List cached models and archives")
    p_cache_list.add_argument("--json", action="store_true")
    cache_sub.add_parser("path", help="Print the cache root")

    p_core = sub.add_parser(
        "core",
        help="Run a low-level Rust archive command",
        add_help=False,
    )
    p_core.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for thintensor")

    return parser


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_version() -> None:
    from . import __version__

    _print(
        f"[bold cyan]ThinTensor[/bold cyan] v{__version__}"
        if _RICH_AVAILABLE
        else f"ThinTensor v{__version__}"
    )


def cmd_convert(args: argparse.Namespace) -> None:
    from .command_runner import convert_hf_model, find_rust_binary
    from .model_cache import format_bytes

    hf_dir = Path(args.hf_dir).resolve()
    out_path = Path(args.out).resolve()

    if not hf_dir.exists():
        _print(f"\u2717 HF directory not found: {hf_dir}")
        raise SystemExit(1)

    if out_path.exists() and not args.overwrite:
        _print(f"\u2717 Output file exists: {out_path}")
        _print("  Use --overwrite to replace it.")
        raise SystemExit(1)

    if not find_rust_binary():
        _print("\u2717 Rust binary 'thintensor' not found.")
        _print("  Build with: cargo build --release")
        raise SystemExit(1)

    _header("ThinTensor Convert")
    _kv("Input", str(hf_dir))
    _kv("Output", str(out_path))
    if args.arch:
        _kv("Architecture", args.arch)
    from .capabilities import analyze_hf_directory

    support = analyze_hf_directory(hf_dir)
    _kv(
        "Runtime route",
        "native ThinTensor" if support.supported else "Transformers compatibility",
    )
    if support.reasons:
        _kv("Native blockers", "; ".join(support.reasons))

    print()
    include_tokenizer = args.include_tokenizer_hashes and not args.no_tokenizer_hashes
    verify = not args.no_verify

    _print("Converting..." if not _RICH_AVAILABLE else "[yellow]Converting...[/yellow]")
    t0 = time.perf_counter()

    success = convert_hf_model(
        hf_dir,
        out_path,
        arch=args.arch,
        include_tokenizer_hashes=include_tokenizer,
        verify=verify,
    )
    elapsed = time.perf_counter() - t0

    if not success:
        _print("\u2717 Conversion failed!")
        raise SystemExit(1)

    # Print summary
    print()
    _header("Conversion Summary")

    # Try to read archive info
    try:
        from .archive import ThinArchive

        archive = ThinArchive(out_path)
        manifest = archive.manifest
        model = manifest.get("model", {})
        pages = manifest.get("pages", [])

        _kv("Model", model.get("arch", "unknown"))
        _kv("Layers", model.get("layers", "?"))
        _kv("Hidden size", model.get("hidden_size", "?"))
        _kv("Heads / KV heads", f"{model.get('heads', '?')} / {model.get('kv_heads', '?')}")

        # Determine attention kind
        heads = model.get("heads", 0)
        kv_heads = model.get("kv_heads", 0)
        if heads and kv_heads:
            if kv_heads == 1:
                attn = "MQA"
            elif kv_heads == heads:
                attn = "MHA"
            else:
                attn = "GQA"
            _kv("Attention", attn)

        _kv("Pages", len(pages))
        _kv("Archive size", format_bytes(out_path.stat().st_size))
        _kv("Verification", "\u2713 PASS" if verify else "skipped")
        _kv("Time", f"{elapsed:.2f}s")
        _kv("Output", str(out_path))

        archive.close()
    except Exception:
        _kv("Output", str(out_path))
        _kv("Size", format_bytes(out_path.stat().st_size))
        _kv("Time", f"{elapsed:.2f}s")


def cmd_pull(args: argparse.Namespace) -> None:
    from .hf_pull import pull_model

    _header("ThinTensor Pull")
    target_dir = args.dir if args.dir else None
    path = pull_model(
        args.model_id,
        target_dir=target_dir,
        revision=args.revision,
        token=args.token,
    )
    from .capabilities import analyze_hf_directory

    support = analyze_hf_directory(path)
    print()
    _kv("Engine", support.engine)
    if support.reasons:
        _kv("Native blockers", "; ".join(support.reasons))
    _kv(
        "Next",
        f"thintensor run {args.model_id} --prompt \"Hello\"",
    )


def cmd_run(args: argparse.Namespace) -> None:
    from .model_cache import resolve_input
    from .profile_presets import (
        get_profile,
        apply_overrides,
        EXPERIMENTAL_WARNING,
        CPU_KV_WARNING,
    )

    if args.chat:
        if args.json:
            raise SystemExit("--json is not available for interactive chat")
        cmd_chat(args)
        return

    resolved = resolve_input(args.model)

    if "error" in resolved:
        _print(f"\u2717 {resolved['error']}")
        raise SystemExit(1)
    _validate_generation_args(args)
    _require_greedy_sampling(args)

    if resolved["kind"] != "archive":
        hf_source = _ensure_hf_source(resolved, quiet=args.json)
        from .capabilities import analyze_hf_directory

        support = analyze_hf_directory(hf_source)
        use_transformers = (
            args.engine == "transformers"
            or (args.engine == "auto" and not support.supported)
        )
        if use_transformers:
            if not args.json:
                _header("ThinTensor Run")
                _kv("Engine", "Transformers compatibility fallback")
                _kv("Model", hf_source)
                if support.reasons:
                    _kv("Native engine skipped", "; ".join(support.reasons))
                print()
            _run_transformers_inference(
                model_source=hf_source,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                context=args.context,
                device=args.device,
                dtype=args.dtype,
                tokenizer_source=args.tokenizer,
                trust_remote_code=args.trust_remote_code,
                json_output=args.json,
            )
            return
        if args.engine == "native" and not support.supported:
            raise SystemExit(
                "model is not supported by the native engine: "
                + "; ".join(support.reasons)
            )
    else:
        archive_model = _archive_model(resolved["path"])
        from .capabilities import analyze_archive_model

        support = analyze_archive_model(archive_model)
        use_transformers = (
            args.engine == "transformers"
            or (args.engine == "auto" and not support.supported)
        )
        if use_transformers:
            if not args.hf_source:
                raise SystemExit(
                    "this archive requires the Transformers compatibility "
                    "engine; pass --hf-source ORIGINAL_HF_DIRECTORY"
                )
            _run_transformers_inference(
                model_source=args.hf_source,
                archive_path=resolved["path"],
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                context=args.context,
                device=args.device,
                dtype=args.dtype,
                tokenizer_source=args.tokenizer,
                trust_remote_code=args.trust_remote_code,
                json_output=args.json,
            )
            return
        if args.engine == "native" and not support.supported:
            raise SystemExit(
                "archive is not supported by the native engine: "
                + "; ".join(support.reasons)
            )

    archive_path = _ensure_archive(resolved, args)
    model = _archive_model(archive_path)

    try:
        profile = get_profile(
            args.profile,
            model=model,
            force=args.force_profile,
        )
        profile = apply_overrides(
            profile,
            head8=getattr(args, "head8", False),
            o_proj_fp8=getattr(args, "o_proj_fp8", False),
            fused_scaled_mlp=getattr(args, "fused_scaled_mlp", False),
            fused_residual_norm=getattr(args, "fused_residual_norm", False),
            allow_experimental=getattr(args, "allow_experimental", False),
        )
        profile = _adapt_profile_for_device(
            profile,
            requested_name=args.profile,
            model=model,
            device=args.device,
        )
    except ValueError as e:
        _print(f"\u2717 {e}")
        raise SystemExit(1)

    if profile.get("experimental") and not args.json:
        _print(EXPERIMENTAL_WARNING if not _RICH_AVAILABLE else f"[yellow]{EXPERIMENTAL_WARNING}[/yellow]")
        print()

    if args.kv_residency in ("cpu_exact", "hybrid_recent") and not args.json:
        _print(CPU_KV_WARNING if not _RICH_AVAILABLE else f"[blue]{CPU_KV_WARNING}[/blue]")
        print()

    if not args.json:
        _header("ThinTensor Run")
        _kv("Archive", archive_path)
        _kv("Profile", profile["name"])
        _kv("Backend", profile["kernel_backend"])
        _kv("Device", args.device)
        _kv("Weight residency", args.residency)
        _kv("Context", args.context)
        _kv("Max new tokens", args.max_new_tokens)
        print()

    _run_inference(
        archive_path=archive_path,
        prompt=args.prompt,
        profile=profile,
        max_new_tokens=args.max_new_tokens,
        context=args.context,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        kv_residency=args.kv_residency,
        kv_gpu_recent_tokens=args.kv_gpu_recent_tokens,
        weight_residency=args.residency,
        gpu_weight_budget=args.gpu_weight_budget,
        prefetch_layers=args.prefetch_layers,
        cpu_offload=args.cpu_offload,
        pin_cpu_pages=args.pin_cpu_pages,
        tokenizer_source=args.tokenizer,
        json_output=args.json,
    )


def cmd_chat(args: argparse.Namespace) -> None:
    from .model_cache import resolve_input
    from .profile_presets import (
        get_profile,
        apply_overrides,
        EXPERIMENTAL_WARNING,
    )

    resolved = resolve_input(args.model)
    if "error" in resolved:
        _print(f"\u2717 {resolved['error']}")
        raise SystemExit(1)
    _validate_generation_args(args)
    _require_greedy_sampling(args)

    if resolved["kind"] != "archive":
        hf_source = _ensure_hf_source(resolved, quiet=False)
        from .capabilities import analyze_hf_directory

        support = analyze_hf_directory(hf_source)
        if args.engine == "transformers" or (
            args.engine == "auto" and not support.supported
        ):
            _run_transformers_chat(
                model_source=hf_source,
                max_new_tokens=args.max_new_tokens,
                context=args.context,
                device=args.device,
                dtype=args.dtype,
                tokenizer_source=args.tokenizer,
                trust_remote_code=args.trust_remote_code,
            )
            return
        if args.engine == "native" and not support.supported:
            raise SystemExit(
                "model is not supported by the native engine: "
                + "; ".join(support.reasons)
            )
    else:
        archive_model = _archive_model(resolved["path"])
        from .capabilities import analyze_archive_model

        support = analyze_archive_model(archive_model)
        if args.engine == "transformers" or (
            args.engine == "auto" and not support.supported
        ):
            if not args.hf_source:
                raise SystemExit(
                    "this archive requires the Transformers compatibility "
                    "engine; pass --hf-source ORIGINAL_HF_DIRECTORY"
                )
            _run_transformers_chat(
                model_source=args.hf_source,
                archive_path=resolved["path"],
                max_new_tokens=args.max_new_tokens,
                context=args.context,
                device=args.device,
                dtype=args.dtype,
                tokenizer_source=args.tokenizer,
                trust_remote_code=args.trust_remote_code,
            )
            return
        if args.engine == "native" and not support.supported:
            raise SystemExit(
                "archive is not supported by the native engine: "
                + "; ".join(support.reasons)
            )
    archive_path = _ensure_archive(resolved, args)
    model = _archive_model(archive_path)

    try:
        profile = get_profile(
            args.profile,
            model=model,
            force=args.force_profile,
        )
        profile = _adapt_profile_for_device(
            profile,
            requested_name=args.profile,
            model=model,
            device=args.device,
        )
        profile = apply_overrides(
            profile,
            allow_experimental=getattr(args, "allow_experimental", False),
        )
    except ValueError as e:
        _print(f"\u2717 {e}")
        raise SystemExit(1)

    if profile.get("experimental"):
        _print(EXPERIMENTAL_WARNING if not _RICH_AVAILABLE else f"[yellow]{EXPERIMENTAL_WARNING}[/yellow]")
        print()

    _header("ThinTensor Chat")
    _kv("Archive", archive_path)
    _kv("Profile", args.profile)
    _kv("Context", args.context)
    print()

    _run_chat_loop(
        archive_path=archive_path,
        profile=profile,
        max_new_tokens=args.max_new_tokens,
        context=args.context,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        tokenizer_source=args.tokenizer,
    )


def cmd_bench(args: argparse.Namespace) -> None:
    from .model_cache import resolve_input
    from .profile_presets import get_profile

    resolved = resolve_input(args.model)
    if "error" in resolved:
        _print(f"\u2717 {resolved['error']}")
        raise SystemExit(1)

    archive_path = _ensure_archive(resolved, args)
    model = _archive_model(archive_path)
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    if not profiles:
        raise SystemExit("--profiles must contain at least one profile")
    if args.steps < 1 or args.warmup < 0:
        raise SystemExit("--steps must be positive and --warmup non-negative")

    if not args.json:
        _header("ThinTensor Causal Decode Benchmark")
        _kv("Archive", archive_path)
        _kv("Profiles", ", ".join(profiles))
        _kv("Steps", args.steps)
        _kv("Warmup", args.warmup)
        _kv("Device", args.device)
        print()

    results = []
    for profile_name in profiles:
        try:
            profile = get_profile(
                profile_name,
                model=model,
                force=args.force_profile,
            )
            profile = _adapt_profile_for_device(
                profile,
                requested_name=profile_name,
                model=model,
                device=args.device,
            )
        except ValueError as e:
            if args.json:
                results.append({"profile": profile_name, "error": str(e)})
                continue
            raise SystemExit(str(e)) from e

        if not args.json:
            _print(
                f"Running profile: {profile['name']}..."
                if not _RICH_AVAILABLE
                else f"[cyan]Running profile: {profile['name']}...[/cyan]"
            )

        result = _run_benchmark_profile(
            archive_path=archive_path,
            profile=profile,
            profile_name=profile["name"],
            steps=args.steps,
            warmup=args.warmup,
            device=args.device,
            dtype=args.dtype,
            max_gpu_temp=args.max_gpu_temp,
            dry_run=args.dry_run,
            quiet=args.json,
        )
        if result:
            results.append(result)
        if not args.json:
            print()

    if results:
        if args.hf_model:
            hf_result = _run_transformers_benchmark(
                model=args.hf_model,
                steps=args.steps,
                warmup=args.warmup,
                device=args.device,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                dry_run=args.dry_run,
                quiet=args.json,
            )
            if hf_result:
                results.append(hf_result)
                hf_speed = hf_result.get("tokens_per_s")
                if isinstance(hf_speed, (int, float)) and hf_speed > 0:
                    for row in results:
                        speed = row.get("tokens_per_s")
                        if row.get("engine") == "thintensor" and isinstance(
                            speed, (int, float)
                        ):
                            row["speedup_vs_transformers"] = speed / hf_speed
        _print_bench_table(results, json_output=args.json, out_dir=args.out)
    if args.require_faster_than_hf and not args.dry_run:
        if not args.hf_model:
            raise SystemExit("--require-faster-than-hf requires --hf-model")
        wins = [
            row
            for row in results
            if row.get("engine") == "thintensor"
            and float(row.get("speedup_vs_transformers") or 0) > 1.0
        ]
        if not wins:
            raise SystemExit(
                "no ThinTensor profile beat the Transformers baseline"
            )
    if any("error" in row for row in results):
        raise SystemExit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    from .profile_presets import (
        get_profile,
        profile_to_runtime_flags,
        subprocess_environment,
    )

    archive = Path(args.archive).resolve()
    if not archive.is_file():
        raise SystemExit(f"archive not found: {archive}")
    model = _archive_model(str(archive))
    profile = get_profile(
        args.profile,
        model=model,
        force=args.force_profile,
    )
    profile = _adapt_profile_for_device(
        profile,
        requested_name=args.profile,
        model=model,
        device=args.device,
    )
    prefill_lens = "1,128" if args.suite == "quick" else "1,8,32,128"
    steps = "1,10" if args.suite == "quick" else "1,10,50"
    runtime_flags = profile_to_runtime_flags(profile)
    runtime_flags = [
        flag for flag in runtime_flags if flag != "--keep-bf16-lm-head"
    ]
    report_path = Path(args.out)
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(_tool_script("compare_hf_thin_logits.py")),
        "--hf-model",
        args.hf_model,
        "--archive",
        str(archive),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--prompt",
        args.prompt,
        "--prefill-lens",
        prefill_lens,
        "--steps",
        steps,
        "--attention-mode",
        "causal_kv",
        "--out",
        str(report_path),
        "--json",
        *runtime_flags,
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.dry_run:
        _emit_command(command, json_output=args.json)
        return
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=subprocess_environment(profile),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise SystemExit(completed.returncode)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"validator returned invalid JSON: {exc}"
        ) from exc
    result["profile"] = profile["name"]
    tier_rank = {"fail": 0, "experimental": 1, "ranking_pass": 2, "exact_pass": 3}
    required_rank = 3 if args.require_tier == "exact" else 2
    accepted = tier_rank.get(result.get("correctness_tier"), 0) >= required_rank
    result["required_tier"] = args.require_tier
    result["accepted"] = accepted
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _header("Correctness Validation")
        _kv("Profile", profile["name"])
        _kv("Tier", result.get("correctness_tier", "unknown"))
        _kv("Attention equivalent", result.get("attention_equivalent"))
        _kv("Ranking equivalent", result.get("ranking_equivalent"))
        _kv("HF equivalent", result.get("hf_equivalent"))
        _kv("Required tier", args.require_tier)
        _kv("Accepted", accepted)
        _kv("Report", str(report_path))
    if not accepted:
        raise SystemExit(2)


def cmd_optimize(args: argparse.Namespace) -> None:
    if args.init:
        if not args.archive or not args.hf_model:
            raise SystemExit("--init requires --archive and --hf-model")
        destination = args.init.resolve()
        if destination.exists():
            raise SystemExit(f"refusing to overwrite existing plan: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        template = _optimizer_plan_template(args.archive, args.hf_model)
        destination.write_text(
            json.dumps(template, indent=2) + "\n",
            encoding="utf-8",
        )
        _print(f"Wrote optimizer plan: {destination}")
        return
    if args.plan is None:
        raise SystemExit("provide PLAN or use --init PATH --archive ... --hf-model ...")
    plan = args.plan.resolve()
    if not plan.is_file():
        raise SystemExit(f"optimizer plan not found: {plan}")
    command = [
        sys.executable,
        str(_tool_script("optimize_runtime_one_by_one.py")),
        str(plan),
    ]
    benchmark_dir = args.benchmark_dir or (
        Path.cwd() / "benchmark_results" / "runtime_optimizer"
    )
    correctness_dir = args.correctness_dir or (
        Path.cwd() / "correctness_results" / "runtime_optimizer"
    )
    report = args.report or (Path.cwd() / "runtime_optimization_report.md")
    command.extend(["--benchmark-dir", str(benchmark_dir)])
    command.extend(["--correctness-dir", str(correctness_dir)])
    command.extend(["--report", str(report)])
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        _emit_command(command, json_output=False)
        return
    raise SystemExit(subprocess.run(command, cwd=Path.cwd(), check=False).returncode)


def cmd_profiles(args: argparse.Namespace) -> None:
    from .profile_presets import (
        PROFILES,
        PROFILE_ALIASES,
        get_profile,
        profile_public_summary,
    )

    action = args.profile_command or "list"
    json_output = getattr(args, "json", False)
    if action == "show":
        model = _archive_model(args.archive) if args.archive else None
        if args.name == "auto" and model is None:
            raise SystemExit(
                "'auto' depends on model capabilities; pass --archive MODEL.thin"
            )
        try:
            payload = get_profile(args.name, model=model)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _header(payload["label"])
            _kv("Name", payload["name"])
            _kv("Description", payload["description"])
            _kv("Experimental", payload["experimental"])
            _kv(
                "Required capabilities",
                payload.get("required_capabilities") or "portable fallback",
            )
            _kv("Quality", payload["quality_contract"])
            _kv("Retention", payload["retention_contract"])
            _kv("Performance", payload["speed_contract"])
            print()
            for key, value in payload.items():
                if key not in {
                    "name", "label", "description", "experimental",
                    "required_capabilities", "intent", "quality_contract",
                    "retention_contract", "speed_contract", "recommended_for",
                    "tradeoffs", "measured_results",
                }:
                    _kv(key.replace("_", " "), value)
        return

    payload = [
        profile_public_summary({**profile, "name": name})
        for name, profile in PROFILES.items()
        if not profile.get("hidden")
    ]
    for alias, target in PROFILE_ALIASES.items():
        payload.append({"name": alias, "alias_for": target})
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _header("Runtime Profiles")
        _kv("auto", "select a native kernel family from model semantics")
        for row in payload:
            if "alias_for" in row:
                _kv(row["name"], f"alias for {row['alias_for']}")
            else:
                marker = " [experimental]" if row["experimental"] else ""
                _kv(row["name"], row["label"] + marker)


def cmd_explain(args: argparse.Namespace) -> None:
    from .profile_presets import (
        PROFILES,
        get_profile,
        profile_compatibility,
        profile_public_summary,
        recommend_profile,
    )

    model = None
    native_support = None
    if args.model:
        candidate = Path(args.model)
        if candidate.suffix == ".thin":
            model = _archive_model(args.model)
            from .capabilities import analyze_archive_model

            native_support = analyze_archive_model(model)
        else:
            from .capabilities import analyze_hf_directory

            native_support = analyze_hf_directory(candidate)
            config_path = candidate / "config.json"
            if config_path.is_file():
                model = json.loads(config_path.read_text(encoding="utf-8"))
    if args.goal:
        profile = recommend_profile(args.goal, model=model)
        payload = profile_public_summary(profile)
        payload["recommendation_goal"] = args.goal
        payload["recommendation_reason"] = profile["recommended_for"]
        if native_support is not None:
            payload["engine_decision"] = native_support.as_dict()
        rows = [payload]
    elif args.profile:
        try:
            profile = get_profile(args.profile, model=model)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        payload = profile_public_summary(profile)
        if native_support is not None:
            payload["engine_decision"] = native_support.as_dict()
        rows = [payload]
    else:
        rows = []
        for name, definition in PROFILES.items():
            if definition.get("hidden"):
                continue
            profile = {**definition, "name": name}
            payload = profile_public_summary(profile)
            if model is not None:
                compatible, reason = profile_compatibility(profile, model)
                payload["compatible"] = compatible
                payload["compatibility_reason"] = reason
            rows.append(payload)

    if args.json:
        print(json.dumps(rows[0] if len(rows) == 1 else rows, indent=2))
        return

    if len(rows) == 1:
        row = rows[0]
        _header(row["label"])
        _kv("Profile", row["name"])
        _kv("Purpose", row["intent"])
        _kv("Quality", row["quality_contract"])
        _kv("Context retention", row["retention_contract"])
        _kv("Performance", row["speed_contract"])
        _kv("Use it for", row["recommended_for"])
        _kv("Status", "experimental opt-in" if row["experimental"] else "supported")
        engine_decision = row.get("engine_decision")
        if engine_decision:
            _kv("Engine", engine_decision["engine"])
            if engine_decision["reasons"]:
                _kv("Native blockers", "; ".join(engine_decision["reasons"]))
        tradeoffs = row.get("tradeoffs") or ()
        if tradeoffs:
            print("\nTradeoffs:")
            for item in tradeoffs:
                print(f"  - {item}")
        print(f"\nRun: thintensor run MODEL.thin --profile {row['name']}")
        return

    _header("Profile Decision Guide")
    print("  safe       -> portable BF16 and highest fidelity")
    print("  balanced   -> native kernels without approximate weight storage")
    print("  max-performance -> fastest validated single-stream model profile")
    print()
    if _RICH_AVAILABLE and _console:
        table = Table(show_lines=True)
        table.add_column("Profile", style="bold cyan")
        table.add_column("Status")
        table.add_column("Purpose")
        table.add_column("Performance contract")
        for row in rows:
            table.add_row(
                row["name"],
                "experimental" if row["experimental"] else "supported",
                row["intent"],
                row["speed_contract"],
            )
        _console.print(table)
    else:
        for row in rows:
            status = "experimental" if row["experimental"] else "supported"
            print(f"  {row['name']:<18} {status:<12} {row['intent']}")
    print("\nInspect one: thintensor explain PROFILE")


def cmd_architectures(args: argparse.Namespace) -> None:
    from .architectures import (
        architecture_rows,
        architecture_status,
    )

    action = args.architecture_command or "list"
    if action == "show":
        entry = architecture_status(args.name)
        if entry is None:
            raise SystemExit(f"architecture is not registered: {args.name}")
        payload = entry.as_dict()
    elif action == "audit":
        candidate = Path(args.model)
        if candidate.suffix == ".thin":
            model = _archive_model(args.model)
            from .capabilities import analyze_archive_model

            support = analyze_archive_model(model)
            raw_name = str(
                model.get("model_type")
                or model.get("raw_arch")
                or model.get("arch")
                or "unknown"
            )
        else:
            from .capabilities import analyze_hf_directory

            support = analyze_hf_directory(candidate)
            raw_name = support.model_type or support.architecture
        entry = architecture_status(raw_name)
        payload = {
            "model": str(candidate),
            "detected": support.as_dict(),
            "registry": entry.as_dict() if entry else None,
            "performance_supported": bool(
                entry is not None and entry.native_status == "verified"
            ),
            "next_gate": (
                None
                if entry is not None and entry.native_status == "verified"
                else "run matched conversion, correctness, and HF speed gates"
            ),
        }
    else:
        payload = architecture_rows()

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if action == "list":
        _header("Architecture Coverage")
        if _RICH_AVAILABLE and _console:
            table = Table(show_lines=True)
            table.add_column("Architecture", style="bold cyan")
            table.add_column("Native status")
            table.add_column("Schema")
            table.add_column("Validated profile")
            table.add_column("vs HF")
            for row in payload:
                speedup = row.get("speedup_vs_hf")
                table.add_row(
                    row["name"],
                    row["native_status"],
                    row["tensor_schema"],
                    row.get("validated_profile") or "-",
                    f"{speedup:.3f}x" if speedup is not None else "-",
                )
            _console.print(table)
        else:
            for row in payload:
                print(
                    f"{row['name']:<14} {row['native_status']:<10} "
                    f"{row['tensor_schema']}"
                )
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_inspect(args: argparse.Namespace) -> None:
    archive_path = Path(args.archive)
    if not archive_path.exists():
        _print(f"\u2717 Archive not found: {archive_path}")
        raise SystemExit(1)

    from .command_runner import (
        inspect_archive_json,
        find_rust_binary,
        verify_archive,
    )

    verified: bool | None = None
    if args.verify:
        if not find_rust_binary():
            raise SystemExit(
                "archive verification requires the Rust binary; "
                "run cargo build --release"
            )
        verified = verify_archive(archive_path)
        if not verified:
            if args.json:
                print(json.dumps({"archive": str(archive_path), "verified": False}))
            raise SystemExit("archive verification failed")

    if find_rust_binary():
        stats = inspect_archive_json(archive_path)
        if stats and args.json:
            if verified is not None:
                stats["verified"] = verified
            print(json.dumps(stats, indent=2))
            return
        if stats:
            _print_inspect_from_stats(stats, archive_path)
            if verified is not None:
                _kv("Verification", "PASS")
            return

    # Fallback: Python-only inspect
    try:
        from .archive import ThinArchive

        archive = ThinArchive(archive_path)
        if args.json:
            print(json.dumps(archive.manifest, indent=2))
        else:
            _print_inspect_from_archive(archive, archive_path)
        archive.close()
    except Exception as e:
        _print(f"\u2717 Failed to inspect archive: {e}")
        raise SystemExit(1)


def cmd_doctor(args: argparse.Namespace) -> None:
    if not args.json:
        _header("ThinTensor Doctor")
        print()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, *, required: bool) -> None:
        checks.append(
            {
                "component": name,
                "ok": bool(ok),
                "required": required,
                "detail": detail,
            }
        )

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    add("Python", sys.version_info >= (3, 10), py_ver, required=True)

    try:
        import torch

        torch_ver = torch.__version__
        add("PyTorch", True, torch_ver, required=True)
    except ImportError:
        add("PyTorch", False, "not installed (pip install 'thintensor[runtime]')", required=True)

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory
            add("CUDA", True, "available", required=True)
            add("GPU", True, gpu_name, required=True)
            add("VRAM", True, _format_bytes(vram), required=True)
        else:
            add("CUDA", False, "not available", required=True)
    except Exception:
        add("CUDA", False, "error checking CUDA", required=True)

    try:
        import triton  # noqa: F401

        add("Triton", True, "installed", required=True)
    except ImportError:
        add("Triton", False, "not installed (pip install triton)", required=True)

    from .command_runner import find_rust_binary

    rust_bin = find_rust_binary()
    if rust_bin:
        add("Rust binary", True, rust_bin, required=True)
    else:
        add("Rust binary", False, "not found (cargo build --release)", required=True)

    try:
        import huggingface_hub  # noqa: F401

        add("HF Hub", True, "installed", required=False)
    except ImportError:
        add("HF Hub", False, "not installed (pip install huggingface_hub)", required=False)

    add("Rich", _RICH_AVAILABLE, "installed" if _RICH_AVAILABLE else "not installed (pip install rich)", required=False)

    from .model_cache import cache_root

    cache = cache_root()
    writable = os.access(str(cache.parent), os.W_OK)
    add("Cache dir", writable, str(cache), required=True)

    passed = all(row["ok"] for row in checks if row["required"])
    if args.json:
        print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    elif _RICH_AVAILABLE and _console:
        table = Table(title="Environment Check", show_lines=False)
        table.add_column("Component", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for row in checks:
            status = "[green]\u2713[/green]" if row["ok"] else "[red]\u2717[/red]"
            table.add_row(row["component"], status, row["detail"])

        _console.print(table)
    else:
        for row in checks:
            status = "\u2713" if row["ok"] else "\u2717"
            print(f"  {status} {row['component']}: {row['detail']}")
    if args.strict and not passed:
        raise SystemExit(1)


def cmd_cache(args: argparse.Namespace) -> None:
    from .model_cache import (
        cache_root,
        format_bytes,
        list_cached_archives,
        list_cached_models,
    )

    action = args.cache_command or "list"
    if action == "path":
        print(cache_root())
        return
    payload = {
        "root": str(cache_root()),
        "models": list_cached_models(),
        "archives": list_cached_archives(),
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    _header("ThinTensor Cache")
    _kv("Root", payload["root"])
    print()
    _print("Models:")
    if payload["models"]:
        for row in payload["models"]:
            _kv(row["name"], row["path"])
    else:
        _print("  (none)")
    print()
    _print("Archives:")
    if payload["archives"]:
        for row in payload["archives"]:
            _kv(row["name"], f"{format_bytes(row['size'])}  {row['path']}")
    else:
        _print("  (none)")


def cmd_core(args: argparse.Namespace) -> None:
    from .command_runner import require_rust_binary

    command = [require_rust_binary(), *args.args]
    raise SystemExit(subprocess.run(command, check=False).returncode)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _archive_model(archive_path: str) -> dict[str, Any]:
    from .archive import ThinArchive

    archive = ThinArchive(archive_path)
    try:
        model = archive.manifest.get("model")
        if not isinstance(model, dict):
            raise ValueError("archive manifest has no model descriptor")
        return dict(model)
    finally:
        archive.close()


def _tool_script(name: str) -> Path:
    source_path = PROJECT_ROOT / "scripts" / name
    if source_path.is_file():
        return source_path
    for root in (PROJECT_ROOT, Path(sys.prefix)):
        installed_path = root / "share" / "thintensor" / "scripts" / name
        if installed_path.is_file():
            return installed_path
    try:
        from importlib.metadata import distribution

        package = distribution("thintensor")
        for entry in package.files or ():
            normalized = str(entry).replace("\\", "/")
            if normalized.endswith(f"share/thintensor/scripts/{name}"):
                located = Path(package.locate_file(entry))
                if located.is_file():
                    return located
    except Exception:
        pass
    raise RuntimeError(
        f"installed ThinTensor package is missing backend tool {name!r}; "
        "reinstall the wheel or run from a complete source checkout"
    )


def _require_greedy_sampling(args: argparse.Namespace) -> None:
    if args.temperature != 0.0 or args.top_p != 1.0 or args.top_k != 0:
        raise SystemExit(
            "this runtime currently implements greedy decoding only; use "
            "--temperature 0 --top-p 1 --top-k 0"
        )


def _validate_generation_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.context <= 0:
        raise SystemExit("--context must be positive")
    if getattr(args, "kv_gpu_recent_tokens", 0) < 0:
        raise SystemExit("--kv-gpu-recent-tokens must be non-negative")
    if getattr(args, "prefetch_layers", 0) < 0:
        raise SystemExit("--prefetch-layers must be non-negative")


def _adapt_profile_for_device(
    profile: dict[str, Any],
    *,
    requested_name: str,
    model: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    if device != "cpu":
        result = dict(profile)
        tied_head_bytes = (
            int(model.get("vocab_size") or 0)
            * int(model.get("hidden_size") or 0)
        )
        if (
            result.get("lm_head_fp8")
            and bool(model.get("tie_word_embeddings"))
            and tied_head_bytes >= 384 * 1024 * 1024
        ):
            # A tied embedding cannot be replaced by the FP8 execution head.
            # Very large vocabularies would therefore require a second
            # multi-hundred-MiB copy before body quantization frees memory.
            result["lm_head_fp8"] = False
            result["keep_bf16_lm_head"] = True
            result["lm_head_topk_guard"] = 0
            result["profile_adaptations"] = [
                *result.get("profile_adaptations", []),
                "disabled duplicate FP8 head for large tied vocabulary",
            ]
        return result
    if requested_name.strip().lower() == "auto":
        from .profile_presets import get_profile

        profile = get_profile("bf16", model=model)
    if any(
        profile.get(key)
        for key in (
            "gate_up_fp8",
            "down_proj_fp8",
            "qkv_fp8",
            "o_proj_fp8",
            "lm_head_fp8",
        )
    ):
        raise ValueError(
            "FP8 profiles require CUDA; use --profile bf16 on CPU"
        )
    result = dict(profile)
    result["kernel_backend"] = "torch"
    result.pop("lm_head_backend", None)
    return result


def _parse_bytes(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"", "0"}:
        return 0
    suffixes = {
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "b": 1,
    }
    for suffix, multiplier in suffixes.items():
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            return int(float(number) * multiplier)
    return int(text)


def _emit_command(command: list[str], *, json_output: bool) -> None:
    import shlex

    if json_output:
        print(json.dumps({"command": command}, indent=2))
    else:
        print(shlex.join(command))


def _optimizer_plan_template(archive: str, hf_model: str) -> dict[str, Any]:
    quality_flags = [
        "--gate-up-fp8",
        "--down-proj-fp8",
        "--down-fp8-layers",
        "8:28",
    ]
    guarded_head_flags = [
        "--lm-head-fp8",
        "--keep-bf16-lm-head",
        "--lm-head-topk-guard",
        "64",
    ]
    correctness_guarded_flags = [
        "--lm-head-fp8",
        "--lm-head-topk-guard",
        "64",
    ]
    return {
        "archive": str(Path(archive).resolve()),
        "hf_model": hf_model,
        "device": "cuda",
        "dtype": "bf16",
        "residency": "all",
        "kernel_backend": "triton",
        "warmup_steps": 10,
        "benchmark_steps": [200, 500],
        "balanced_abba": True,
        "minimum_speedup": 0.02,
        "cosine_tolerance": 0.0005,
        "top5_tolerance": 0.0,
        "distribution_tolerance": 0.001,
        "baseline": {
            "name": "bf16_triton",
            "hypothesis": "Lock the real causal-KV BF16 baseline.",
            "expected_bottleneck": "large projection weight bandwidth",
            "benchmark_args": [],
            "correctness_args": [],
            "stress_args": [],
            "stress_mode": "bf16_triton",
        },
        "candidates": [
            {
                "name": "quality_body_fp8",
                "hypothesis": (
                    "Reduce gate/up and middle down-projection weight bytes "
                    "without quantizing attention or KV."
                ),
                "expected_bottleneck": "MLP projection weight bandwidth",
                "implementation_summary": (
                    "FP8 gate/up in all layers and down projection in 8:28."
                ),
                "benchmark_args": quality_flags,
                "correctness_args": quality_flags,
                "stress_args": [],
                "stress_mode": "retained_fp8",
            },
            {
                "name": "guarded_head",
                "hypothesis": (
                    "Use an FP8 vocabulary shortlist while retaining exact "
                    "BF16 candidate verification."
                ),
                "expected_bottleneck": "LM-head weight bandwidth",
                "implementation_summary": "FP8 shortlist plus BF16 top-64 guard.",
                "benchmark_args": guarded_head_flags,
                "correctness_args": correctness_guarded_flags,
                "stress_args": [],
                "stress_mode": "quality_8_28_head8_topk_guard",
            },
        ],
    }


def _ensure_hf_source(
    resolved: dict[str, Any],
    *,
    quiet: bool,
) -> str:
    """Return a local HF directory, downloading a model id when required."""
    if resolved["kind"] == "hf_dir":
        return str(Path(resolved["path"]).resolve())
    if resolved["kind"] != "hf_model_id":
        raise ValueError("a Hugging Face directory or model id is required")
    model_path = Path(resolved["model_path"])
    if resolved.get("needs_pull") or not model_path.exists():
        if not quiet:
            _print(
                "Pulling model..."
                if not _RICH_AVAILABLE
                else "[yellow]Pulling model...[/yellow]"
            )
        from .hf_pull import pull_model

        pull_model(resolved["path"], target_dir=model_path, quiet=quiet)
    return str(model_path.resolve())


def _ensure_archive(resolved: dict, args: argparse.Namespace) -> str:
    """Ensure we have a .thin archive, pulling/converting if needed."""
    quiet = bool(getattr(args, "json", False))
    if resolved["kind"] == "archive":
        return resolved["path"]

    if resolved["kind"] == "hf_model_id":
        if resolved.get("needs_pull"):
            if not quiet:
                _print("Pulling model..." if not _RICH_AVAILABLE else "[yellow]Pulling model...[/yellow]")
            from .hf_pull import pull_model

            pull_model(
                resolved["path"],
                target_dir=resolved.get("model_path"),
                quiet=quiet,
            )

        # Check if archive already cached
        archive_path = resolved.get("archive_path", "")
        if Path(archive_path).exists():
            if not quiet:
                _print(f"Using cached archive: {archive_path}")
            return archive_path

        # Convert
        model_path = resolved.get("model_path", "")
        if not Path(model_path).exists():
            _print(f"\u2717 Model directory not found: {model_path}")
            raise SystemExit(1)

        if not quiet:
            _print("Converting to .thin..." if not _RICH_AVAILABLE else "[yellow]Converting to .thin...[/yellow]")
        from .command_runner import convert_hf_model

        success = convert_hf_model(model_path, archive_path)
        if not success:
            _print("\u2717 Conversion failed!")
            raise SystemExit(1)
        return archive_path

    if resolved["kind"] == "hf_dir":
        archive_path = resolved.get("archive_path", "")
        if Path(archive_path).exists():
            if not quiet:
                _print(f"Using cached archive: {archive_path}")
            return archive_path

        if not quiet:
            _print("Converting to .thin..." if not _RICH_AVAILABLE else "[yellow]Converting to .thin...[/yellow]")
        from .command_runner import convert_hf_model

        success = convert_hf_model(resolved["path"], archive_path)
        if not success:
            _print("\u2717 Conversion failed!")
            raise SystemExit(1)
        return archive_path

    _print(f"\u2717 Cannot resolve model: {resolved.get('path', '')}")
    raise SystemExit(1)


def _parse_dtype(value: str):
    """Parse dtype string to torch.dtype."""
    import torch

    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping.get(value, torch.bfloat16)


def _load_transformers_model(
    *,
    model_source: str,
    archive_path: Optional[str] = None,
    tokenizer_source: Optional[str],
    device: str,
    dtype: str,
    trust_remote_code: bool,
):
    """Load the compatibility engine without importing Transformers at startup."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = tokenizer_source or model_source
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=trust_remote_code,
    )
    if archive_path:
        from .hf_loader import load_thin_model

        model, _diagnostics = load_thin_model(
            archive_path,
            model_source,
            device=device,
            dtype=_parse_dtype(dtype),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=_parse_dtype(dtype),
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
        model.to(torch.device(device))
    model.eval()
    return model, tokenizer


def _run_transformers_inference(
    *,
    model_source: str,
    archive_path: Optional[str] = None,
    prompt: str,
    max_new_tokens: int,
    context: int,
    device: str,
    dtype: str,
    tokenizer_source: Optional[str],
    trust_remote_code: bool,
    json_output: bool,
) -> None:
    """Compatibility path for causal-LM architectures not yet native."""
    import torch

    load_start = time.perf_counter()
    model, tokenizer = _load_transformers_model(
        model_source=model_source,
        archive_path=archive_path,
        tokenizer_source=tokenizer_source,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    load_s = time.perf_counter() - load_start
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    if prompt_tokens + max_new_tokens > context:
        raise ValueError(
            f"prompt ({prompt_tokens} tokens) plus --max-new-tokens "
            f"({max_new_tokens}) exceeds --context ({context})"
        )
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = output[0, prompt_tokens:]
    generated_count = int(generated.numel())
    text = tokenizer.decode(generated, skip_special_tokens=True)
    result = {
        "engine": "transformers",
        "native_optimized": False,
        "model": archive_path or model_source,
        "hf_source": model_source,
        "load_s": load_s,
        "generated_tokens": generated_count,
        "decode_s": elapsed,
        "tokens_per_s": generated_count / elapsed if elapsed else 0.0,
        "text": text,
        "warning": (
            "Compatibility fallback: this result is not evidence that "
            "ThinTensor outperforms Transformers."
        ),
    }
    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print(text)
        print()
        _kv("Engine", "Transformers compatibility fallback")
        _kv("Speed", f"{result['tokens_per_s']:.2f} tok/s")
        _kv("Load time", f"{load_s:.2f}s")


def _run_transformers_chat(
    *,
    model_source: str,
    archive_path: Optional[str] = None,
    max_new_tokens: int,
    context: int,
    device: str,
    dtype: str,
    tokenizer_source: Optional[str],
    trust_remote_code: bool,
) -> None:
    """Interactive compatibility chat using the model's chat template."""
    import torch

    model, tokenizer = _load_transformers_model(
        model_source=model_source,
        archive_path=archive_path,
        tokenizer_source=tokenizer_source,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    messages: list[dict[str, str]] = []
    _header("ThinTensor Chat")
    _kv("Engine", "Transformers compatibility fallback")
    _kv("Commands", "/reset, /exit")
    print()
    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            break
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break
        if prompt == "/reset":
            messages.clear()
            print("context cleared")
            continue
        messages.append({"role": "user", "content": prompt})
        if getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
        else:
            encoded = tokenizer(
                "\n".join(
                    f"{item['role']}: {item['content']}" for item in messages
                )
                + "\nassistant:",
                return_tensors="pt",
            )["input_ids"].to(device)
        if int(encoded.shape[-1]) + max_new_tokens > context:
            messages.pop()
            print("context limit reached; use /reset or increase --context")
            continue
        with torch.inference_mode():
            output = model.generate(
                input_ids=encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
        response = tokenizer.decode(
            output[0, encoded.shape[-1]:],
            skip_special_tokens=True,
        )
        print(f"assistant> {response}")
        messages.append({"role": "assistant", "content": response})


def _run_inference(
    *,
    archive_path: str,
    prompt: str,
    profile: dict,
    max_new_tokens: int,
    context: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    device: str,
    dtype: str,
    kv_residency: str,
    kv_gpu_recent_tokens: int,
    weight_residency: str,
    gpu_weight_budget: str,
    prefetch_layers: int,
    cpu_offload: bool,
    pin_cpu_pages: bool,
    tokenizer_source: Optional[str],
    json_output: bool,
) -> None:
    """Run greedy causal inference with explicit weight and KV residency."""
    import torch
    from .profile_presets import (
        activate_profile_environment,
        profile_to_runtime_kwargs,
    )

    activate_profile_environment(profile)

    torch_dtype = _parse_dtype(dtype)

    if seed > 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    if not json_output:
        _print("Loading model..." if not _RICH_AVAILABLE else "[yellow]Loading model...[/yellow]")
    t0 = time.perf_counter()

    from .gpu_runtime import (
        PagedKVCache,
        ThinGpuPagePool,
        ThinGpuCausalLMRuntime,
        ThinGpuWeights,
    )

    if weight_residency == "stream":
        budget = _parse_bytes(gpu_weight_budget)
        if budget <= 0:
            raise ValueError(
                "--residency stream requires a positive --gpu-weight-budget"
            )
        weights = ThinGpuPagePool(
            archive_path,
            device=device,
            dtype=torch_dtype,
            vram_budget_bytes=budget,
            prefetch_distance=prefetch_layers,
            cpu_offload=cpu_offload,
            pin_cpu_pages=pin_cpu_pages,
            down_proj_fp8=bool(profile.get("down_proj_fp8")),
            gate_up_fp8=bool(profile.get("gate_up_fp8")),
            qkv_fp8=bool(profile.get("qkv_fp8")),
            o_proj_fp8=bool(profile.get("o_proj_fp8")),
            down_fp8_layer_spec=profile.get("down_fp8_layer_spec"),
        )
        weights.warm_start()
    else:
        weights = ThinGpuWeights(archive_path, device=device, dtype=torch_dtype)
    manifest = weights.manifest
    model_info = manifest.get("model", {})

    layers = int(model_info.get("layers", 28))
    kv_heads = int(model_info.get("kv_heads", model_info.get("heads", 8)))
    head_dim = int(model_info.get("head_dim", model_info.get("hidden_size", 2048) // model_info.get("heads", 8)))

    kv_cache = PagedKVCache(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        device=torch.device(device),
        dtype=torch_dtype,
        block_size=int(profile.get("kv_block_size") or 16),
        residency=kv_residency,
        gpu_recent_tokens=kv_gpu_recent_tokens,
    )

    runtime_kwargs = profile_to_runtime_kwargs(profile)
    exact_prefill = bool(profile.get("exact_prefill"))
    if exact_prefill and weight_residency != "all":
        weights.close()
        raise ValueError("exact prefill currently requires all-resident weights")
    prefill_runtime = None
    runtime = None
    if exact_prefill:
        prefill_runtime = ThinGpuCausalLMRuntime(
            weights,
            kv_cache=kv_cache,
            kernel_backend="torch",
            attention_mode="causal_kv",
            attention_backend="torch",
            exact_hf_mode=True,
        )
    else:
        runtime = ThinGpuCausalLMRuntime(
            weights,
            kv_cache=kv_cache,
            prefetch_distance=(
                prefetch_layers if weight_residency == "stream" else 0
            ),
            evict_completed_layers=weight_residency == "stream",
            **runtime_kwargs,
        )

    if device == "cuda":
        torch.cuda.synchronize()
    load_time = time.perf_counter() - t0

    if not json_output:
        _kv("Load time", f"{load_time:.2f}s")
        _kv("Resident weights", _format_bytes(weights.resident_weight_bytes))
        print()

    token_ids = _tokenize_prompt(
        prompt,
        archive_path,
        manifest,
        tokenizer_source=tokenizer_source,
    )
    if len(token_ids) + max_new_tokens > context:
        weights.close()
        raise ValueError(
            f"prompt ({len(token_ids)} tokens) plus --max-new-tokens "
            f"({max_new_tokens}) exceeds --context ({context})"
        )

    if not json_output:
        _print("Output:" if not _RICH_AVAILABLE else "[bold]Output:[/bold]")
        print()

    generated_ids: list[int] = []
    decoded_text = ""

    for i, tid in enumerate(token_ids):
        assert prefill_runtime is not None or runtime is not None
        hidden = (prefill_runtime or runtime).forward_token(
            tid,
            token_index=i,
        )
    if prefill_runtime is not None:
        del prefill_runtime
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        runtime = ThinGpuCausalLMRuntime(
            weights,
            kv_cache=kv_cache,
            prefetch_distance=0,
            evict_completed_layers=False,
            **runtime_kwargs,
        )
    assert runtime is not None
    runtime.begin_decode(len(token_ids))
    if device == "cuda":
        torch.cuda.synchronize()
    t_gen = time.perf_counter()

    for step in range(max_new_tokens):
        token_index = len(token_ids) + step
        next_id_tensor = runtime.next_token_tensor(hidden)

        if device == "cuda":
            torch.cuda.synchronize()
        next_id = int(next_id_tensor.item())
        generated_ids.append(next_id)

        next_text = _decode_tokens(
            generated_ids,
            archive_path,
            manifest,
            tokenizer_source=tokenizer_source,
        )
        if not json_output:
            delta = (
                next_text[len(decoded_text):]
                if next_text.startswith(decoded_text)
                else next_text
            )
            print(delta, end="", flush=True)
        decoded_text = next_text

        if _is_eos(
            next_id,
            manifest,
            archive_path=archive_path,
            tokenizer_source=tokenizer_source,
        ) or step + 1 == max_new_tokens:
            break

        hidden = runtime.forward_token(next_id, token_index=token_index)

    if device == "cuda":
        torch.cuda.synchronize()
    gen_time = time.perf_counter() - t_gen
    tokens_generated = len(generated_ids)
    tok_per_s = tokens_generated / gen_time if gen_time > 0 else 0
    ms_per_tok = (gen_time / tokens_generated * 1000) if tokens_generated > 0 else 0

    if json_output:
        result = {
            "engine": "thintensor",
            "native_optimized": True,
            "profile": profile["name"],
            "weight_residency": weight_residency,
            "kv_residency": kv_residency,
            "prompt_tokens": len(token_ids),
            "generated_token_ids": generated_ids,
            "text": decoded_text,
            "tokens_generated": tokens_generated,
            "tokens_per_s": tok_per_s,
            "ms_per_token": ms_per_tok,
            "load_time_s": load_time,
            "decode_time_s": gen_time,
            "resident_weight_bytes": weights.resident_weight_bytes,
            "measurement_mode": "interactive_synchronous",
            "benchmark_comparable": False,
        }
        if device == "cuda":
            result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print()
        print()
        _header("Stats")
        _kv("Tokens generated", tokens_generated)
        _kv("Speed", f"{tok_per_s:.1f} tok/s")
        _kv("Latency", f"{ms_per_tok:.1f} ms/token")
        _kv("Resident weights", _format_bytes(weights.resident_weight_bytes))
        if device == "cuda":
            _kv("Peak GPU memory", _format_bytes(torch.cuda.max_memory_allocated()))

    weights.close()


def _run_chat_loop(
    *,
    archive_path: str,
    profile: dict,
    max_new_tokens: int,
    context: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    device: str,
    dtype: str,
    tokenizer_source: Optional[str],
) -> None:
    """Interactive chat loop."""
    import torch
    from .profile_presets import (
        activate_profile_environment,
        profile_to_runtime_kwargs,
    )

    activate_profile_environment(profile)

    torch_dtype = _parse_dtype(dtype)

    if seed > 0:
        torch.manual_seed(seed)

    _print("Loading model..." if not _RICH_AVAILABLE else "[yellow]Loading model...[/yellow]")
    t0 = time.perf_counter()

    from .gpu_runtime import (
        PagedKVCache,
        ThinGpuCausalLMRuntime,
        ThinGpuWeights,
    )

    weights = ThinGpuWeights(archive_path, device=device, dtype=torch_dtype)
    manifest = weights.manifest
    model_info = manifest.get("model", {})
    layers = int(model_info.get("layers", 28))
    kv_heads = int(model_info.get("kv_heads", model_info.get("heads", 8)))
    head_dim = int(model_info.get("head_dim", model_info.get("hidden_size", 2048) // model_info.get("heads", 8)))

    kv_cache = PagedKVCache(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        device=torch.device(device),
        dtype=torch_dtype,
        block_size=int(profile.get("kv_block_size") or 16),
    )

    runtime_kwargs = profile_to_runtime_kwargs(profile)
    exact_prefill = bool(profile.get("exact_prefill"))
    if exact_prefill:
        runtime = ThinGpuCausalLMRuntime(
            weights,
            kv_cache=kv_cache,
            kernel_backend="torch",
            attention_mode="causal_kv",
            attention_backend="torch",
            exact_hf_mode=True,
        )
    else:
        runtime = ThinGpuCausalLMRuntime(
            weights,
            kv_cache=kv_cache,
            **runtime_kwargs,
        )
    adaptive_weights_active = False

    load_time = time.perf_counter() - t0
    _kv("Loaded in", f"{load_time:.2f}s")
    _kv("Resident weights", _format_bytes(weights.resident_weight_bytes))
    print()

    total_tokens = 0
    total_gen_time = 0.0

    _print("Type your message. Commands: /exit /reset /stats /save <file>")
    print()

    token_index = 0
    chat_history: list[str] = []
    messages: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("\u276f ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input.strip():
            continue

        # Handle commands
        if user_input.strip().startswith("/"):
            cmd = user_input.strip().lower()
            if cmd == "/exit":
                break
            elif cmd == "/reset":
                kv_cache.reset()
                token_index = 0
                total_tokens = 0
                total_gen_time = 0.0
                chat_history.clear()
                messages.clear()
                _print("Chat reset.")
                continue
            elif cmd == "/stats":
                tok_per_s = total_tokens / total_gen_time if total_gen_time > 0 else 0
                _kv("Total tokens", total_tokens)
                _kv("Avg speed", f"{tok_per_s:.1f} tok/s")
                _kv("Resident weights", _format_bytes(weights.resident_weight_bytes))
                if device == "cuda":
                    _kv("Peak GPU", _format_bytes(torch.cuda.max_memory_allocated()))
                continue
            elif cmd.startswith("/save"):
                parts = user_input.strip().split(maxsplit=1)
                if len(parts) > 1:
                    save_path = Path(parts[1])
                    try:
                        save_path.write_text("\n".join(chat_history) + "\n", encoding="utf-8")
                        _print(f"Chat transcript saved to {save_path}")
                    except Exception as e:
                        _print(f"Failed to save transcript: {e}")
                else:
                    _print("Usage: /save <filename>")
                continue
            elif cmd == "/profile":
                _kv("Profile", profile.get("description", "unknown"))
                continue

        chat_history.append(f"User: {user_input}")
        messages.append({"role": "user", "content": user_input})
        token_ids = _tokenize_chat_messages(
            messages,
            archive_path,
            manifest,
            tokenizer_source=tokenizer_source,
        )
        if len(token_ids) + max_new_tokens > context:
            messages.pop()
            chat_history.pop()
            _print(
                "Context limit reached. Use /reset or restart with a larger "
                "--context."
            )
            continue
        # Rebuild the exact chat-template context each turn. This is more
        # expensive than incremental string concatenation, but it preserves
        # role/control tokens and avoids corrupting the conversation prefix.
        if exact_prefill and adaptive_weights_active:
            del runtime
            weights.close()
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            weights = ThinGpuWeights(
                archive_path,
                device=device,
                dtype=torch_dtype,
            )
            runtime = ThinGpuCausalLMRuntime(
                weights,
                kv_cache=kv_cache,
                kernel_backend="torch",
                attention_mode="causal_kv",
                attention_backend="torch",
                exact_hf_mode=True,
            )
            adaptive_weights_active = False
        kv_cache.reset(reuse_pages=True)
        token_index = 0
        for tid in token_ids:
            hidden = runtime.forward_token(tid, token_index=token_index)
            token_index += 1
        if exact_prefill:
            del runtime
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            runtime = ThinGpuCausalLMRuntime(
                weights,
                kv_cache=kv_cache,
                **runtime_kwargs,
            )
            adaptive_weights_active = True
        runtime.begin_decode(token_index)
        if device == "cuda":
            torch.cuda.synchronize()
        t_gen = time.perf_counter()

        # Generate
        gen_count = 0
        response_ids: list[int] = []
        response = ""
        for step in range(max_new_tokens):
            next_id_tensor = runtime.next_token_tensor(hidden)
            if device == "cuda":
                torch.cuda.synchronize()
            next_id = int(next_id_tensor.item())
            gen_count += 1
            response_ids.append(next_id)

            next_response = _decode_tokens(
                response_ids,
                archive_path,
                manifest,
                tokenizer_source=tokenizer_source,
            )
            delta = (
                next_response[len(response):]
                if next_response.startswith(response)
                else next_response
            )
            print(delta, end="", flush=True)
            response = next_response

            if _is_eos(
                next_id,
                manifest,
                archive_path=archive_path,
                tokenizer_source=tokenizer_source,
            ) or step + 1 == max_new_tokens:
                break

            hidden = runtime.forward_token(next_id, token_index=token_index)
            token_index += 1

        messages.append({"role": "assistant", "content": response})
        chat_history.append(f"Assistant: {response}")
        gen_time = time.perf_counter() - t_gen
        total_tokens += gen_count
        total_gen_time += gen_time
        tok_per_s = gen_count / gen_time if gen_time > 0 else 0

        print()
        if _RICH_AVAILABLE:
            _print(f"[dim]{gen_count} tokens, {tok_per_s:.1f} tok/s[/dim]")
        else:
            print(f"  [{gen_count} tokens, {tok_per_s:.1f} tok/s]")
        print()

    weights.close()
    _print("Goodbye!")


def _run_benchmark_profile(
    *,
    archive_path: str,
    profile: dict,
    profile_name: str,
    steps: int,
    warmup: int,
    device: str,
    dtype: str,
    max_gpu_temp: int,
    dry_run: bool,
    quiet: bool,
) -> Optional[dict]:
    """Run the trusted real causal-KV decode benchmark in a fresh process."""
    from .profile_presets import (
        profile_to_runtime_flags,
        subprocess_environment,
    )

    runtime_flags = [
        flag
        for flag in profile_to_runtime_flags(profile)
        if flag != "--exact-prefill"
    ]
    command = [
        sys.executable,
        str(_tool_script("thin_runtime.py")),
        "run",
        archive_path,
        "--device",
        device,
        "--dtype",
        dtype,
        "--residency",
        "all",
        "--steps",
        str(steps),
        "--warmup-steps",
        str(warmup),
        "--attention-mode",
        "causal_kv",
        "--max-gpu-temp",
        str(max_gpu_temp),
        "--json",
        *runtime_flags,
    ]
    if profile.get("experimental_int8_tensorcore"):
        command.append("--experimental-int8-tensorcore")
    if dry_run:
        return {
            "profile": profile_name,
            "engine": "thintensor",
            "command": command,
            "dry_run": True,
        }
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=subprocess_environment(profile),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        if completed.stderr and not quiet:
            print(completed.stderr, file=sys.stderr, end="")
        return {
            "profile": profile_name,
            "error": f"benchmark exited with status {completed.returncode}",
            "command": command,
        }
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "profile": profile_name,
            "error": f"benchmark returned invalid JSON: {exc}",
            "command": command,
        }
    return {
        "profile": profile_name,
        "engine": "thintensor",
        "tokens_per_s": raw.get("tokens_per_s"),
        "ms_per_token": raw.get("ms_per_token"),
        "resident_weight_bytes": raw.get("resident_weight_bytes"),
        "gpu_peak_allocated_bytes": raw.get("gpu_peak_allocated_bytes"),
        "effective_bandwidth_gb_s": raw.get("effective_bandwidth_gb_s"),
        "bytes_moved_per_token": raw.get("bytes_moved_per_token"),
        "steps": raw.get("steps"),
        "warmup_steps": raw.get("warmup_steps"),
        "attention_mode": "causal_kv",
        "correctness_required": True,
        "steady_state_eligible": steps >= 200 and warmup >= 10,
        "warning": (
            None
            if steps >= 200 and warmup >= 10
            else "smoke run only; use at least 10 warmup and 200 measured steps"
        ),
        "raw": raw,
    }


def _run_transformers_benchmark(
    *,
    model: str,
    steps: int,
    warmup: int,
    device: str,
    dtype: str,
    trust_remote_code: bool,
    dry_run: bool,
    quiet: bool,
) -> dict[str, Any]:
    """Benchmark the same single-stream greedy loop through Transformers."""
    command = [
        sys.executable,
        str(_tool_script("bench_hf_transformers_decode.py")),
        "--model",
        model,
        "--steps",
        str(steps),
        "--warmup-steps",
        str(warmup),
        "--device",
        device,
        "--dtype",
        dtype,
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    if dry_run:
        return {
            "profile": "transformers",
            "engine": "transformers",
            "command": command,
            "dry_run": True,
        }
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        if completed.stderr and not quiet:
            print(completed.stderr, file=sys.stderr, end="")
        return {
            "profile": "transformers",
            "engine": "transformers",
            "error": (
                "Transformers benchmark exited with status "
                f"{completed.returncode}"
            ),
            "command": command,
        }
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "profile": "transformers",
            "engine": "transformers",
            "error": f"Transformers benchmark returned invalid JSON: {exc}",
            "command": command,
        }
    return {
        "profile": "transformers",
        "engine": "transformers",
        "tokens_per_s": raw.get("tokens_per_s"),
        "ms_per_token": raw.get("ms_per_token"),
        "resident_weight_bytes": 0,
        "gpu_peak_allocated_bytes": raw.get("gpu_peak_allocated_bytes"),
        "steps": raw.get("steps"),
        "warmup_steps": warmup,
        "attention_mode": "causal_kv",
        "steady_state_eligible": steps >= 200 and warmup >= 10,
        "raw": raw,
    }


def _print_bench_table(results: list[dict], json_output: bool = False, out_dir: Optional[str] = None) -> None:
    """Print benchmark results as a table."""
    if out_dir:
        destination = Path(out_dir)
        if destination.suffix.lower() == ".json":
            out_file = destination
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            out_file = destination / "benchmark_results.json"
        out_file.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if json_output:
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    if results and all(row.get("dry_run") for row in results):
        for row in results:
            _emit_command(row["command"], json_output=False)
        return

    if _RICH_AVAILABLE and _console:
        table = Table(title="Benchmark Results", show_lines=True)
        table.add_column("Profile", style="bold cyan")
        table.add_column("tok/s", justify="right")
        table.add_column("ms/token", justify="right")
        table.add_column("Resident Weights", justify="right")
        table.add_column("Peak GPU", justify="right")
        table.add_column("vs HF", justify="right")

        for r in results:
            if "error" in r:
                table.add_row(r["profile"], "ERROR", "ERROR", "-", "-", "-")
                continue
            speedup = r.get("speedup_vs_transformers")
            table.add_row(
                r["profile"],
                f"{r['tokens_per_s']:.1f}",
                f"{r['ms_per_token']:.1f}",
                _format_bytes(r.get("resident_weight_bytes", 0)),
                _format_bytes(r.get("gpu_peak_allocated_bytes", 0)),
                f"{speedup:.2f}x" if speedup is not None else "-",
            )

        _console.print(table)
    else:
        header = f"{'Profile':<15} {'tok/s':>8} {'ms/tok':>8} {'Weights':>14} {'Peak GPU':>12}"
        print(header)
        print("-" * len(header))
        for r in results:
            if "error" in r:
                print(f"{r['profile']:<15} ERROR: {r['error']}")
                continue
            print(
                f"{r['profile']:<15} {r['tokens_per_s']:>8.1f} "
                f"{r['ms_per_token']:>8.1f} "
                f"{_format_bytes(r.get('resident_weight_bytes', 0)):>14} "
                f"{_format_bytes(r.get('gpu_peak_allocated_bytes', 0)):>12}"
            )

    if out_dir:
        _print(f"\nResults saved to {out_file}")


def _print_inspect_from_stats(stats: dict, archive_path: Path) -> None:
    """Print inspect output from Rust JSON stats."""
    from .model_cache import format_bytes

    model = stats.get("model", {})

    _header("ThinTensor Inspect")
    _kv("Archive", str(archive_path))
    _kv("Archive size", format_bytes(archive_path.stat().st_size))
    print()

    _kv("Architecture", model.get("arch", "unknown"))
    if model.get("raw_arch"):
        _kv("Raw architecture", model["raw_arch"])
    _kv("Layers", model.get("layers", "?"))
    _kv("Hidden size", model.get("hidden_size", "?"))
    _kv("Heads / KV heads", f"{model.get('heads', '?')} / {model.get('kv_heads', '?')}")

    head_dim = model.get("head_dim")
    if head_dim:
        _kv("Head dim", head_dim)

    _kv("DType", model.get("dtype", "?"))

    # Attention kind
    heads = model.get("heads", 0)
    kv_heads = model.get("kv_heads", 0)
    if heads and kv_heads:
        if kv_heads == 1:
            _kv("Attention", "MQA (multi-query)")
        elif kv_heads == heads:
            _kv("Attention", "MHA (multi-head)")
        else:
            _kv("Attention", f"GQA (grouped, {heads // kv_heads} groups)")

    print()
    _kv("Pages", stats.get("page_count", "?"))
    _kv("Raw weight bytes", format_bytes(stats.get("total_raw_bytes", 0)))
    _kv("Stored bytes", format_bytes(stats.get("total_stored_bytes", 0)))

    execution = stats.get("execution", {})
    if execution:
        print()
        _kv("Execution stages", execution.get("stage_count", "?"))
        _kv("Page refs", execution.get("page_ref_count", "?"))
        unreferenced = execution.get("unreferenced_pages", [])
        if unreferenced:
            _kv("Unreferenced pages", len(unreferenced))

    # Per-op breakdown
    per_op = stats.get("per_op", [])
    if per_op and _RICH_AVAILABLE and _console:
        print()
        table = Table(title="Per-Op Breakdown", show_lines=False)
        table.add_column("Op", style="bold")
        table.add_column("Pages", justify="right")
        table.add_column("Raw bytes", justify="right")

        for op in per_op[:12]:
            table.add_row(op["name"], str(op["pages"]), format_bytes(op["raw_bytes"]))

        _console.print(table)
    elif per_op:
        print()
        print("  Per-Op:")
        for op in per_op[:12]:
            print(f"    {op['name']}: {op['pages']} pages, {format_bytes(op['raw_bytes'])}")


def _print_inspect_from_archive(archive, archive_path: Path) -> None:
    """Print inspect output from Python ThinArchive."""
    from .model_cache import format_bytes

    manifest = archive.manifest
    model = manifest.get("model", {})
    pages = manifest.get("pages", [])

    _header("ThinTensor Inspect")
    _kv("Archive", str(archive_path))
    _kv("Archive size", format_bytes(archive_path.stat().st_size))
    _kv("Format version", manifest.get("version", 0))
    print()

    _kv("Architecture", model.get("arch", "unknown"))
    _kv("Layers", model.get("layers", "?"))
    _kv("Hidden size", model.get("hidden_size", "?"))
    _kv("Heads / KV heads", f"{model.get('heads', '?')} / {model.get('kv_heads', '?')}")
    _kv("DType", model.get("dtype", "?"))
    print()
    _kv("Pages", len(pages))

    # Memory plan
    mem = manifest.get("memory_plan", {})
    if mem:
        _kv("Scratch bytes", format_bytes(mem.get("scratch_bytes", 0)))
        _kv("Min VRAM", format_bytes(mem.get("min_vram_bytes", 0)))
        _kv("Recommended VRAM", format_bytes(mem.get("recommended_vram_bytes", 0)))
        kv = mem.get("kv_cache", {})
        if kv:
            _kv(
                "KV policy",
                f"recent {kv.get('recent_tokens_high_precision', 256)} high precision, "
                f"older {kv.get('old_tokens_codec', 'q4')}",
            )

    # Execution tape
    tape = manifest.get("execution_tape", [])
    if tape:
        _kv("Execution stages", len(tape))


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

_tokenizer_cache: dict[str, Any] = {}


def _get_tokenizer(
    archive_path: str,
    manifest: dict,
    *,
    tokenizer_source: Optional[str] = None,
):
    """Resolve an explicit, adjacent, or cached tokenizer."""
    cache_key = f"{archive_path}\0{tokenizer_source or ''}"
    if cache_key in _tokenizer_cache:
        return _tokenizer_cache[cache_key]

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        if tokenizer_source:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                clean_up_tokenization_spaces=False,
            )
        else:
            archive_dir = Path(archive_path).parent
            if (archive_dir / "tokenizer.json").exists() or (
                archive_dir / "tokenizer_config.json"
            ).exists():
                tokenizer = AutoTokenizer.from_pretrained(
                    str(archive_dir),
                    clean_up_tokenization_spaces=False,
                )
            else:
                from .model_cache import cached_model_path

                archive_stem = Path(archive_path).stem
                model_name = archive_stem.replace("--", "/")

                for candidate in [
                    cached_model_path(model_name),
                    Path(archive_stem),
                ]:
                    if candidate.exists() and (
                        (candidate / "tokenizer.json").exists()
                        or (candidate / "tokenizer_config.json").exists()
                    ):
                        tokenizer = AutoTokenizer.from_pretrained(
                            str(candidate),
                            clean_up_tokenization_spaces=False,
                        )
                        break
    except ImportError as exc:
        raise RuntimeError(
            "text generation requires transformers; install "
            "'thintensor[runtime]'"
        ) from exc
    except Exception as exc:
        if tokenizer_source:
            raise RuntimeError(
                f"failed to load tokenizer {tokenizer_source!r}: {exc}"
            ) from exc

    _tokenizer_cache[cache_key] = tokenizer
    return tokenizer


def _tokenize_prompt(
    prompt: str,
    archive_path: str,
    manifest: dict,
    *,
    tokenizer_source: Optional[str] = None,
) -> list[int]:
    tokenizer = _get_tokenizer(
        archive_path,
        manifest,
        tokenizer_source=tokenizer_source,
    )
    if tokenizer is None:
        raise RuntimeError(
            "no tokenizer found. Place tokenizer files beside the archive or "
            "pass --tokenizer MODEL_OR_DIRECTORY"
        )
    return tokenizer.encode(prompt)


def _tokenize_chat_messages(
    messages: list[dict[str, str]],
    archive_path: str,
    manifest: dict,
    *,
    tokenizer_source: Optional[str] = None,
) -> list[int]:
    tokenizer = _get_tokenizer(
        archive_path,
        manifest,
        tokenizer_source=tokenizer_source,
    )
    if tokenizer is None:
        raise RuntimeError(
            "chat requires a tokenizer; pass --tokenizer MODEL_OR_DIRECTORY"
        )
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]
    transcript = "".join(
        f"{message['role'].capitalize()}: {message['content']}\n"
        for message in messages
    )
    return tokenizer.encode(transcript + "Assistant:")


def _decode_token(
    token_id: int,
    archive_path: str,
    manifest: dict,
    *,
    tokenizer_source: Optional[str] = None,
) -> Optional[str]:
    """Decode a token ID to string."""
    tokenizer = _get_tokenizer(
        archive_path,
        manifest,
        tokenizer_source=tokenizer_source,
    )
    if tokenizer is not None:
        try:
            return tokenizer.decode([token_id])
        except Exception:
            return None
    return None


def _decode_tokens(
    token_ids: list[int],
    archive_path: str,
    manifest: dict,
    *,
    tokenizer_source: Optional[str] = None,
) -> str:
    tokenizer = _get_tokenizer(
        archive_path,
        manifest,
        tokenizer_source=tokenizer_source,
    )
    if tokenizer is None:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _is_eos(
    token_id: int,
    manifest: dict,
    *,
    archive_path: str,
    tokenizer_source: Optional[str] = None,
) -> bool:
    """Check if a token is EOS."""
    tokenizer = _get_tokenizer(
        archive_path,
        manifest,
        tokenizer_source=tokenizer_source,
    )

    if tokenizer is not None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            if isinstance(eos_id, (list, tuple)):
                return token_id in eos_id
            return token_id == eos_id

    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "core":
        forwarded = sys.argv[2:]
        if forwarded[:1] == ["--"]:
            forwarded = forwarded[1:]
        cmd_core(argparse.Namespace(args=forwarded))
        return

    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        cmd_version()
        return

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "convert": cmd_convert,
        "pull": cmd_pull,
        "run": cmd_run,
        "chat": cmd_chat,
        "bench": cmd_bench,
        "validate": cmd_validate,
        "optimize": cmd_optimize,
        "profiles": cmd_profiles,
        "explain": cmd_explain,
        "architectures": cmd_architectures,
        "inspect": cmd_inspect,
        "doctor": cmd_doctor,
        "cache": cmd_cache,
        "core": cmd_core,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as e:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(e)}))
        else:
            _print(f"\n\u2717 Error: {e}")
        if os.environ.get("THINTENSOR_DEBUG"):
            import traceback

            traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
