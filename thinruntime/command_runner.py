"""Command runner utilities for invoking the Rust thintensor binary."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def find_rust_binary() -> Optional[str]:
    """Find the thintensor Rust binary.

    Search order:
    1. THINTENSOR_BIN environment variable
    2. target/release/thintensor (relative to project root)
    3. target/debug/thintensor (relative to project root)
    4. System PATH
    """
    # 1. Environment variable
    env_bin = os.environ.get("THINTENSOR_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin

    # 2. Relative to project root
    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent

    for build_dir in ["release", "debug"]:
        candidate = project_root / "target" / build_dir / "thintensor"
        if candidate.exists():
            return str(candidate)

    # 3. System PATH
    found = shutil.which("thintensor")
    if found:
        return found

    return None


def require_rust_binary() -> str:
    """Find the Rust binary or exit with helpful message."""
    binary = find_rust_binary()
    if binary is None:
        print("\u2717 Could not find the 'thintensor' Rust binary.")
        print("")
        print("  Build it with:")
        print("    cargo build --release")
        print("")
        print("  Or set THINTENSOR_BIN environment variable.")
        raise SystemExit(1)
    return binary


def run_rust_command(
    subcommand: str,
    args: list[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a thintensor Rust CLI subcommand.

    Args:
        subcommand: Rust CLI subcommand (e.g. 'convert-hf', 'verify', 'inspect')
        args: Additional arguments
        capture: If True, capture stdout/stderr
        check: If True, raise on non-zero exit

    Returns:
        CompletedProcess result
    """
    binary = require_rust_binary()
    cmd = [binary, subcommand] + args

    kwargs: dict = {}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    return subprocess.run(cmd, check=check, **kwargs)


def convert_hf_model(
    hf_dir: str | Path,
    out_path: str | Path,
    *,
    arch: Optional[str] = None,
    include_tokenizer_hashes: bool = True,
    verify: bool = True,
) -> bool:
    """Convert a HuggingFace model directory to .thin archive.

    Returns True on success, False on failure.
    """
    args = [str(hf_dir), str(out_path)]

    if arch:
        args.extend(["--arch", arch])
    if not include_tokenizer_hashes:
        args.append("--no-tokenizer")

    try:
        run_rust_command("convert-hf", args)
    except subprocess.CalledProcessError:
        return False

    if verify:
        try:
            run_rust_command("verify", [str(out_path)])
        except subprocess.CalledProcessError:
            print("\u26a0  Archive verification failed!")
            return False

    return True


def verify_archive(archive_path: str | Path) -> bool:
    """Verify a .thin archive. Returns True if valid."""
    try:
        run_rust_command("verify", [str(archive_path)], capture=True)
        return True
    except subprocess.CalledProcessError:
        return False


def inspect_archive_json(archive_path: str | Path) -> Optional[dict]:
    """Run Rust inspect and return stats as JSON."""
    try:
        result = run_rust_command(
            "stats",
            [str(archive_path), "--json"],
            capture=True,
        )
        return json.loads(result.stdout.decode("utf-8"))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None
