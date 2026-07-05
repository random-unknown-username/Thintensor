"""HuggingFace model pull/download support."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def check_huggingface_hub() -> bool:
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


def print_install_instructions() -> None:
    """Print instructions for installing huggingface_hub."""
    print("\n\u26a0  huggingface_hub is not installed.")
    print("   Install it with:")
    print("")
    print("     pip install huggingface_hub")
    print("")
    print("   Or install all ThinTensor dependencies:")
    print("")
    print("     pip install huggingface_hub torch")
    print("")


def pull_model(
    model_id: str,
    target_dir: Optional[str | Path] = None,
    revision: str = "main",
    token: Optional[str] = None,
    quiet: bool = False,
    backend: str = "auto",
) -> Path:
    """Download a HuggingFace model.

    Args:
        model_id: HuggingFace model identifier (e.g. 'HuggingFaceTB/SmolLM3-3B')
        target_dir: Directory to save the model (default: cache dir)
        revision: Git revision to download
        token: HuggingFace API token

    Returns:
        Path to the downloaded model directory
    """
    if target_dir is None:
        from .model_cache import cached_model_path
        target_dir = cached_model_path(model_id)

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Resolve token from env if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )

    if backend not in {"auto", "python", "hf-cli"}:
        raise ValueError("backend must be auto, python, or hf-cli")
    hf_binary = shutil.which("hf")
    if backend == "auto" and hf_binary is not None:
        authenticated = subprocess.run(
            [hf_binary, "auth", "whoami"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if authenticated:
            backend = "hf-cli"
    if backend == "auto":
        backend = "python"

    if not quiet:
        print(f"Pulling {model_id} \u2192 {target_dir}")
        print(f"  Revision: {revision}")
        if token:
            print("  Token: ****")
        print()

    if backend == "hf-cli":
        if hf_binary is None:
            raise RuntimeError(
                "the `hf` CLI is required for --download-backend hf-cli"
            )
        command = [
            hf_binary,
            "download",
            model_id,
            "--revision",
            revision,
            "--local-dir",
            str(target_dir),
        ]
        if quiet:
            command.extend(["--format", "quiet"])
        environment = dict(os.environ)
        if token:
            # Keep credentials out of argv and process listings.
            environment["HF_TOKEN"] = token
        try:
            subprocess.run(command, env=environment, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Hugging Face CLI download failed. For gated models, run "
                "`hf auth login`, accept the model license on the Hub, then "
                "retry with `thintensor pull MODEL --download-backend hf-cli`."
            ) from exc
        if not quiet:
            print(f"\u2713 Downloaded to {target_dir}")
        return target_dir

    if not check_huggingface_hub():
        print_install_instructions()
        raise SystemExit(1)

    from huggingface_hub import snapshot_download

    try:
        result_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            revision=revision,
            token=token,
        )
        if not quiet:
            print(f"\u2713 Downloaded to {result_path}")
        return Path(result_path)
    except Exception as e:
        if not quiet:
            print(f"\u2717 Download failed: {e}")
        raise RuntimeError(
            f"download failed for {model_id}. For gated models, run "
            "`hf auth login` and retry with `--download-backend hf-cli`."
        ) from e
