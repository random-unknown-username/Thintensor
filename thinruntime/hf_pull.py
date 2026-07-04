"""HuggingFace model pull/download support."""

from __future__ import annotations

import os
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
    if not check_huggingface_hub():
        print_install_instructions()
        raise SystemExit(1)

    from huggingface_hub import snapshot_download

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

    print(f"Pulling {model_id} \u2192 {target_dir}")
    print(f"  Revision: {revision}")
    if token:
        print("  Token: ****")
    print()

    try:
        result_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            revision=revision,
            token=token,
            local_dir_use_symlinks=False,
        )
        print(f"\u2713 Downloaded to {result_path}")
        return Path(result_path)
    except Exception as e:
        print(f"\u2717 Download failed: {e}")
        raise
