"""ThinTensor model cache management."""

from __future__ import annotations

import os
from pathlib import Path


def cache_root() -> Path:
    """Get the cache root directory, respecting THINTENSOR_CACHE env var."""
    env = os.environ.get("THINTENSOR_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "thintensor"


def models_dir() -> Path:
    """Directory for downloaded HF models."""
    d = cache_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def archives_dir() -> Path:
    """Directory for converted .thin archives."""
    d = cache_root() / "archives"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_id_to_dirname(model_id: str) -> str:
    """Convert a HF model id like 'HuggingFaceTB/SmolLM3-3B' to a safe dir name."""
    return model_id.replace("/", "--")


def cached_model_path(model_id: str) -> Path:
    """Get the expected cache path for a HF model."""
    return models_dir() / model_id_to_dirname(model_id)


def cached_archive_path(model_id: str) -> Path:
    """Get the expected cache path for a .thin archive."""
    return archives_dir() / f"{model_id_to_dirname(model_id)}.thin"


def is_cached_model(model_id: str) -> bool:
    """Check if a HF model is cached locally."""
    path = cached_model_path(model_id)
    return path.exists() and any(path.glob("*.safetensors"))


def is_cached_archive(model_id: str) -> bool:
    """Check if a .thin archive is cached."""
    return cached_archive_path(model_id).exists()


def is_thin_archive(path: str) -> bool:
    """Check if a path points to a .thin archive file."""
    return path.endswith(".thin") and Path(path).is_file()


def is_hf_directory(path: str) -> bool:
    """Check if a path is a local HF model directory."""
    p = Path(path)
    if not p.is_dir():
        return False
    return (p / "config.json").exists() or any(p.glob("*.safetensors"))


def looks_like_hf_model_id(value: str) -> bool:
    """Check if a string looks like a HuggingFace model id (org/name)."""
    if "/" not in value:
        return False
    parts = value.split("/")
    if len(parts) != 2:
        return False
    return all(part and not part.startswith(".") for part in parts)


def resolve_input(value: str) -> dict:
    """Resolve an input argument to its type and path.

    Returns dict with:
        kind: 'archive' | 'hf_dir' | 'hf_model_id'
        path: resolved path (for archive/hf_dir) or model id
        needs_pull: whether the model needs to be downloaded
        needs_convert: whether the model needs conversion to .thin
    """
    # 1. .thin archive file
    if is_thin_archive(value):
        return {
            "kind": "archive",
            "path": str(Path(value).resolve()),
            "needs_pull": False,
            "needs_convert": False,
        }

    # 2. path ends with .thin but file doesn't exist
    if value.endswith(".thin"):
        return {
            "kind": "archive",
            "path": str(Path(value).resolve()),
            "needs_pull": False,
            "needs_convert": False,
            "error": f"Archive not found: {value}",
        }

    # 2b. Implicit archive reference (bare name or HF ID) that exists in cache
    possible_archive = cached_archive_path(value)
    if possible_archive.exists():
        return {
            "kind": "archive",
            "path": str(possible_archive),
            "needs_pull": False,
            "needs_convert": False,
        }

    # 3. Local HF directory
    if is_hf_directory(value):
        archive = cached_archive_path(Path(value).name)
        return {
            "kind": "hf_dir",
            "path": str(Path(value).resolve()),
            "archive_path": str(archive),
            "needs_pull": False,
            "needs_convert": True,
        }

    # 4. Local directory (without HF markers)
    if Path(value).is_dir():
        return {
            "kind": "hf_dir",
            "path": str(Path(value).resolve()),
            "archive_path": str(cached_archive_path(Path(value).name)),
            "needs_pull": False,
            "needs_convert": True,
        }

    # 5. Looks like HF model id
    if looks_like_hf_model_id(value):
        model_path = cached_model_path(value)
        archive_path = cached_archive_path(value)
        return {
            "kind": "hf_model_id",
            "path": value,
            "model_path": str(model_path),
            "archive_path": str(archive_path),
            "needs_pull": not is_cached_model(value),
            "needs_convert": not is_cached_archive(value),
        }

    # 6. Unknown
    return {
        "kind": "unknown",
        "path": value,
        "error": (
            f"Cannot resolve '{value}'. Provide a .thin archive, "
            "HF directory, or HF model id (org/name)."
        ),
    }


def list_cached_archives() -> list[dict]:
    """List all cached .thin archives."""
    results = []
    arch_dir = archives_dir()
    if arch_dir.exists():
        for f in sorted(arch_dir.glob("*.thin")):
            results.append({
                "name": f.stem.replace("--", "/"),
                "path": str(f),
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
            })
    return results


def list_cached_models() -> list[dict]:
    """List all cached HF models."""
    results = []
    mod_dir = models_dir()
    if mod_dir.exists():
        for d in sorted(mod_dir.iterdir()):
            if d.is_dir() and any(d.glob("*.safetensors")):
                results.append({
                    "name": d.name.replace("--", "/"),
                    "path": str(d),
                })
    return results


def format_bytes(value: int) -> str:
    """Format bytes as human-readable string."""
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
