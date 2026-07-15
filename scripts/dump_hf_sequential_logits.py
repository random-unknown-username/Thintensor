#!/usr/bin/env python3
"""Run an exact HF Qwen3.5-family reference one decoder layer at a time.

This avoids loading a 27B BF16 checkpoint into 30 GiB of RAM.  The official
Transformers decoder implementation is instantiated on `meta`; one layer's
parameters are mapped from SafeTensors, executed on CUDA, and released before
the next layer.  Embedding rows and vocabulary-head rows are sliced so the two
multi-gigabyte global matrices never need to be resident in full.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.archive import ThinArchive
from thinruntime.gpu_runtime import load_tensor_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument(
        "--archive",
        help="Load verified BF16 tensors from a Thin archive after source shards were consumed",
    )
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--head-chunk-rows", type=int, default=8192)
    parser.add_argument("--revision")
    parser.add_argument(
        "--save-input-stats-npz",
        help="Save per-projection input absmax/RMS vectors for activation-aware packing",
    )
    parser.add_argument(
        "--save-input-covariance-npz",
        help="Save per-projection 32-channel covariance blocks for GPTQ error feedback",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_dir = Path(args.hf_dir).resolve()
    trajectory_path = Path(args.trajectory).resolve()
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    steps = [int(value) for value in trajectory["steps"]]
    prompt_tokens = [int(value) for value in trajectory["prompt_token_ids"]]
    teacher_tokens = [int(value) for value in trajectory["teacher_token_ids"]]
    max_step = max(steps)
    token_ids = prompt_tokens + teacher_tokens[: max_step - 1]
    positions = [len(prompt_tokens) + step - 2 for step in steps]
    if min(positions) < 0 or max(positions) >= len(token_ids):
        raise ValueError("trajectory checkpoints are outside the token sequence")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    config = AutoConfig.from_pretrained(hf_dir).text_config
    config._attn_implementation = "eager"
    index = json.loads(
        (hf_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    archive = ThinArchive(args.archive, run_verify=False) if args.archive else None
    started = time.perf_counter()

    hidden = embedding_rows(
        hf_dir,
        index,
        "model.language_model.embed_tokens.weight",
        token_ids,
        device,
        dtype,
        archive,
    ).unsqueeze(0)
    sequence_length = hidden.shape[1]
    base_positions = torch.arange(sequence_length, device=device)
    four_axis_positions = base_positions.view(1, 1, -1).expand(4, 1, -1)
    text_positions = four_axis_positions[0]
    rope_positions = four_axis_positions[1:]
    rotary = Qwen3_5TextRotaryEmbedding(config, device=device)
    position_embeddings = rotary(hidden, rope_positions)
    causal_mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=text_positions,
    )
    del rotary

    layer_seconds: list[float] = []
    input_stats: dict[str, dict[str, np.ndarray]] = {}
    with torch.inference_mode():
        for layer_index, layer_type in enumerate(config.layer_types):
            layer_started = time.perf_counter()
            layer = load_layer(
                hf_dir,
                index,
                config,
                layer_index,
                device,
                dtype,
                archive,
            )
            hooks = register_input_stat_hooks(
                layer,
                layer_index,
                input_stats,
                capture_covariance=bool(args.save_input_covariance_npz),
            ) if args.save_input_stats_npz or args.save_input_covariance_npz else []
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=(causal_mask if layer_type == "full_attention" else None),
                position_ids=text_positions,
                past_key_values=None,
                use_cache=False,
            )
            for hook in hooks:
                hook.remove()
            del layer
            gc.collect()
            torch.cuda.empty_cache()
            layer_seconds.append(time.perf_counter() - layer_started)
            print(
                f"HF sequential layer {layer_index + 1}/{config.num_hidden_layers} "
                f"({layer_type}) {layer_seconds[-1]:.3f}s",
                flush=True,
            )

        norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
            device=device, dtype=dtype
        )
        norm.weight.copy_(
            load_tensor(
                hf_dir,
                index,
                "model.language_model.norm.weight",
                device,
                dtype,
                archive,
            )
        )
        selected_hidden = norm(hidden[:, positions, :]).squeeze(0)
        del norm, hidden, position_embeddings, causal_mask
        torch.cuda.empty_cache()
        logits = chunked_lm_head(
            hf_dir,
            index,
            "lm_head.weight",
            selected_hidden,
            int(config.vocab_size),
            args.head_chunk_rows,
            device,
            dtype,
            archive,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = out.with_suffix(".npz")
    arrays = {f"step_{step}": logits[row] for row, step in enumerate(steps)}
    np.savez_compressed(arrays_path, **arrays)
    input_stats_path = None
    input_covariance_path = None
    input_stats_mapping: dict[str, dict[str, str]] = {}
    if args.save_input_stats_npz:
        input_stats_path = Path(args.save_input_stats_npz)
        input_stats_path.parent.mkdir(parents=True, exist_ok=True)
        stat_arrays: dict[str, np.ndarray] = {}
        for number, (page_id, stats) in enumerate(sorted(input_stats.items())):
            prefix = f"projection_{number:04d}"
            input_stats_mapping[page_id] = {
                "absmax": prefix + "_absmax",
                "rms": prefix + "_rms",
            }
            stat_arrays[prefix + "_absmax"] = stats["absmax"]
            stat_arrays[prefix + "_rms"] = stats["rms"]
        np.savez_compressed(input_stats_path, **stat_arrays)
    if args.save_input_covariance_npz:
        input_covariance_path = Path(args.save_input_covariance_npz)
        input_covariance_path.parent.mkdir(parents=True, exist_ok=True)
        covariance_arrays: dict[str, np.ndarray] = {}
        for number, (page_id, stats) in enumerate(sorted(input_stats.items())):
            prefix = f"projection_{number:04d}"
            input_stats_mapping.setdefault(page_id, {})["covariance32"] = (
                prefix + "_covariance32"
            )
            covariance_arrays[prefix + "_covariance32"] = stats["covariance32"]
        np.savez_compressed(input_covariance_path, **covariance_arrays)
    checkpoints = []
    for row, step in enumerate(steps):
        values = logits[row]
        top = np.argpartition(values, -5)[-5:]
        top = top[np.argsort(values[top])[::-1]]
        checkpoints.append(
            {
                "step": step,
                "sequence_position": positions[row],
                "top5_token_ids": [int(value) for value in top],
                "top5_logits": [float(values[value]) for value in top],
                "all_values_finite": bool(np.isfinite(values).all()),
            }
        )
    report = {
        "schema": "thintensor.hf_sequential_logits.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hf_dir": str(hf_dir),
        "archive": str(Path(args.archive).resolve()) if args.archive else None,
        "revision": args.revision,
        "architecture": type(config).__name__,
        "transformers_implementation": (
            "Qwen3_5DecoderLayer eager full recompute, one layer resident at a time"
        ),
        "dtype": args.dtype,
        "device": str(device),
        "trajectory": str(trajectory_path),
        "prompt_token_ids": prompt_tokens,
        "teacher_token_ids": teacher_tokens,
        "steps": steps,
        "vocab_size": int(config.vocab_size),
        "layer_seconds": layer_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "arrays": str(arrays_path.resolve()),
        "arrays_sha256": sha256(arrays_path),
        "input_stats": (
            str(input_stats_path.resolve()) if input_stats_path else None
        ),
        "input_stats_sha256": (
            sha256(input_stats_path) if input_stats_path else None
        ),
        "input_stats_mapping": input_stats_mapping,
        "input_covariance": (
            str(input_covariance_path.resolve()) if input_covariance_path else None
        ),
        "input_covariance_sha256": (
            sha256(input_covariance_path) if input_covariance_path else None
        ),
        "checkpoints": checkpoints,
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if archive is not None:
        archive.close()
    print(f"wrote {out} and {arrays_path}")


def load_layer(
    hf_dir: Path,
    index: dict[str, str],
    config: Any,
    layer_index: int,
    device: torch.device,
    dtype: torch.dtype,
    archive: ThinArchive | None = None,
) -> Qwen3_5DecoderLayer:
    prefix = f"model.language_model.layers.{layer_index}."
    names = [name for name in index if name.startswith(prefix)]
    if not names:
        raise RuntimeError(f"no SafeTensors entries found for layer {layer_index}")
    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(config, layer_index)
    state = {
        name.removeprefix(prefix): load_tensor(
            hf_dir, index, name, device, dtype, archive
        )
        for name in names
    }
    missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"layer {layer_index} state mismatch: missing={missing}, unexpected={unexpected}"
        )
    layer.eval()
    return layer


def load_tensor(
    hf_dir: Path,
    index: dict[str, str],
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    archive: ThinArchive | None = None,
) -> torch.Tensor:
    if archive is not None:
        canonical = canonical_archive_name(name)
        tensor, _ = load_tensor_view(archive, canonical)
        return tensor.to(device=device, dtype=dtype)
    shard = hf_dir / index[name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).to(device=device, dtype=dtype)


def embedding_rows(
    hf_dir: Path,
    index: dict[str, str],
    name: str,
    token_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
    archive: ThinArchive | None = None,
) -> torch.Tensor:
    if archive is not None:
        source, _ = load_tensor_view(archive, canonical_archive_name(name))
        return source[token_ids].to(device=device, dtype=dtype)
    shard = hf_dir / index[name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        source = handle.get_slice(name)
        rows = [source[token : token + 1] for token in token_ids]
    return torch.cat(rows, dim=0).to(device=device, dtype=dtype)


def chunked_lm_head(
    hf_dir: Path,
    index: dict[str, str],
    name: str,
    hidden: torch.Tensor,
    vocab_size: int,
    chunk_rows: int,
    device: torch.device,
    dtype: torch.dtype,
    archive: ThinArchive | None = None,
) -> np.ndarray:
    if chunk_rows <= 0:
        raise ValueError("--head-chunk-rows must be positive")
    output = np.empty((hidden.shape[0], vocab_size), dtype=np.float32)
    if archive is not None:
        source, _ = load_tensor_view(archive, canonical_archive_name(name))
        for start in range(0, vocab_size, chunk_rows):
            end = min(vocab_size, start + chunk_rows)
            weight = source[start:end].to(device=device, dtype=dtype)
            values = torch.matmul(hidden, weight.t()).float().cpu().numpy()
            output[:, start:end] = values
            del weight, values
        return output
    shard = hf_dir / index[name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        source = handle.get_slice(name)
        for start in range(0, vocab_size, chunk_rows):
            end = min(vocab_size, start + chunk_rows)
            weight = source[start:end].to(device=device, dtype=dtype)
            values = torch.matmul(hidden, weight.t()).float().cpu().numpy()
            output[:, start:end] = values
            del weight, values
    return output


def canonical_archive_name(name: str) -> str:
    prefix = "model.language_model."
    return "model." + name.removeprefix(prefix) if name.startswith(prefix) else name


def register_input_stat_hooks(
    layer: Qwen3_5DecoderLayer,
    layer_index: int,
    output: dict[str, dict[str, np.ndarray]],
    *,
    capture_covariance: bool = False,
) -> list[Any]:
    hooks = []
    for module_name, module in layer.named_modules():
        weight = getattr(module, "weight", None)
        if not module_name or not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        page_id = f"model.layers.{layer_index}.{module_name}.weight"

        def capture(_module: Any, inputs: tuple[Any, ...], *, page_id: str = page_id) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            values = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            output[page_id] = {
                "absmax": values.abs().amax(dim=0).cpu().numpy(),
                "rms": values.square().mean(dim=0).sqrt().cpu().numpy(),
            }
            if capture_covariance:
                if values.shape[1] % 32:
                    raise ValueError(
                        f"projection input width {values.shape[1]} is not divisible by 32"
                    )
                grouped = values.reshape(values.shape[0], -1, 32)
                covariance = torch.einsum("ngi,ngj->gij", grouped, grouped)
                covariance.div_(max(1, int(values.shape[0])))
                output[page_id]["covariance32"] = covariance.cpu().numpy()

        hooks.append(module.register_forward_pre_hook(capture))
    return hooks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
