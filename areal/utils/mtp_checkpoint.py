# SPDX-License-Identifier: Apache-2.0

"""Preserve frozen Qwen MTP tensors without materializing a training-side head."""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

_QWEN_TYPES = {"qwen3_5", "qwen3_5_moe", "qwen3_next"}
_COPY_CHUNK_BYTES = 8 * 1024 * 1024


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header = json.loads(stream.read(size))
    return header, size + 8


def _inventory(
    directory: Path, exclude: Path | None = None
) -> dict[str, tuple[Path, dict[str, Any], int]]:
    """Inspect actual files, since Bridge's index may contain dropped MTP keys."""
    tensors = {}
    for path in sorted(directory.glob("*.safetensors")):
        if path == exclude:
            continue
        header, data_start = _read_header(path)
        data_size = path.stat().st_size - data_start
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            if not 0 <= begin <= end <= data_size:
                raise ValueError(f"Invalid tensor offsets for {name} in {path}")
            if name in tensors:
                raise ValueError(f"Duplicate tensor {name} in {directory}")
            tensors[name] = (path, info, data_start)
    return tensors


def _copy_mtp_shard(
    tensors: dict[str, tuple[Path, dict[str, Any], int]], path: Path
) -> None:
    """Copy raw tensor payloads in bounded CPU buffers, including uncommon dtypes."""
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (_, info, _) in sorted(tensors.items()):
        begin, end = info["data_offsets"]
        size = end - begin
        header[name] = {**info, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for source, info, data_start in (v for _, v in sorted(tensors.items())):
            begin, end = info["data_offsets"]
            remaining = end - begin
            with source.open("rb") as stream:
                stream.seek(data_start + begin)
                while remaining:
                    chunk = stream.read(min(remaining, _COPY_CHUNK_BYTES))
                    if not chunk:
                        raise ValueError(f"Truncated MTP source shard: {source}")
                    output.write(chunk)
                    remaining -= len(chunk)


def _restore_mtp_config(source: dict, destination: dict) -> None:
    for key, value in source.items():
        if key.startswith("mtp_") or key == "num_nextn_predict_layers":
            destination[key] = value
        elif isinstance(value, dict) and isinstance(destination.get(key), dict):
            _restore_mtp_config(value, destination[key])


def restore_frozen_mtp_weights(source_path: str, output_path: str) -> bool:
    """Complete a Qwen HF export with the original frozen ``mtp.*`` tensors.

    Call on rank zero after the bridge finishes exporting the trained backbone.
    ``source_path`` must be the resolved local directory used to initialize the
    bridge, not a different config/assets directory supplied to the exporter.
    Unsupported model families return False and retain their existing export
    policy. A supported model with missing/partial MTP weights raises instead of
    silently producing an incomplete deployment checkpoint. No CUDA or model
    construction is involved; DCP training-state serialization is unchanged.
    """
    source = Path(source_path)
    output = Path(output_path)
    source_config = json.loads((source / "config.json").read_text())
    if source_config.get("model_type") not in _QWEN_TYPES:
        return False
    if source.resolve() == output.resolve():
        raise ValueError("Frozen MTP source and HF export directory must differ")

    source_tensors = _inventory(source)
    mtp = {
        name: value for name, value in source_tensors.items() if name.startswith("mtp.")
    }
    if not mtp:
        raise ValueError(f"No frozen MTP tensors found in {source}")
    # An index can refer to an absent shard even when other MTP tensors exist.
    source_index = source / "model.safetensors.index.json"
    if source_index.exists():
        weight_map = json.loads(source_index.read_text())["weight_map"]
        for name, filename in weight_map.items():
            if name.startswith("mtp.") and (
                name not in mtp or mtp[name][0].name != filename
            ):
                raise ValueError(f"Missing or misplaced frozen MTP tensor: {name}")

    single = output / "model.safetensors"
    backbone = output / "model-backbone.safetensors"
    previous_backbone = None
    if single.exists() and backbone.exists():
        # A subsequent Bridge export may recreate model.safetensors. Replace
        # the previous backbone only if the new file supplies every same key.
        new_header, _ = _read_header(single)
        old_header, _ = _read_header(backbone)
        if new_header.keys() - {"__metadata__"} != old_header.keys() - {"__metadata__"}:
            raise ValueError(f"Conflicting single-file HF export: {backbone}")
        previous_backbone = backbone
    saved = _inventory(output, exclude=previous_backbone)
    present = {name for name in saved if name.startswith("mtp.")}
    if present and present != mtp.keys():
        raise ValueError("HF export contains partial or unexpected MTP weights")
    for name in present:
        for field in ("dtype", "shape"):
            if saved[name][1][field] != mtp[name][1][field]:
                raise ValueError(f"Incompatible saved MTP tensor: {name}")
    if not any(not name.startswith("mtp.") for name in saved):
        raise ValueError("Cannot add frozen MTP to an empty backbone export")

    config = json.loads((output / "config.json").read_text())
    _restore_mtp_config(source_config, config)
    target = output / "model-frozen-mtp.safetensors"
    # Bridge's strict=False export can leave an empty MTP-only source shard
    # when resuming from one of our HF exports. It is safe to replace that
    # shell, but never overwrite a shard containing unrelated tensors.
    if not present and any(path == target for path, _, _ in saved.values()):
        raise ValueError(f"Refusing to overwrite unrelated weights: {target}")

    with tempfile.TemporaryDirectory(prefix=".frozen-mtp-", dir=output) as temp:
        staging = Path(temp)
        if not present:
            _copy_mtp_shard(mtp, staging / target.name)
            os.replace(staging / target.name, target)
        # HF loaders prefer model.safetensors over an index if both exist.
        if single.exists():
            os.replace(single, backbone)
        actual = _inventory(output)
        index = {
            "metadata": {
                "total_size": sum(
                    info["data_offsets"][1] - info["data_offsets"][0]
                    for _, info, _ in actual.values()
                )
            },
            "weight_map": {name: path.name for name, (path, _, _) in actual.items()},
        }
        for filename, content in (
            ("model.safetensors.index.json", index),
            ("config.json", config),
        ):
            staged = staging / filename
            staged.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
            os.replace(staged, output / filename)
    return True
