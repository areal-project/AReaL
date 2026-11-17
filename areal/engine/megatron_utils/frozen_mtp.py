# SPDX-License-Identifier: Apache-2.0

"""CPU-owned MTP weights for full HF checkpoints with MTP training disabled."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_MTP_COUNTS = ("mtp_num_hidden_layers", "num_nextn_predict_layers")


def is_mtp_key(name: str) -> bool:
    return any(part == "mtp" or part.startswith("nextn") for part in name.split("."))


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def read_mtp_tensors(source: str) -> dict[str, dict[str, torch.Tensor]]:
    """Read every declared MTP tensor, rejecting missing shards or keys."""
    root = Path(source)
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as f:
            weight_map = {name: "model.safetensors" for name in f.keys()}
    shards: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        if is_mtp_key(name):
            shards.setdefault(filename, []).append(name)
    result = {}
    for filename, names in shards.items():
        with safe_open(root / filename, framework="pt", device="cpu") as f:
            result[filename] = {name: f.get_tensor(name).clone() for name in names}
    return result


def _mtp_config(config: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in config.items():
        if key in _MTP_COUNTS:
            result[key] = value
        elif isinstance(value, dict):
            nested = _mtp_config(value)
            if nested:
                result[key] = nested
    return result


def _merge_config(config: dict[str, Any], mtp: dict[str, Any]) -> None:
    for key, value in mtp.items():
        if isinstance(value, dict):
            _merge_config(config.setdefault(key, {}), value)
        else:
            config[key] = value


def validate_mtp_checkpoint(source: str) -> None:
    """Require MTP weights before Bridge loads a trainable head."""
    root = Path(source)
    index = root / "model.safetensors.index.json"
    if index.exists():
        names = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as f:
            names = f.keys()
    if not any(is_mtp_key(name) for name in names):
        raise ValueError(f"No MTP weights found in {source}")


@dataclass
class FrozenMTPState:
    """An immutable-by-ownership snapshot, independent of the original files."""

    shards: dict[str, dict[str, torch.Tensor]]
    config: dict[str, Any]

    @classmethod
    def load(cls, source: str) -> FrozenMTPState:
        root = Path(source)
        config = json.loads((root / "config.json").read_text())
        shards = read_mtp_tensors(str(root))
        if not shards:
            raise ValueError(f"No frozen MTP weights found in {source}")
        return cls(shards=shards, config=_mtp_config(config))

    def export(self, destination: str) -> None:
        """Merge only MTP-owned tensors and rebuild a complete shard index."""
        root = Path(destination)
        existing = {}
        for path in root.glob("*.safetensors"):
            for key in _read_safetensors_header(path):
                if key in existing:
                    raise ValueError(f"Duplicate tensor in HF export: {key}")
                existing[key] = path.name
        single = root / "model.safetensors"
        sidecar = root / "model-mtp.safetensors"
        updates = {}
        for filename, tensors in self.shards.items():
            for key, tensor in tensors.items():
                target = existing.get(key)
                if target is None:
                    if single.exists():
                        # Transformers prefers this file over the shard index.
                        target = single.name
                    elif (root / filename).exists():
                        target = filename
                    else:
                        target = sidecar.name
                updates.setdefault(target, {})[key] = tensor
        for filename, tensors in updates.items():
            path = root / filename
            merged = {}
            if path.exists():
                with safe_open(path, framework="pt", device="cpu") as f:
                    merged = {key: f.get_tensor(key).clone() for key in f.keys()}
            merged.update(tensors)
            save_file(merged, path, metadata={"format": "pt"})
        weight_map = {}
        total_size = 0
        for path in sorted(root.glob("*.safetensors")):
            for key, metadata in _read_safetensors_header(path).items():
                weight_map[key] = path.name
                begin, end = metadata["data_offsets"]
                total_size += end - begin
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        # Only the intentionally unbuilt MTP tensors may have been absent.
        missing = set(index.get("weight_map", {})) - weight_map.keys()
        if missing:
            raise ValueError(f"Incomplete HF export; missing {sorted(missing)}")
        index["weight_map"] = weight_map
        index.setdefault("metadata", {})["total_size"] = total_size
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
        config_path = root / "config.json"
        config = json.loads(config_path.read_text())
        _merge_config(config, self.config)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
