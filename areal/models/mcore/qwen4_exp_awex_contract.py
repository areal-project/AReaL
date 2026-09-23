# SPDX-License-Identifier: Apache-2.0
"""Derive frozen exclusions and verify loaded weights against the checkpoint.

Only the validated BF16, unit-scale PLE layout is supported. Verification streams
checkpoint tensors in bounded CPU chunks; it runs at initialization/recovery, not
on every weight update. Checkpoint paths are deliberately absent from the proof.
"""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from awex.models.qwen4_exp_contract import Qwen4ExpFrozenContract
from safetensors import safe_open
from torch import nn

_PLE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.ple\.ple_embedding"
    r"\.ngram_embedding\.shard_(\d+)\.weight"
)
_CHUNK_BYTES = 16 * 1024 * 1024


class FrozenCheckpoint:
    def __init__(self, directory: Path, language_model_only: bool):
        self.directory = directory
        config = json.loads((directory / "config.json").read_text())
        if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
            raise ValueError("Frozen checkpoint requires Qwen4Exp")
        self.weight_map = json.loads(
            (directory / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.tables: dict[str, list[str]] = {}
        shards: dict[str, dict[int, str]] = {}
        for name in self.weight_map:
            match = _PLE.fullmatch(name)
            if match:
                canonical = (
                    f"model.layers.{match[1]}.ple.ple_embedding.ngram_embedding.weight"
                )
                numbered = shards.setdefault(canonical, {})
                number = int(match[2])
                if number in numbered:
                    raise ValueError(f"Duplicate PLE source shard: {name}")
                numbered[number] = name
        text_config = config.get("text_config", config)
        parts = text_config["split_ngram_parts"]
        if type(parts) is not int or parts < 1:
            raise ValueError("Invalid PLE checkpoint shard count")
        for name, numbered in shards.items():
            if set(numbered) != set(range(parts)):
                raise ValueError(f"Missing PLE source shards: {name}")
            self.tables[name] = [numbered[i] for i in range(parts)]
        self.visual_sources: dict[str, str] = {}
        for name in self.weight_map:
            if name.startswith("model.visual."):
                canonical = name.replace(".attn.qkv.", ".attn.qkv_proj.")
                if canonical in self.visual_sources:
                    raise ValueError(f"Duplicate frozen vision source: {canonical}")
                self.visual_sources[canonical] = name
        visual = frozenset(self.visual_sources)
        identity = json.dumps(
            {"config": config, "weight_map": self.weight_map},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.contract = Qwen4ExpFrozenContract(
            hashlib.sha256(identity).hexdigest(),
            frozenset(self.tables),
            visual,
            language_model_only,
            True,
            schema_version=2,
        )
        self._shapes: dict[str, tuple[int, ...]] = {}

    @property
    def source_names(self) -> set[str]:
        return set(self.contract.visual_parameter_names).union(
            *(set(names) for names in self.tables.values())
        )

    def shape(self, name: str) -> tuple[int, ...]:
        if name not in self._shapes:
            with safe_open(
                self.directory / self.weight_map[self.visual_sources.get(name, name)],
                framework="pt",
                device="cpu",
            ) as source:
                value = source.get_slice(self.visual_sources.get(name, name))
                if value.get_dtype() != "BF16":
                    raise ValueError(f"Frozen checkpoint requires BF16 weights: {name}")
                self._shapes[name] = tuple(value.get_shape())
        return self._shapes[name]

    def table_layout(self, name: str) -> tuple[list[tuple[str, int, int]], int, int]:
        names = self.tables[name]
        shapes = [self.shape(n) for n in names]
        if any(len(s) != 2 or s[1] != shapes[0][1] for s in shapes):
            raise ValueError(f"Invalid PLE source shapes: {name}")
        total = sum(s[0] for s in shapes)
        size = math.ceil(total / len(names))
        if any(s[0] != min(size, total - i * size) for i, s in enumerate(shapes)):
            raise ValueError(f"Unsupported PLE shard layout: {name}")
        scale_name = names[0].replace("shard_0.weight", "weight_scale")
        if scale_name in self.weight_map:
            with safe_open(
                self.directory / self.weight_map[scale_name],
                framework="pt",
                device="cpu",
            ) as source:
                scale = source.get_tensor(scale_name)
                if scale.numel() != 1 or not torch.equal(
                    scale.float().reshape(1), torch.ones(1)
                ):
                    raise ValueError(f"Frozen PLE requires unit scale: {name}")
        return (
            [
                (n, i * size, i * size + s[0])
                for i, (n, s) in enumerate(zip(names, shapes, strict=True))
            ],
            total,
            shapes[0][1],
        )

    def verify_source(
        self,
        name: str,
        parameter: nn.Parameter | None = None,
        segments: list[tuple[int, int, int, int]] | None = None,
    ) -> dict[str, Any]:
        """Hash a source and compare its selected slices with a live parameter.

        Segments are (axis, source_start, source_end, destination_start). Reading
        the whole source shard gives a partition-independent proof; only slices
        owned by this rank are compared. No full table or all-gather is allocated.
        """
        shape = self.shape(name)
        if not shape or not all(d > 0 for d in shape):
            raise ValueError(f"Unsupported frozen tensor shape: {name}")
        if parameter is not None and parameter.dtype != torch.bfloat16:
            raise ValueError(f"Frozen live weight must be BF16: {name}")
        rows = max(1, _CHUNK_BYTES // (2 * math.prod(shape[1:])))
        digest = hashlib.sha256()
        with safe_open(
            self.directory / self.weight_map[self.visual_sources.get(name, name)],
            framework="pt",
            device="cpu",
        ) as source:
            tensor = source.get_slice(self.visual_sources.get(name, name))
            for start in range(0, shape[0], rows):
                end = min(start + rows, shape[0])
                chunk = tensor[start:end]
                digest.update(chunk.contiguous().view(torch.uint8).numpy().tobytes())
                for axis, lo, hi, dest in segments or []:
                    if parameter is None:
                        raise ValueError("A live parameter is required for comparison")
                    if axis == 0:
                        a, b = max(start, lo), min(end, hi)
                        if a >= b:
                            continue
                        expected = chunk[a - start : b - start]
                        actual = parameter.detach()[dest + a - lo : dest + b - lo].cpu()
                    elif axis == 1:
                        expected = chunk[:, lo:hi]
                        actual = parameter.detach()[
                            start:end, dest : dest + hi - lo
                        ].cpu()
                    else:
                        raise ValueError("Unsupported frozen tensor partition axis")
                    if not torch.equal(actual, expected):
                        raise ValueError(
                            f"Frozen weight differs from checkpoint: {name}"
                        )
        return {"shape": list(shape), "sha256": digest.hexdigest()}

    def verify_table(
        self, name: str, embedding: nn.Module, side: str
    ) -> dict[str, Any]:
        layout, total, width = self.table_layout(name)
        parameter = embedding.weight
        if side == "actor":
            start, end = embedding.vocab_start_index, embedding.vocab_end_index
            declared_total = embedding.num_embeddings
        else:
            indices = embedding.shard_indices
            start, end = indices.org_vocab_start_index, indices.org_vocab_end_index
            declared_total = embedding.org_vocab_size
            scale = embedding.weight_scale.detach().float().cpu().reshape(-1)
            if scale.numel() != 1 or not torch.equal(scale, torch.ones(1)):
                raise ValueError(f"Frozen inference PLE requires unit scale: {name}")
        if (
            declared_total != total
            or not 0 <= start < end <= total
            or parameter.ndim != 2
            or parameter.shape[1] != width
            or parameter.shape[0] < end - start
            or parameter.requires_grad
        ):
            raise ValueError(f"Unsupported frozen PLE partition: {name}")
        proof = {}
        for source, lo, hi in layout:
            a, b = max(start, lo), min(end, hi)
            if a < b:
                proof[source] = self.verify_source(
                    source, parameter, [(0, a - lo, b - lo, a - start)]
                )
        return proof


def merge_frozen_proofs(checkpoint: FrozenCheckpoint, records: list[dict]) -> dict:
    """Require every source shard and identical evidence from replica owners."""
    merged = {}
    ranges: dict[str, set[tuple[int, int]]] = {}
    for record in records:
        if record["contract"] != checkpoint.contract.to_dict():
            raise ValueError("Training frozen checkpoint descriptions differ")
        for name, interval in record["ranges"].items():
            ranges.setdefault(name, set()).add(tuple(interval))
        for name, evidence in record["proof"].items():
            if name in merged and merged[name] != evidence:
                raise ValueError(f"Training frozen checkpoint contents differ: {name}")
            merged[name] = evidence
    if merged.keys() != checkpoint.source_names:
        raise ValueError("Training frozen source coverage is incomplete")
    if ranges.keys() != checkpoint.tables.keys():
        raise ValueError("Training frozen PLE ownership is incomplete")
    for name, intervals in ranges.items():
        cursor = 0
        for lo, hi in sorted(intervals):
            if lo != cursor or hi <= lo:
                raise ValueError(
                    f"Invalid or incomplete frozen PLE row ownership: {name}"
                )
            cursor = hi
        if cursor != checkpoint.table_layout(name)[1]:
            raise ValueError(f"Incomplete frozen PLE row ownership: {name}")
    return merged


def check_frozen_proof(actual: dict, expected: dict) -> None:
    for name, value in actual.items():
        if expected.get(name) != value:
            raise ValueError(
                f"Frozen checkpoint content changed or differs between engines: {name}"
            )


def visual_segments(
    module: nn.Module, name: str, shape: tuple[int, ...]
) -> tuple[list, tuple[int, ...]]:
    """Use the actual vision layer's attention-TP layout, never the global TP."""
    kind = type(module).__name__
    if kind in ("ColumnParallelLinear", "RowParallelLinear", "QKVParallelLinear"):
        if getattr(module, "use_presharded_weights", False):
            raise ValueError(f"Unsupported presharded frozen vision weight: {name}")
        rank, size = module.tp_rank, module.tp_size
        if not 0 <= rank < size:
            raise ValueError("Invalid vision TP rank")
        if kind == "QKVParallelLinear":
            if (
                module.total_num_heads != module.total_num_kv_heads
                or module.kv_tp_size != size
                or module.kv_tp_rank != rank
                or module.v_head_size != module.head_size
            ):
                raise ValueError("Unsupported frozen vision QKV layout")
            full = module.total_num_heads * module.head_size
            if shape[0] != 3 * full or full % size:
                raise ValueError("Invalid frozen vision QKV shape")
            part = full // size
            return [
                (0, i * full + rank * part, i * full + (rank + 1) * part, i * part)
                for i in range(3)
            ], (3 * part, *shape[1:])
        if kind == "RowParallelLinear" and name.endswith(".bias"):
            return [(0, 0, shape[0], 0)], shape
        axis = int(kind == "RowParallelLinear")
        if shape[axis] % size:
            raise ValueError("Nondivisible frozen vision TP shape")
        if kind == "ColumnParallelLinear" and len(module.output_partition_sizes) != 1:
            raise ValueError("Unsupported packed frozen vision projection")
        part = shape[axis] // size
        local = list(shape)
        local[axis] = part
        return [(axis, rank * part, (rank + 1) * part, 0)], tuple(local)
    # These model layers are replicated in the supported SGLang loader.
    if kind not in (
        "Conv3d",
        "Conv3dLayer",
        "LayerNorm",
        "Embedding",
        "Linear",
        "RMSNorm",
    ):
        raise ValueError(f"Unsupported frozen vision module: {kind} ({name})")
    return [(0, 0, shape[0], 0)], shape
