# SPDX-License-Identifier: Apache-2.0
"""Exact metadata boundaries for Qwen4Exp immutable inference state.

This declaration does not prove tensor values or lifecycle preservation. Runtime
integration must additionally bind the checkpoint evidence and visual backup,
validate original parameters before each exchange, and use the same declaration
for metadata and payload converters. No automatic registration or exclusion is
performed here.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn


@dataclass(frozen=True)
class Qwen4ExpFrozenContract:
    checkpoint_manifest_sha256: str
    ple_table_names: frozenset[str]
    visual_parameter_names: frozenset[str]
    language_model_only: bool
    freeze_ple_table: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported Qwen4Exp frozen contract schema")
        if self.language_model_only is not True or self.freeze_ple_table is not True:
            raise ValueError("Frozen exclusions require language-only and frozen PLE")
        if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_manifest_sha256):
            raise ValueError("Expected a SHA256 checkpoint manifest identity")
        for names in (self.ple_table_names, self.visual_parameter_names):
            if not isinstance(names, frozenset) or not names:
                raise ValueError("Frozen parameter names must be nonempty frozen sets")
        for name in self.ple_table_names:
            if not re.fullmatch(
                r"model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding\.weight",
                name,
            ):
                raise ValueError(f"Invalid frozen PLE table name: {name}")
        for name in self.visual_parameter_names:
            if not re.fullmatch(
                r"model\.visual\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", name
            ):
                raise ValueError(f"Invalid frozen visual parameter name: {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "language_model_only": self.language_model_only,
            "freeze_ple_table": self.freeze_ple_table,
            "ple_table_names": sorted(self.ple_table_names),
            "visual_parameter_names": sorted(self.visual_parameter_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Qwen4ExpFrozenContract":
        data = dict(payload)
        expected = {
            "schema_version",
            "checkpoint_manifest_sha256",
            "language_model_only",
            "freeze_ple_table",
            "ple_table_names",
            "visual_parameter_names",
        }
        if data.keys() != expected:
            raise ValueError("Missing or unexpected Qwen4Exp frozen contract fields")
        for key in ("ple_table_names", "visual_parameter_names"):
            names = data[key]
            if not isinstance(names, list) or not all(
                isinstance(n, str) for n in names
            ):
                raise ValueError(f"Expected a string list for {key}")
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate names in {key}")
            data[key] = frozenset(names)
        return cls(**data)

    def excludes(self, name: str, side: Literal["actor", "inference"]) -> bool:
        if side not in ("actor", "inference"):
            raise ValueError(f"Unknown contract side: {side}")
        return name in self.ple_table_names or (
            side == "inference" and name in self.visual_parameter_names
        )

    @staticmethod
    def _validate_table(name: str, parameter: nn.Parameter) -> None:
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(
                f"Validate original Parameter objects, not detached tensors: {name}"
            )
        if parameter.requires_grad:
            raise ValueError(f"PLE table is trainable: {name}")
        if parameter.dtype != torch.bfloat16 or parameter.ndim != 2:
            raise ValueError(f"Expected the validated BF16 PLE table layout: {name}")

    def validate_actor_parameters(
        self,
        parameters: Mapping[str, nn.Parameter],
        local_table_names: frozenset[str],
    ) -> None:
        """Validate canonical original parameters on this PP stage, before detach.

        The caller must verify global PP ownership coverage separately; a PP stage
        without PLE legitimately has an empty local table set.
        """
        if not local_table_names <= self.ple_table_names:
            raise ValueError("Local PLE ownership is outside the frozen contract")
        if any(name.startswith("model.visual.") for name in parameters):
            raise ValueError(
                "Language-only actor unexpectedly contains visual parameters"
            )
        observed = {name for name in parameters if ".ple_embedding." in name}
        if observed != local_table_names:
            raise ValueError(
                "Actor PLE parameters do not match declared local ownership"
            )
        for name in local_table_names:
            self._validate_table(name, parameters[name])

    def validate_inference_parameters(
        self,
        parameters: Mapping[str, nn.Parameter],
        preserved_visual_names: frozenset[str],
    ) -> None:
        """Require exact exclusions and the same keys in the visual backup path."""
        observed_visual = {n for n in parameters if n.startswith("model.visual.")}
        if observed_visual != self.visual_parameter_names:
            raise ValueError(
                "Inference visual parameters differ from the frozen contract"
            )
        if preserved_visual_names != self.visual_parameter_names:
            raise ValueError("Visual preservation keys differ from transfer exclusions")
        observed_tables = {n for n in parameters if ".ple_embedding." in n}
        if observed_tables != self.ple_table_names:
            raise ValueError("Inference PLE parameters differ from the frozen contract")
        for name in self.ple_table_names:
            self._validate_table(name, parameters[name])
            if parameters[name].device.type != "cpu":
                raise ValueError(f"Expected the validated CPU PLE residency: {name}")


def load_frozen_contract(
    manifest_path: Path, model_directory: Path
) -> Qwen4ExpFrozenContract:
    """Check declared identity and exact names against checkpoint/evidence metadata.

    This validates config/index bytes and the frozen-state evidence manifest.
    It does not read all live tensor values or authorize a runtime dependency.
    """
    manifest = json.loads(manifest_path.read_text())
    contract = Qwen4ExpFrozenContract.from_dict(manifest["contract"])
    basis = manifest["identity_basis"]
    encoded = json.dumps(basis, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != contract.checkpoint_manifest_sha256:
        raise ValueError("Frozen contract evidence identity changed")
    config = (model_directory / "config.json").read_bytes()
    index = (model_directory / "model.safetensors.index.json").read_bytes()
    if hashlib.sha256(config).hexdigest() != basis["config_sha256"]:
        raise ValueError("Checkpoint config differs from the frozen contract")
    if hashlib.sha256(index).hexdigest() != basis["weight_index_sha256"]:
        raise ValueError("Checkpoint weight index differs from the frozen contract")
    if json.loads(config).get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise ValueError("Checkpoint is not the declared Qwen4Exp architecture")
    table_names = set()
    source_names = set()
    for shard in basis["ple_source_shards"]:
        name = shard["name"]
        match = re.fullmatch(
            r"model\.language_model\.layers\.(\d+)\.ple\.ple_embedding"
            r"\.ngram_embedding\.shard_\d+\.weight",
            name,
        )
        if match is None or name in source_names:
            raise ValueError("Invalid or duplicate PLE source shard name")
        source_names.add(name)
        table_names.add(
            f"model.layers.{match[1]}.ple.ple_embedding.ngram_embedding.weight"
        )
    if table_names != contract.ple_table_names:
        raise ValueError("PLE exclusions differ from the source evidence")
    if not basis["visual_reference"]:
        raise ValueError("Missing visual preservation evidence")
    for reference in basis["visual_reference"]:
        names = [parameter["name"] for parameter in reference["parameters"]]
        if (
            len(names) != len(set(names))
            or set(names) != contract.visual_parameter_names
        ):
            raise ValueError("Visual exclusions differ from the preservation evidence")
    return contract
