# SPDX-License-Identifier: Apache-2.0
"""Store a fitted KV mapper with the model identities it is valid for, and refuse any other pair.

A mapper is only meaningful for the exact (source checkpoint, target checkpoint) pair, layout
and rotary parameters it was fitted on; ``validate_pair`` fails on the first field that differs.
On disk an artifact is a directory with ``mapper.json`` (identity, contract, provenance) and
``mapper.safetensors`` (float32 weights), written atomically.
"""

import hashlib
import json
import os
import pathlib
import shutil
import tempfile
from dataclasses import asdict, dataclass

import torch

from areal.experimental.kvmap.contract import FeatureContract

ARTIFACT_FORMAT_VERSION = 1
SUPPORTED_ROPE_TYPE = "default"
META_FILE = "mapper.json"
TENSOR_FILE = "mapper.safetensors"
META_KEYS = frozenset(
    {
        "format_version",
        "source",
        "target",
        "contract",
        "lambda_",
        "diagnostics",
        "provenance",
    }
)


class MapperMismatchError(ValueError):
    """The artifact does not describe the requested source/target pair."""


@dataclass(frozen=True, kw_only=True)
class ModelIdentity:
    """Everything a cache layout depends on, plus the exact checkpoint digest."""

    checkpoint_sha256: str
    architecture: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    rope_type: str
    cache_dtype: str

    def __post_init__(self) -> None:
        if len(self.checkpoint_sha256) != 64:
            raise ValueError(
                f"checkpoint_sha256 must be a 64-hex digest, got {self.checkpoint_sha256!r}"
            )
        if self.rope_type != SUPPORTED_ROPE_TYPE:
            raise ValueError(
                f"rope_type {self.rope_type!r} is unsupported; this mapper supports {SUPPORTED_ROPE_TYPE!r} only"
            )

    @staticmethod
    def from_json(raw: dict) -> "ModelIdentity":
        expected = set(ModelIdentity.__dataclass_fields__)
        if set(raw) != expected:
            raise ValueError(
                f"identity keys {sorted(raw)} must be exactly {sorted(expected)}"
            )
        return ModelIdentity(**raw)


@dataclass(frozen=True, kw_only=True)
class MapperArtifact:
    """A fitted mapper: per target layer, affine maps for keys and for values."""

    source: ModelIdentity
    target: ModelIdentity
    contract: FeatureContract
    lambda_: float
    key_weights: tuple[torch.Tensor, ...]
    key_biases: tuple[torch.Tensor, ...]
    value_weights: tuple[torch.Tensor, ...]
    value_biases: tuple[torch.Tensor, ...]
    diagnostics: dict
    provenance: dict

    def __post_init__(self) -> None:
        layers = self.contract.target_num_layers
        for name in ("key_weights", "key_biases", "value_weights", "value_biases"):
            if len(getattr(self, name)) != layers:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} entries, contract has {layers} target layers"
                )
        for layer in range(layers):
            expected_weight = (
                self.contract.feature_dim(layer),
                self.contract.target_dim,
            )
            for weights, biases, kind in (
                (self.key_weights, self.key_biases, "key"),
                (self.value_weights, self.value_biases, "value"),
            ):
                if tuple(weights[layer].shape) != expected_weight:
                    raise ValueError(
                        f"{kind} weight for target layer {layer} has shape {tuple(weights[layer].shape)}, contract requires {expected_weight}"
                    )
                if tuple(biases[layer].shape) != (self.contract.target_dim,):
                    raise ValueError(
                        f"{kind} bias for target layer {layer} has shape {tuple(biases[layer].shape)}, contract requires ({self.contract.target_dim},)"
                    )
        if self.target.num_layers != layers:
            raise ValueError(
                f"target identity has {self.target.num_layers} layers, contract has {layers}"
            )
        if self.source.num_layers != self.contract.source_num_layers:
            raise ValueError(
                f"source identity has {self.source.num_layers} layers, contract expects {self.contract.source_num_layers}"
            )

    def validate_pair(self, *, source: ModelIdentity, target: ModelIdentity) -> None:
        """Raise ``MapperMismatchError`` naming the first differing field."""
        for role, expected, actual in (
            ("source", self.source, source),
            ("target", self.target, target),
        ):
            for field, value in asdict(expected).items():
                if getattr(actual, field) != value:
                    raise MapperMismatchError(
                        f"mapper {role}.{field} is {value!r} but the requested {role} has {getattr(actual, field)!r}; "
                        "fit a mapper for this exact pair or fall back to an exact rebuild"
                    )

    def save(self, directory: pathlib.Path) -> str:
        """Write the artifact atomically and return its sha256 over both files."""
        from safetensors.torch import save_file

        directory = pathlib.Path(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            raise FileExistsError(
                f"{directory} exists; artifacts are immutable, choose a new directory"
            )
        tensors = {}
        for layer in range(self.contract.target_num_layers):
            # clone: safetensors refuses tensors that share storage, and the identity artifact does
            tensors[f"key_weight.{layer}"] = (
                self.key_weights[layer].to(torch.float32).clone()
            )
            tensors[f"key_bias.{layer}"] = (
                self.key_biases[layer].to(torch.float32).clone()
            )
            tensors[f"value_weight.{layer}"] = (
                self.value_weights[layer].to(torch.float32).clone()
            )
            tensors[f"value_bias.{layer}"] = (
                self.value_biases[layer].to(torch.float32).clone()
            )
        meta = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "source": asdict(self.source),
            "target": asdict(self.target),
            "contract": self.contract.to_json(),
            "lambda_": self.lambda_,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
        }
        staging = pathlib.Path(
            tempfile.mkdtemp(prefix=directory.name + ".partial-", dir=directory.parent)
        )
        try:
            save_file(tensors, str(staging / TENSOR_FILE))
            (staging / META_FILE).write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n"
            )
            os.rename(staging, directory)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return artifact_sha256(directory)

    @staticmethod
    def load(directory: pathlib.Path) -> "MapperArtifact":
        """Read an artifact, rejecting unknown keys and unsupported format versions."""
        from safetensors.torch import load_file

        directory = pathlib.Path(directory)
        meta = json.loads((directory / META_FILE).read_text())
        if set(meta) != META_KEYS:
            raise ValueError(
                f"{directory / META_FILE} keys {sorted(meta)} must be exactly {sorted(META_KEYS)}"
            )
        if meta["format_version"] != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {meta['format_version']} is not {ARTIFACT_FORMAT_VERSION}; refit or convert it"
            )
        contract = FeatureContract.from_json(meta["contract"])
        tensors = load_file(str(directory / TENSOR_FILE))
        layers = range(contract.target_num_layers)
        return MapperArtifact(
            source=ModelIdentity.from_json(meta["source"]),
            target=ModelIdentity.from_json(meta["target"]),
            contract=contract,
            lambda_=float(meta["lambda_"]),
            key_weights=tuple(tensors[f"key_weight.{layer}"] for layer in layers),
            key_biases=tuple(tensors[f"key_bias.{layer}"] for layer in layers),
            value_weights=tuple(tensors[f"value_weight.{layer}"] for layer in layers),
            value_biases=tuple(tensors[f"value_bias.{layer}"] for layer in layers),
            diagnostics=meta["diagnostics"],
            provenance=meta["provenance"],
        )


def identity_artifact(
    *, source: ModelIdentity, target: ModelIdentity, provenance: dict
) -> MapperArtifact:
    """The same-layer mapper with unit weights and zero biases; the identity-bypass control."""
    if (source.num_layers, source.num_kv_heads, source.head_dim) != (
        target.num_layers,
        target.num_kv_heads,
        target.head_dim,
    ):
        raise ValueError(
            "identity_artifact requires identical layer count, kv heads and head dim"
        )
    contract = FeatureContract.same_layer(
        num_layers=source.num_layers,
        kv_heads=source.num_kv_heads,
        head_dim=source.head_dim,
    )
    eye = torch.eye(contract.target_dim, dtype=torch.float32)
    zero = torch.zeros(contract.target_dim, dtype=torch.float32)
    layers = contract.target_num_layers
    return MapperArtifact(
        source=source,
        target=target,
        contract=contract,
        lambda_=0.0,
        key_weights=(eye,) * layers,
        key_biases=(zero,) * layers,
        value_weights=(eye,) * layers,
        value_biases=(zero,) * layers,
        diagnostics={"kind": "identity"},
        provenance=provenance,
    )


def artifact_sha256(directory: pathlib.Path) -> str:
    """Digest of the two artifact files in a fixed order."""
    digest = hashlib.sha256()
    for name in (META_FILE, TENSOR_FILE):
        digest.update(name.encode())
        digest.update((pathlib.Path(directory) / name).read_bytes())
    return digest.hexdigest()
