# SPDX-License-Identifier: Apache-2.0
"""Fix which source layers feed each target layer and how their heads pack into one feature row.

Packing order for one target layer with selected source layers ``(l_1, ..., l_k)``:
source layer major, then KV head, then head dimension, so a feature row has width
``k * source_kv_heads * source_head_dim``. Rows are batch major, then token. This order is
part of a mapper artifact's identity; changing it invalidates every fitted weight.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, kw_only=True)
class FeatureContract:
    """Source-layer selection and head layout shared by fitting and application."""

    source_layers_by_target: tuple[tuple[int, ...], ...]
    source_num_layers: int
    source_kv_heads: int
    source_head_dim: int
    target_kv_heads: int
    target_head_dim: int

    def __post_init__(self) -> None:
        if len(self.source_layers_by_target) == 0:
            raise ValueError("contract needs at least one target layer")
        for target_layer, sources in enumerate(self.source_layers_by_target):
            if len(sources) == 0:
                raise ValueError(f"target layer {target_layer} selects no source layer")
            if len(set(sources)) != len(sources):
                raise ValueError(
                    f"target layer {target_layer} selects a source layer twice: {sources}"
                )
            bad = [s for s in sources if s < 0 or s >= self.source_num_layers]
            if len(bad) != 0:
                raise ValueError(
                    f"target layer {target_layer} selects source layers {bad} outside "
                    f"[0, {self.source_num_layers})"
                )
        for name in (
            "source_num_layers",
            "source_kv_heads",
            "source_head_dim",
            "target_kv_heads",
            "target_head_dim",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")

    @property
    def target_num_layers(self) -> int:
        return len(self.source_layers_by_target)

    @property
    def target_dim(self) -> int:
        return self.target_kv_heads * self.target_head_dim

    def feature_dim(self, target_layer: int) -> int:
        return (
            len(self.source_layers_by_target[target_layer])
            * self.source_kv_heads
            * self.source_head_dim
        )

    @staticmethod
    def same_layer(
        *, num_layers: int, kv_heads: int, head_dim: int
    ) -> "FeatureContract":
        """The same-architecture variant: target layer ``l`` reads only source layer ``l``."""
        return FeatureContract(
            source_layers_by_target=tuple((layer,) for layer in range(num_layers)),
            source_num_layers=num_layers,
            source_kv_heads=kv_heads,
            source_head_dim=head_dim,
            target_kv_heads=kv_heads,
            target_head_dim=head_dim,
        )

    def to_json(self) -> dict:
        return {
            "source_layers_by_target": [list(s) for s in self.source_layers_by_target],
            "source_num_layers": self.source_num_layers,
            "source_kv_heads": self.source_kv_heads,
            "source_head_dim": self.source_head_dim,
            "target_kv_heads": self.target_kv_heads,
            "target_head_dim": self.target_head_dim,
        }

    @staticmethod
    def from_json(raw: dict) -> "FeatureContract":
        expected = {
            "source_layers_by_target",
            "source_num_layers",
            "source_kv_heads",
            "source_head_dim",
            "target_kv_heads",
            "target_head_dim",
        }
        if set(raw) != expected:
            raise ValueError(
                f"contract keys {sorted(raw)} must be exactly {sorted(expected)}"
            )
        return FeatureContract(
            source_layers_by_target=tuple(
                tuple(int(s) for s in sources)
                for sources in raw["source_layers_by_target"]
            ),
            source_num_layers=int(raw["source_num_layers"]),
            source_kv_heads=int(raw["source_kv_heads"]),
            source_head_dim=int(raw["source_head_dim"]),
            target_kv_heads=int(raw["target_kv_heads"]),
            target_head_dim=int(raw["target_head_dim"]),
        )


def pack_features(
    layers: Sequence[torch.Tensor], source_layers: Sequence[int]
) -> torch.Tensor:
    """Pack ``[batch, kv_heads, tokens, d]`` tensors of the selected layers into ``[batch * tokens, k * kv_heads * d]``."""
    if len(source_layers) == 0:
        raise ValueError("pack_features needs at least one source layer")
    pieces = []
    for layer in source_layers:
        x = layers[layer]
        if x.ndim != 4:
            raise ValueError(
                f"layer {layer} must be [batch, kv_heads, tokens, d], got {tuple(x.shape)}"
            )
        batch, kv_heads, tokens, head_dim = x.shape
        pieces.append(
            x.permute(0, 2, 1, 3).reshape(batch * tokens, kv_heads * head_dim)
        )
    return torch.cat(pieces, dim=1)


def unpack_target(
    rows: torch.Tensor, *, batch: int, tokens: int, kv_heads: int, head_dim: int
) -> torch.Tensor:
    """Inverse of the target-side packing: ``[batch * tokens, kv_heads * d]`` to ``[batch, kv_heads, tokens, d]``."""
    if rows.shape != (batch * tokens, kv_heads * head_dim):
        raise ValueError(
            f"rows {tuple(rows.shape)} do not match batch {batch}, tokens {tokens}, kv_heads {kv_heads}, d {head_dim}"
        )
    return (
        rows.reshape(batch, tokens, kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    )
