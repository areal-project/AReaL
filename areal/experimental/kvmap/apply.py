# SPDX-License-Identifier: Apache-2.0
"""Translate a source model's dense KV cache into the target model's cache with a fitted artifact.

For each target layer ``l`` with selected source layers ``S_l``:
    keys:   Y_K = pack(remove_rotary(K_s[i]) for i in S_l) @ W_K[l] + b_K[l], then apply target rotary
    values: Y_V = pack(V_s[i] for i in S_l) @ W_V[l] + b_V[l]
Positions are an input: the same token positions are used for removing the source rotation
and applying the target one. Contract: caches are lists over layers of
``[batch, kv_heads, tokens, head_dim]``; ``positions`` is ``[batch, tokens]``.
"""

from dataclasses import dataclass

import torch

from areal.experimental.kvmap.artifact import MapperArtifact
from areal.experimental.kvmap.contract import pack_features, unpack_target
from areal.experimental.kvmap.rope import apply_rotary, remove_rotary, rotary_cos_sin

COMPUTE_DTYPE = torch.float32  # bf16 caches are widened once; the map itself is float32


@dataclass(frozen=True, kw_only=True)
class DenseCache:
    """Per-layer keys and values of one model for one batch of sequences."""

    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values) or len(self.keys) == 0:
            raise ValueError(
                f"keys and values must be non-empty lists of equal length, got {len(self.keys)} and {len(self.values)}"
            )
        shape = tuple(self.keys[0].shape)
        for layer, (k, v) in enumerate(zip(self.keys, self.values)):
            if k.ndim != 4 or tuple(k.shape) != shape or tuple(v.shape) != shape:
                raise ValueError(
                    f"layer {layer}: keys {tuple(k.shape)} and values {tuple(v.shape)} must all be {shape} = [batch, kv_heads, tokens, head_dim]"
                )

    @property
    def batch(self) -> int:
        return int(self.keys[0].shape[0])

    @property
    def tokens(self) -> int:
        return int(self.keys[0].shape[2])


def translate_cache(
    *,
    source: DenseCache,
    positions: torch.Tensor,
    artifact: MapperArtifact,
    output_dtype: torch.dtype,
) -> DenseCache:
    """Return the target cache predicted from ``source`` at ``positions``."""
    contract = artifact.contract
    _assert_source_layout(source, artifact)
    if tuple(positions.shape) != (source.batch, source.tokens):
        raise ValueError(
            f"positions {tuple(positions.shape)} must be [batch, tokens] = {(source.batch, source.tokens)}"
        )
    device = source.keys[0].device
    source_cos, source_sin = rotary_cos_sin(
        positions,
        head_dim=artifact.source.head_dim,
        theta=artifact.source.rope_theta,
        dtype=COMPUTE_DTYPE,
    )
    target_cos, target_sin = rotary_cos_sin(
        positions,
        head_dim=artifact.target.head_dim,
        theta=artifact.target.rope_theta,
        dtype=COMPUTE_DTYPE,
    )
    content_keys = [
        remove_rotary(k.to(COMPUTE_DTYPE), source_cos, source_sin) for k in source.keys
    ]
    values = [v.to(COMPUTE_DTYPE) for v in source.values]
    target_keys, target_values = [], []
    for layer in range(contract.target_num_layers):
        selected = contract.source_layers_by_target[layer]
        key_rows = pack_features(content_keys, selected) @ artifact.key_weights[
            layer
        ].to(device, COMPUTE_DTYPE) + artifact.key_biases[layer].to(
            device, COMPUTE_DTYPE
        )
        value_rows = pack_features(values, selected) @ artifact.value_weights[layer].to(
            device, COMPUTE_DTYPE
        ) + artifact.value_biases[layer].to(device, COMPUTE_DTYPE)
        unpack = dict(
            batch=source.batch,
            tokens=source.tokens,
            kv_heads=contract.target_kv_heads,
            head_dim=contract.target_head_dim,
        )
        target_keys.append(
            apply_rotary(unpack_target(key_rows, **unpack), target_cos, target_sin).to(
                output_dtype
            )
        )
        target_values.append(unpack_target(value_rows, **unpack).to(output_dtype))
    return DenseCache(keys=tuple(target_keys), values=tuple(target_values))


def _assert_source_layout(source: DenseCache, artifact: MapperArtifact) -> None:
    contract = artifact.contract
    if len(source.keys) != contract.source_num_layers:
        raise ValueError(
            f"source cache has {len(source.keys)} layers, artifact expects {contract.source_num_layers}"
        )
    _, kv_heads, _, head_dim = source.keys[0].shape
    if (kv_heads, head_dim) != (contract.source_kv_heads, contract.source_head_dim):
        raise ValueError(
            f"source cache has {kv_heads} kv heads of dim {head_dim}, artifact expects {contract.source_kv_heads} of dim {contract.source_head_dim}"
        )
