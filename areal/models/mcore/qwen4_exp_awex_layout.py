# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp bridge GDN rows to AWEX's inference-rank-major representation.

Inputs are full tensors gathered along dimension zero in training TP rank
order. They contain whole key-head groups, not rank-local Q/K/V categories.
Outputs retain the full dimension zero; the AWEX writer subsequently takes
its training-rank slice and the transfer plan redistributes inference slices.
"""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class Qwen4ExpGDNLayout:
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int

    def __post_init__(self) -> None:
        for value in (
            self.num_key_heads,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
        ):
            if value <= 0:
                raise ValueError("GDN head counts and dimensions must be positive")
        if self.num_value_heads % self.num_key_heads:
            raise ValueError("GDN value heads must divide into whole key-head groups")

    def _categories(
        self,
        parameter: torch.Tensor,
        train_tp_size: int,
        infer_tp_size: int,
        *,
        component: Literal["input", "conv", "qkvz", "ba"],
    ) -> tuple[torch.Tensor, ...]:
        for size in (train_tp_size, infer_tp_size):
            if size <= 0 or self.num_key_heads % size:
                raise ValueError("GDN TP sizes must divide the key-head count")
        ratio = self.num_value_heads // self.num_key_heads
        value_width = ratio * self.value_head_dim
        qkv = (self.key_head_dim, self.key_head_dim, value_width)
        dimensions = {
            "input": (*qkv, value_width, ratio, ratio),
            "conv": qkv,
            "qkvz": (*qkv, value_width),
            "ba": (ratio, ratio),
        }
        if component not in dimensions:
            raise ValueError(f"Unknown GDN component: {component}")
        widths = dimensions[component]
        expected_rows = self.num_key_heads * sum(widths)
        if parameter.ndim < 2 or parameter.shape[0] != expected_rows:
            raise ValueError(
                f"Expected full GDN tensor with {expected_rows} rows; "
                f"got {tuple(parameter.shape)}"
            )
        grouped = parameter.reshape(
            self.num_key_heads, sum(widths), *parameter.shape[1:]
        )
        return tuple(
            part.reshape(-1, *parameter.shape[1:])
            for part in grouped.split(widths, dim=1)
        )

    @staticmethod
    def _pack(categories: tuple[torch.Tensor, ...], infer_tp_size: int) -> torch.Tensor:
        shards = [category.chunk(infer_tp_size, dim=0) for category in categories]
        return torch.cat(
            [
                torch.cat([parts[rank] for parts in shards], dim=0)
                for rank in range(infer_tp_size)
            ],
            dim=0,
        ).contiguous()

    def pack_input(
        self,
        parameter: torch.Tensor,
        train_tp_size: int,
        infer_tp_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        categories = self._categories(
            parameter, train_tp_size, infer_tp_size, component="input"
        )
        return (
            self._pack(categories[:4], infer_tp_size),
            self._pack(categories[4:], infer_tp_size),
        )

    def pack_conv(
        self,
        parameter: torch.Tensor,
        train_tp_size: int,
        infer_tp_size: int,
    ) -> torch.Tensor:
        categories = self._categories(
            parameter, train_tp_size, infer_tp_size, component="conv"
        )
        return self._pack(categories, infer_tp_size)

    def pack_decoupled(
        self,
        parameter: torch.Tensor,
        train_tp_size: int,
        infer_tp_size: int,
        component: Literal["qkvz", "ba"],
    ) -> torch.Tensor:
        if component not in ("qkvz", "ba"):
            raise ValueError(f"Expected a decoupled GDN component, got {component}")
        categories = self._categories(
            parameter, train_tp_size, infer_tp_size, component=component
        )
        return self._pack(categories, infer_tp_size)


def pack_qwen4_exp_gated_qkv(
    parameter: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    infer_tp_size: int,
) -> torch.Tensor:
    """Preserve bridge Q/gate head interleaving and replicate inference KV heads."""
    if min(num_heads, num_kv_heads, head_dim, infer_tp_size) <= 0:
        raise ValueError("Attention geometry and TP size must be positive")
    if num_heads % num_kv_heads or num_heads % infer_tp_size:
        raise ValueError("Query heads must divide into KV groups and TP shards")
    if max(num_kv_heads, infer_tp_size) % min(num_kv_heads, infer_tp_size):
        raise ValueError("KV heads and TP size must divide one another")
    query_rows = 2 * (num_heads // num_kv_heads) * head_dim
    rows = query_rows + 2 * head_dim
    if parameter.ndim < 1 or parameter.shape[0] != num_kv_heads * rows:
        raise ValueError("Unexpected full Qwen4Exp gated QKV tensor shape")
    tail = parameter.shape[1:]
    groups = parameter.reshape(num_kv_heads, rows, *tail)
    # Bridge directly concatenates HF q_proj (already Q/gate interleaved), K, V.
    query = groups[:, :query_rows].reshape(num_heads, 2 * head_dim, *tail)
    key = groups[:, query_rows : query_rows + head_dim]
    value = groups[:, query_rows + head_dim :]
    query_parts = query.chunk(infer_tp_size, dim=0)
    if infer_tp_size >= num_kv_heads:
        replicas = infer_tp_size // num_kv_heads
        key_parts = [key[rank // replicas] for rank in range(infer_tp_size)]
        value_parts = [value[rank // replicas] for rank in range(infer_tp_size)]
    else:
        key_parts = key.chunk(infer_tp_size, dim=0)
        value_parts = value.chunk(infer_tp_size, dim=0)
    return torch.cat(
        [
            torch.cat([part.reshape(-1, *tail) for part in parts], dim=0)
            for parts in zip(query_parts, key_parts, value_parts, strict=True)
        ],
        dim=0,
    ).contiguous()
