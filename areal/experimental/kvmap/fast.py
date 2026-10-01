# SPDX-License-Identifier: Apache-2.0
"""Apply a same-layer mapper to all layers and any number of tokens in one batched multiply.

For every target layer ``l`` and token ``n`` at position ``p_n`` (formulas as in ``apply.py``):
    Y_K[l, n] = (remove_rotary(K[l, n], p_n) as a row of width kv_heads * head_dim) @ W_K[l] + b_K[l]
    K'[l, n]  = apply_rotary(Y_K[l, n], p_n)
    V'[l, n]  = V[l, n] @ W_V[l] + b_V[l]
computed with two ``torch.baddbmm`` calls over the stacked layer axis instead of a Python loop over
layers. Contract: ``keys`` and ``values`` are ``[layers, tokens, kv_heads, head_dim]`` (the layout of a
paged KV pool gathered at ``tokens`` slots); ``positions`` is ``[tokens]`` and may mix several
sequences. Only same-layer artifacts are supported here; others must use ``apply.translate_cache``.
"""

import torch

from areal.experimental.kvmap.artifact import MapperArtifact
from areal.experimental.kvmap.rope import rotate_half

INITIAL_POSITIONS = 4096


class StackedMapper:
    """A mapper artifact prepared once on a device for batched application."""

    def __init__(self, artifact: MapperArtifact, *, device: torch.device | str):
        contract = artifact.contract
        same_layer = tuple((layer,) for layer in range(contract.target_num_layers))
        if contract.source_layers_by_target != same_layer:
            raise ValueError(
                "StackedMapper supports same-layer artifacts only; use apply.translate_cache "
                "for artifacts that read several source layers"
            )
        if (contract.source_kv_heads, contract.source_head_dim) != (
            contract.target_kv_heads,
            contract.target_head_dim,
        ):
            raise ValueError("StackedMapper requires equal source and target head layout")
        self.layers = contract.target_num_layers
        self.kv_heads = contract.target_kv_heads
        self.head_dim = contract.target_head_dim
        self.width = self.kv_heads * self.head_dim
        self.source_theta = artifact.source.rope_theta
        self.target_theta = artifact.target.rope_theta
        self.device = torch.device(device)
        f32 = dict(device=self.device, dtype=torch.float32)
        self.key_weight = torch.stack(artifact.key_weights).to(**f32)  # [L, F, D]
        self.value_weight = torch.stack(artifact.value_weights).to(**f32)
        self.key_bias = torch.stack(artifact.key_biases).to(**f32).unsqueeze(1)  # [L, 1, D]
        self.value_bias = torch.stack(artifact.value_biases).to(**f32).unsqueeze(1)
        self._table_positions = 0
        self._ensure_tables(INITIAL_POSITIONS)

    def _ensure_tables(self, positions: int) -> None:
        if positions <= self._table_positions:
            return
        size = max(positions, 2 * self._table_positions)
        index = torch.arange(size, device=self.device, dtype=torch.float32)
        exponent = torch.arange(0, self.head_dim, 2, device=self.device, dtype=torch.float32)

        def tables(theta: float) -> tuple[torch.Tensor, torch.Tensor]:
            freqs = index[:, None] * (1.0 / (theta ** (exponent / self.head_dim)))[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)  # [P, d]
            return emb.cos(), emb.sin()

        self._source_cos, self._source_sin = tables(self.source_theta)
        self._target_cos, self._target_sin = tables(self.target_theta)
        self._table_positions = size

    @torch.no_grad()
    def translate(
        self, *, keys: torch.Tensor, values: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``[layers, tokens, kv_heads, head_dim]`` keys and values; shapes and dtypes are kept."""
        expected = (self.layers, positions.shape[0], self.kv_heads, self.head_dim)
        if tuple(keys.shape) != expected or tuple(values.shape) != expected:
            raise ValueError(
                f"keys {tuple(keys.shape)} and values {tuple(values.shape)} must both be {expected} "
                "= [layers, tokens, kv_heads, head_dim]"
            )
        tokens = positions.shape[0]
        if tokens == 0:
            return keys, values
        self._ensure_tables(int(positions.max()) + 1)
        cos_s = self._source_cos[positions][None, :, None, :]  # [1, N, 1, d]
        sin_s = self._source_sin[positions][None, :, None, :]
        cos_t = self._target_cos[positions][None, :, None, :]
        sin_t = self._target_sin[positions][None, :, None, :]
        k = keys.to(torch.float32)
        content = k * cos_s - rotate_half(k) * sin_s
        mapped = torch.baddbmm(
            self.key_bias, content.reshape(self.layers, tokens, self.width), self.key_weight
        ).reshape(self.layers, tokens, self.kv_heads, self.head_dim)
        new_keys = mapped * cos_t + rotate_half(mapped) * sin_t
        new_values = torch.baddbmm(
            self.value_bias,
            values.to(torch.float32).reshape(self.layers, tokens, self.width),
            self.value_weight,
        ).reshape(self.layers, tokens, self.kv_heads, self.head_dim)
        return new_keys.to(keys.dtype), new_values.to(values.dtype)
