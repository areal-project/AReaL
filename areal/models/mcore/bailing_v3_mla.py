# SPDX-License-Identifier: Apache-2.0

"""Gated MLA self-attention for BailingMoeV3 (Ling V3).

v3 enables *head-wise gated attention* on top of standard MLA: a per-head sigmoid gate
(produced by a ``hidden_size -> num_attention_heads`` projection from the layer input)
scales the core attention output before the output projection.

    gate = sigmoid(linear_gate(hidden_states))           # [s, b, num_heads]
    core_attn_out = core_attn_out.view(s, b, H, d) * gate[:, :, :, None]
    output = linear_proj(core_attn_out)

Reference: ant-megatron ``attention.py::apply_gated_attention_linear_gate`` with
``gated_attention_proj_granularity_type=head_wise`` and
``gated_attention_input_tensor_type=linear_qkv_input``.

Implementation note:
    Rather than copy mcore's (version-specific, monolithic) ``MLASelfAttention.forward``,
    we inject the gate via a ``forward_pre_hook`` on ``linear_proj``. This only relies on
    the MLA invariant that the forward ends with ``self.linear_proj(core_attn_out)`` where
    ``core_attn_out`` is ``[s, b, num_heads_local * v_head_dim]`` (head-major), so it is
    robust across megatron-core versions.
"""

import torch
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.spec_utils import build_module

from areal.utils import logging

logger = logging.getLogger("BailingV3MLA")

try:
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
except ImportError:  # pragma: no cover
    from megatron.core.tensor_parallel import (
        ColumnParallelLinear as TEColumnParallelLinear,
    )


class BailingV3MLASelfAttention(MLASelfAttention):
    """MLA self-attention with v3 head-wise gated attention.

    Built exactly like mcore ``MLASelfAttention`` (same submodules / params) plus an
    extra ``linear_gate`` projection. The gate is applied to the core attention output
    just before ``linear_proj`` via a forward pre-hook.
    """

    def __init__(
        self,
        config,
        submodules,
        layer_number,
        attn_mask_type=None,
        **kwargs,
    ):
        if attn_mask_type is not None:
            super().__init__(
                config=config,
                submodules=submodules,
                layer_number=layer_number,
                attn_mask_type=attn_mask_type,
                **kwargs,
            )
        else:
            super().__init__(
                config=config,
                submodules=submodules,
                layer_number=layer_number,
                **kwargs,
            )

        # Head-wise gate: hidden_size -> num_attention_heads (column-parallel; the head
        # dimension is TP-sharded so the local output matches core_attn_out's local heads).
        self.linear_gate = build_module(
            TEColumnParallelLinear,
            config.hidden_size,
            config.num_attention_heads,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="gate",
        )

        self._gate_hidden_states = None
        # Apply the gate to linear_proj's input. with_kwargs defaults to False, so the
        # hook receives positional args and its return value replaces them.
        self.linear_proj.register_forward_pre_hook(self._apply_gate_pre_hook)

    def _apply_gate_pre_hook(self, module, args):
        """Scale ``core_attn_out`` (linear_proj input) head-wise by ``sigmoid(gate)``."""
        if self._gate_hidden_states is None or not args:
            return None
        core_attn_out = args[0]
        gate, _ = self.linear_gate(self._gate_hidden_states)
        gate = torch.sigmoid(gate.float()).type_as(core_attn_out)
        seq_len, batch = core_attn_out.shape[:2]
        num_heads_local = gate.shape[-1]
        # core_attn_out must be head-major [s, b, num_heads_local * v_head_dim] for the
        # per-head gate to align (the invariant this hook relies on).
        assert core_attn_out.shape[-1] % num_heads_local == 0, (
            f"core_attn_out last dim {core_attn_out.shape[-1]} not divisible by "
            f"num_heads_local {num_heads_local}; gated-MLA head-major assumption broken."
        )
        gated = (
            core_attn_out.view(seq_len, batch, num_heads_local, -1)
            * gate[:, :, :, None]
        )
        gated = gated.reshape(core_attn_out.shape)
        return (gated,) + tuple(args[1:])

    def forward(self, hidden_states, *args, **kwargs):
        # Stash the layer input so the linear_proj pre-hook can compute the gate.
        self._gate_hidden_states = hidden_states
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._gate_hidden_states = None
