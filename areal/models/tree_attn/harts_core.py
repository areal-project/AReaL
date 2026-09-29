# SPDX-License-Identifier: Apache-2.0

"""Differentiable execution of a compact tree GDN plan."""

from collections.abc import Callable

import torch

from areal.models.tree_attn.harts_plan import LinearAttentionPlan


def run_linear_attention_plan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    plan: LinearAttentionPlan,
    kernel: Callable,
) -> torch.Tensor:
    """Run compact tokens once and replay only GDN inputs at branch joins.

    Inputs have a singleton batch dimension. Each public FLA call packs all
    currently ready segments as variable-length sequences. A requested state
    inside a sequence ends a segment, so its final state remains differentiable
    and can be passed into later segments without copying a vendor kernel.
    """
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("tree GDN requires a singleton batch dimension")
    output = value.new_zeros(value.shape)
    states: dict[int, torch.Tensor] = {}
    output_rows: list[int] = []
    output_pieces: list[torch.Tensor] = []
    zero_state = torch.zeros(
        (query.shape[2], key.shape[-1], value.shape[-1]),
        device=query.device,
        dtype=torch.float32,
    )

    pending = []
    for round_sequences in plan.rounds:
        for sequence in round_sequences:
            token_indices = sequence.replay_indices + sequence.compact_indices
            if not token_indices:
                continue
            semantic = set(sequence.compact_indices)
            segments = []
            start = 0
            for end, token in enumerate(token_indices, 1):
                if token in semantic and token in plan.boundary_state_tokens:
                    segments.append((token_indices[start:end], token))
                    start = end
            if start < len(token_indices):
                segments.append((token_indices[start:], None))
            pending.append(
                {
                    "segments": segments,
                    "position": 0,
                    "source": sequence.initial_state_token,
                    "semantic": semantic,
                }
            )

    while pending:
        ready = [
            item
            for item in pending
            if item["source"] is None or item["source"] in states
        ]
        if not ready:
            raise ValueError("linear-attention plan contains a state dependency cycle")
        indices = []
        offsets = [0]
        initial_states = []
        for item in ready:
            segment, _ = item["segments"][item["position"]]
            indices.extend(segment)
            offsets.append(len(indices))
            source = item["source"]
            initial_states.append(zero_state if source is None else states[source])
        rows = torch.tensor(indices, device=query.device, dtype=torch.long)
        cu_seqlens = torch.tensor(offsets, device=query.device, dtype=torch.int32)
        cu_seqlens_cpu = torch.tensor(offsets, dtype=torch.int32)
        packed = [
            tensor.index_select(1, rows) for tensor in (query, key, value, gate, beta)
        ]
        core_output, final_states = kernel(
            *packed,
            initial_state=torch.stack(initial_states),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            use_qk_l2norm_in_kernel=False,
        )
        for seq_idx, item in enumerate(ready):
            segment, boundary_token = item["segments"][item["position"]]
            semantic_positions = [
                offsets[seq_idx] + j
                for j, token in enumerate(segment)
                if token in item["semantic"]
            ]
            if semantic_positions:
                output_rows.extend(indices[position] for position in semantic_positions)
                output_pieces.append(
                    core_output.index_select(
                        1, torch.tensor(semantic_positions, device=query.device)
                    )
                )
            if boundary_token is not None:
                states[boundary_token] = final_states[seq_idx]
            item["source"] = boundary_token
            item["position"] += 1
            if item["position"] == len(item["segments"]):
                pending.remove(item)
    if output_pieces:
        output = output.index_copy(
            1,
            torch.tensor(output_rows, device=query.device),
            torch.cat(output_pieces, dim=1),
        )
    return output
