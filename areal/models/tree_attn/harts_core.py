# SPDX-License-Identifier: Apache-2.0

"""Differentiable execution of a compact tree GDN plan."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from areal.models.tree_attn.harts_plan import LinearAttentionPlan


@dataclass
class _CallSequence:
    indices: list[int]
    source: int | None
    semantic_rows: list[int]
    semantic_positions: list[int]
    boundary_token: int | None = None


def _split_call_schedule(plan: LinearAttentionPlan) -> list[list[_CallSequence]]:
    """Use final-state exports without repeating extra prefix work."""
    pending: list[_CallSequence] = []
    for round_sequences in plan.rounds:
        for sequence in round_sequences:
            indices = sequence.replay_indices + sequence.compact_indices
            semantic = set(sequence.compact_indices)
            source = sequence.initial_state_token
            start = 0
            for end, token in enumerate(indices, 1):
                if token not in semantic or token not in plan.boundary_state_tokens:
                    continue
                segment = indices[start:end]
                positions = [i for i, row in enumerate(segment) if row in semantic]
                pending.append(
                    _CallSequence(
                        segment,
                        source,
                        [segment[i] for i in positions],
                        positions,
                        token,
                    )
                )
                source = token
                start = end
            if start < len(indices):
                segment = indices[start:]
                positions = [i for i, row in enumerate(segment) if row in semantic]
                pending.append(
                    _CallSequence(
                        segment,
                        source,
                        [segment[i] for i in positions],
                        positions,
                    )
                )

    calls: list[list[_CallSequence]] = []
    ready_states: set[int] = set()
    while pending:
        ready = [
            sequence
            for sequence in pending
            if sequence.source is None or sequence.source in ready_states
        ]
        if not ready:
            raise ValueError("linear-attention plan contains a state dependency cycle")
        calls.append(ready)
        ready_ids = {id(sequence) for sequence in ready}
        pending = [sequence for sequence in pending if id(sequence) not in ready_ids]
        ready_states.update(
            sequence.boundary_token
            for sequence in ready
            if sequence.boundary_token is not None
        )
    return calls


def _packed_round_schedule(plan: LinearAttentionPlan) -> list[list[_CallSequence]]:
    """Export internal states with extra prefixes in the same packed call."""
    calls: list[list[_CallSequence]] = []
    for round_sequences in plan.rounds:
        packed: list[_CallSequence] = []
        for sequence in round_sequences:
            indices = sequence.replay_indices + sequence.compact_indices
            if not indices:
                continue
            replay_length = len(sequence.replay_indices)
            end_token = indices[-1]
            packed.append(
                _CallSequence(
                    indices,
                    sequence.initial_state_token,
                    sequence.compact_indices,
                    list(range(replay_length, len(indices))),
                    end_token if end_token in plan.boundary_state_tokens else None,
                )
            )
            for position, token in enumerate(
                sequence.compact_indices, start=replay_length + 1
            ):
                if token in plan.boundary_state_tokens and position < len(indices):
                    packed.append(
                        _CallSequence(
                            indices[:position],
                            sequence.initial_state_token,
                            [],
                            [],
                            token,
                        )
                    )
        if packed:
            calls.append(packed)
    return calls


def run_linear_attention_plan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    plan: LinearAttentionPlan,
    kernel: Callable,
) -> torch.Tensor:
    """Execute compact tokens with at most one public FLA call per planned round.

    Splitting at requested states avoids redundant core work when it already
    fits the minimum-call budget. Otherwise, state-only prefix sequences are
    packed into the same call as their producer. Their final states are
    differentiable, and their outputs are omitted from semantic rows.
    """
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("tree GDN requires a singleton batch dimension")
    split_calls = _split_call_schedule(plan)
    calls = (
        split_calls
        if len(split_calls) <= len(plan.rounds)
        else _packed_round_schedule(plan)
    )
    output = value.new_zeros(value.shape)
    states: dict[int, torch.Tensor] = {}
    output_rows: list[int] = []
    output_pieces: list[torch.Tensor] = []
    zero_state = None

    for call in calls:
        indices: list[int] = []
        offsets = [0]
        initial_states = []
        semantic_positions: list[int] = []
        needs_final_state = any(
            sequence.boundary_token is not None for sequence in call
        )
        needs_initial_state = any(sequence.source is not None for sequence in call)
        for sequence in call:
            if sequence.source is not None and sequence.source not in states:
                raise ValueError("linear-attention round needs an unavailable state")
            indices.extend(sequence.indices)
            semantic_positions.extend(
                offsets[-1] + position for position in sequence.semantic_positions
            )
            offsets.append(len(indices))
            if needs_initial_state:
                if sequence.source is None:
                    if zero_state is None:
                        zero_state = torch.zeros(
                            (query.shape[2], key.shape[-1], value.shape[-1]),
                            device=query.device,
                            dtype=torch.float32,
                        )
                    initial_states.append(zero_state)
                else:
                    initial_states.append(states[sequence.source])
            output_rows.extend(sequence.semantic_rows)

        rows = torch.tensor(indices, device=query.device, dtype=torch.long)
        cu_seqlens = torch.tensor(offsets, device=query.device, dtype=torch.int32)
        cu_seqlens_cpu = torch.tensor(offsets, dtype=torch.int32)
        packed = [
            tensor.index_select(1, rows) for tensor in (query, key, value, gate, beta)
        ]
        core_output, final_states = kernel(
            *packed,
            initial_state=torch.stack(initial_states) if needs_initial_state else None,
            output_final_state=needs_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            use_qk_l2norm_in_kernel=False,
        )
        if semantic_positions:
            output_pieces.append(
                core_output.index_select(
                    1, torch.tensor(semantic_positions, device=query.device)
                )
            )
        for sequence_idx, sequence in enumerate(call):
            if sequence.boundary_token is not None:
                assert final_states is not None
                states[sequence.boundary_token] = final_states[sequence_idx]

    if output_pieces:
        output = output.index_copy(
            1,
            torch.tensor(output_rows, device=query.device),
            torch.cat(output_pieces, dim=1),
        )
    return output
