# SPDX-License-Identifier: Apache-2.0

"""CPU reference checks for streamed fixed-batch loss and retry semantics.

GPU/MCore execution is covered by the separate distributed acceptance script.
These tests deliberately do not mock Megatron DDP or optimizer internals.
"""

import pytest
import torch

from areal.engine.megatron_utils.stream_update import (
    StreamUpdateState,
    stream_gradient_scale,
)


@pytest.mark.parametrize("per_token_loss", [False, True])
@pytest.mark.parametrize("replica_chunks", [[[1, 3], [8], []], [[2], [], [5, 1, 4]]])
def test_streamed_unequal_chunks_and_empty_replica_match_global_token_mean(
    per_token_loss, replica_chunks
):
    # Arrange: unequal token masses, microbatch counts, and an idle DP replica.
    dtype = torch.float64
    theta = torch.tensor([0.3, -0.7], dtype=dtype, requires_grad=True)
    generator = torch.Generator().manual_seed(71)
    samples = [
        [torch.randn(size, 2, generator=generator, dtype=dtype) for size in chunks]
        for chunks in replica_chunks
    ]
    all_samples = torch.cat([chunk for replica in samples for chunk in replica])
    reference_loss = (all_samples @ theta).square().mean()
    reference_grad = torch.autograd.grad(reference_loss, theta)[0]
    global_mass = all_samples.shape[0]
    dp_size = len(samples)

    # Act: the local schedule accumulates raw numerator gradients; commit applies
    # DP-size compensation only for MCore's legacy averaged-gradient contract.
    replica_grads = []
    for replica in samples:
        numerator_grad = torch.zeros_like(theta)
        for chunk in replica:
            chunk_mean = (chunk @ theta).square().mean()
            numerator = chunk_mean * chunk.shape[0]
            numerator_grad += torch.autograd.grad(numerator, theta)[0]
        replica_grads.append(numerator_grad)
    scale = stream_gradient_scale(global_mass, dp_size, per_token_loss=per_token_loss)
    committed_grad = torch.stack(replica_grads).sum(0) * scale
    if not per_token_loss:
        committed_grad /= dp_size

    # Assert: absent replicas contribute no denominator, and chunks of different
    # size have the same weight they have in the ordinary single global batch.
    torch.testing.assert_close(committed_grad, reference_grad, rtol=1e-12, atol=1e-12)


def test_stream_chunk_retry_does_not_add_token_mass_twice():
    state = StreamUpdateState("batch-12")
    first = state.complete_chunk(7, 23)

    retry = state.check_chunk(7)

    assert retry == first
    assert state.local_mass == 23
    with pytest.raises(RuntimeError, match="twice"):
        state.complete_chunk(7, 23)
    assert state.local_mass == 23


def test_failed_stream_chunk_cannot_be_replayed_or_committed():
    state = StreamUpdateState("batch-12")
    state.complete_chunk(1, 8)
    state.failed = True

    with pytest.raises(RuntimeError, match="abort"):
        state.check_chunk(1)
    with pytest.raises(RuntimeError, match="abort"):
        state.check_chunk(2)


@pytest.mark.parametrize("global_mass,dp_size", [(0, 4), (-1, 4), (12, 0)])
def test_stream_normalization_rejects_invalid_denominators(global_mass, dp_size):
    with pytest.raises(ValueError):
        stream_gradient_scale(global_mass, dp_size, per_token_loss=True)
