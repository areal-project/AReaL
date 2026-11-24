# SPDX-License-Identifier: Apache-2.0

"""Score centering against tensor-parallel vocabulary shards.

Megatron hands the correction one vocabulary shard per rank, so the selected
log-probabilities must come out identical to a dense ``log_softmax`` and the
gradient must stay unscaled: the forward reduces a SUM collective, but its
local backward has to behave like the identity or every rank would multiply its
gradient by ``tp_size``.

The cases run on CPU through gloo in two processes, no GPU and no Megatron
import required.
"""

import contextlib
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.utils.score_centering import vocab_parallel_selected_logprobs

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Requires Gloo",
)

ROWS, VOCAB, K, CHUNK = 6, 32, 4, 4


def _find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _dense_reference():
    """Dense logits, sampler ids and their reference log-probs."""
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(ROWS, VOCAB, generator=generator, dtype=torch.float64)
    ids = logits.softmax(-1).topk(K, dim=-1).indices
    log_probs = logits.log_softmax(-1)
    reference = log_probs.gather(-1, ids)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return logits, ids, reference, entropy


def _worker(rank: int, world_size: int, port: int, mode: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        logits, ids, reference, entropy_reference = _dense_reference()
        shard = VOCAB // world_size
        start = rank * shard
        local = logits[:, start : start + shard].clone().requires_grad_(True)

        selected, entropy = vocab_parallel_selected_logprobs(
            local,
            ids,
            tp_group=dist.group.WORLD,
            vocab_size=VOCAB,
            chunk_size=CHUNK,
        )

        if mode == "values":
            torch.testing.assert_close(selected, reference, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(
                entropy, entropy_reference, rtol=1e-8, atol=1e-10
            )
            return

        selected.sum().backward()
        dense = logits.clone().requires_grad_(True)
        dense.log_softmax(-1).gather(-1, ids).sum().backward()
        torch.testing.assert_close(
            local.grad, dense.grad[:, start : start + shard], rtol=1e-10, atol=1e-12
        )
    finally:
        dist.destroy_process_group()


def _run(mode: str) -> None:
    port = _find_free_port()
    mp.spawn(_worker, args=(2, port, mode), nprocs=2, join=True)


def test_vocab_parallel_selected_logprobs_matches_dense_when_tp2():
    """Test that two vocabulary shards reproduce dense log-probs and entropy."""
    _run("values")


def test_vocab_parallel_gradient_is_not_scaled_by_tp_size_when_tp2():
    """Test that the shard gradient equals the dense slice, not tp_size times it."""
    _run("grads")


def test_vocab_parallel_rejects_ids_outside_the_shards_when_tp2():
    """Test that an id beyond the covered vocabulary is refused, not clamped."""
    port = _find_free_port()
    mp.spawn(_worker_reject, args=(2, port), nprocs=2, join=True)


def _worker_reject(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        logits, ids, _, _ = _dense_reference()
        shard = VOCAB // world_size
        local = logits[:, rank * shard : (rank + 1) * shard].clone()
        bad_ids = ids.clone()
        bad_ids[0, 0] = VOCAB  # one past the last id
        # Every rank evaluates the same rows, so every rank must reject it.
        with pytest.raises((RuntimeError, ValueError)):
            vocab_parallel_selected_logprobs(
                local,
                bad_ids,
                tp_group=dist.group.WORLD,
                vocab_size=VOCAB,
                chunk_size=CHUNK,
            )
    finally:
        dist.destroy_process_group()
