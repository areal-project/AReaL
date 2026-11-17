# SPDX-License-Identifier: Apache-2.0

"""Verify that duplicated (replicated) params are not double-counted in the
TP-reduced global grad norm.

This is the numerical consequence that ``MegatronEngine._mark_duplicated_params``
relies on when it clears ``tensor_model_parallel`` on replicated params. It
mirrors Megatron's ``param_is_not_tensor_parallel_duplicate`` selection:
a parameter is included in *this* rank's local grad-norm sum iff it is
tensor-parallel-sharded (each rank holds a unique shard) OR this is tp_rank 0.
The local sum-of-squares is then all-reduced (SUM) over the tensor-parallel
group.

The test runs on CPU via gloo (no GPU / megatron required); the full optimizer
path is exercised separately by the GPU integration test.
"""

import contextlib
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _param_included_on_rank(param: torch.nn.Parameter, tp_rank: int) -> bool:
    """Mirror Megatron's param_is_not_tensor_parallel_duplicate."""
    return bool(getattr(param, "tensor_model_parallel", False)) or tp_rank == 0


def _tp_global_grad_norm(params, group, tp_rank: int) -> float:
    local_sq = torch.zeros((), dtype=torch.float64)
    for p in params:
        if _param_included_on_rank(p, tp_rank):
            local_sq += (p.grad.double() ** 2).sum()
    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=group)
    return local_sq.sqrt().item()


def _worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = dist.group.WORLD

        # A replicated Linear weight: every TP rank holds an identical copy,
        # so its grad is identical on every rank.
        weight = torch.nn.Linear(4, 4, bias=False).weight
        weight.grad = torch.ones_like(weight)
        single_copy_norm = weight.grad.norm().double().item()

        # Correct: marked duplicated -> counted once (only on tp_rank 0).
        weight.tensor_model_parallel = False
        norm_fixed = _tp_global_grad_norm([weight], group, rank)

        # Buggy (TE default): mis-marked as TP-sharded -> counted on every rank
        # and SUM-reduced, inflating the norm by sqrt(world_size).
        weight.tensor_model_parallel = True
        norm_inflated = _tp_global_grad_norm([weight], group, rank)

        torch.testing.assert_close(norm_fixed, single_copy_norm, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(
            norm_inflated,
            single_copy_norm * (world_size**0.5),
            rtol=1e-9,
            atol=1e-9,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_duplicated_param_not_double_counted_in_tp_grad_norm(world_size: int):
    """A replicated param counts once when tensor_model_parallel is False, and
    is inflated by sqrt(tp) when it is (incorrectly) left True."""
    port = _find_free_port()
    mp.spawn(
        _worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )
