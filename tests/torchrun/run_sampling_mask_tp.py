# SPDX-License-Identifier: Apache-2.0
"""Two-rank TP probability/gradient comparison with a dense FP32 reference."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from areal.infra.sampling_mask.consumer import LoadedSamplingMask, aggregate_metrics
from areal.models.mcore.vocab_parallel_head import _ChunkedVocabParallelLMHead
from areal.utils import logging
from areal.utils.sampling_mask import pack_uint24, restricted_logprobs


def main():
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=torch.device("cuda", device))
    group = dist.group.WORLD
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    assert world == 2
    torch.manual_seed(71)
    ids = torch.tensor([1, 4, 9, 0, 2, 3, 10, 7, 5, 6], dtype=torch.int32)
    offsets = torch.tensor([0, 3, 7, 8, 10], dtype=torch.int64)
    labels = torch.tensor([4, 10, 7, 5], device=device)
    mask = torch.zeros(4, 12, dtype=torch.bool, device=device)
    mask[
        torch.repeat_interleave(
            torch.arange(4, device=device), offsets.diff().to(device)
        ),
        ids.to(device).long(),
    ] = True
    support = LoadedSamplingMask(pack_uint24(ids), offsets, 12)
    upstream = torch.tensor([0.1, -0.3, 0.2, -0.6], device=device)
    for chunk_size in [1, 3, 16]:
        hidden = torch.randn(4, 1, 16, device=device, requires_grad=True)
        full_weight = torch.randn(12, 16, device=device, requires_grad=True)
        weight = full_weight.detach().chunk(world)[rank].clone().requires_grad_()
        reference_hidden = hidden.detach().clone().requires_grad_()
        actual = _ChunkedVocabParallelLMHead.apply(
            hidden,
            weight,
            None,
            labels,
            0.7,
            chunk_size,
            False,
            True,
            False,
            group,
            0.5,
            True,
            support,
        )[0]
        logits = (reference_hidden.reshape(4, 16) @ full_weight.t()) * 0.5 / 0.7
        expected = (
            logits.masked_fill(~mask, -torch.inf)
            .log_softmax(-1)
            .gather(1, labels[:, None])
            .flatten()
        )
        actual.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(
            hidden.grad, reference_hidden.grad, atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            weight.grad, full_weight.grad.chunk(world)[rank], atol=1e-4, rtol=0
        )
    # Ordinary autograd path must keep only each rank's local gradient.
    logits = torch.randn(4, 12, device=device, requires_grad=True)
    local = logits.detach().chunk(world, -1)[rank].clone().requires_grad_()
    actual, _ = restricted_logprobs(
        local,
        labels,
        ids.to(device),
        offsets.to(device),
        temperature=0.7,
        vocab_size=12,
        tp_group=group,
    )
    expected = (
        (logits / 0.7)
        .masked_fill(~mask, -torch.inf)
        .log_softmax(-1)
        .gather(1, labels[:, None])
        .flatten()
    )
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        local.grad, logits.grad.chunk(world, -1)[rank], atol=1e-4, rtol=0
    )
    cpu_group = dist.new_group(list(range(world)), backend="gloo")
    counters = SimpleNamespace(
        _candidate_histogram=torch.bincount(offsets.diff(), minlength=13).numpy(),
        id_bytes=30,
        offset_bytes=40,
        control_bytes=17 if rank == 1 else 0,
        _sessions={0: SimpleNamespace(payload_bytes=30 if rank == 1 else 0)},
        _cache_bytes=1024 + rank,
    )
    metrics = aggregate_metrics(
        counters, vocab_size=12, logical_owner=rank == 0, cpu_group=cpu_group
    )
    assert metrics["sampling_mask/lifetime/received_rows"] == 4
    assert metrics["sampling_mask/lifetime/id_payload_bytes"] == 30
    assert metrics["sampling_mask/lifetime/gin_payload_bytes"] == 30
    assert metrics["sampling_mask/lifetime/control_body_bytes"] == 17
    assert metrics["sampling_mask/cpu_cache_bytes_sum"] == 2049
    assert metrics["sampling_mask/cpu_cache_bytes_max_rank"] == 1025
    dist.destroy_process_group(cpu_group)
    logging.getLogger("MegatronEngine").warning(
        "Sampling mask TP=2 FP32 probability and gradient tests passed"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
