# SPDX-License-Identifier: Apache-2.0
"""Two processes: real sampling and multi-epoch PPO on shared memory or GIN."""

import os

import pytest
import torch
import torch.distributed as dist

from areal.engine.sglang_sampling_mask import candidate_distribution
from areal.infra.sampling_mask import consumer as consumer_module
from areal.infra.sampling_mask import service as service_module
from areal.infra.sampling_mask.consumer import SamplingMaskConsumer
from areal.infra.sampling_mask.lifetime import LeaseRegistry
from areal.infra.sampling_mask.protocol import REFERENCE_KEYS
from areal.infra.sampling_mask.service import MaskProducer
from areal.infra.sampling_mask.store import shared_namespace
from areal.utils.functional import ppo_actor_loss_fn
from areal.utils.sampling_mask import restricted_logprobs


def main():
    # Exercise manifest slicing and slot wraparound with many bounded chunks.
    consumer_module.SLOT_BYTES = service_module.SLOT_BYTES = 1024
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    device = 0 if torch.cuda.device_count() == 1 else int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    torch.manual_seed(124)
    logits = torch.randn(129, 257, device=device)
    metadata = [None]
    producer = None
    owner = None
    if rank == 0:
        owner = LeaseRegistry()
        lease = owner.create()
        producer = MaskProducer(device, max_sessions=1)
        ids, logp, counts = candidate_distribution(
            logits,
            torch.full((129, 1), 0.8, device=device),
            torch.full((129,), 0.95, device=device),
            torch.full((129,), 32, device=device),
            candidate_bound=32,
        )
        column = torch.multinomial(logp.exp(), 1)
        tokens = ids.gather(1, column).flatten()
        # Model 129 decode steps so the real source read path also exercises
        # bounded coalescing of small immutable CSR blocks.
        refs = []
        for step in range(129):
            refs.extend(
                producer.publish(
                    ids[step : step + 1],
                    counts[step : step + 1],
                    tokens[step : step + 1],
                    [7],
                    [lease],
                )
            )
        # Advance far beyond the former retention window while this batch is
        # still queued. A different completed workflow must not expire it.
        later = owner.create()
        producer.publish(ids[:1], counts[:1], tokens[:1], [200], [later])
        owner.release(later[2])
        metadata[0] = (
            refs,
            tokens.cpu().tolist(),
            logp.gather(1, column).flatten().cpu().tolist(),
            shared_namespace(),
        )
    dist.broadcast_object_list(metadata, src=0)
    if rank == 1:
        refs, tokens, old, namespace = metadata[0]
        data = {
            key: torch.tensor([ref[i] for ref in refs])
            for i, key in enumerate(REFERENCE_KEYS)
        }
        labels_cpu = torch.tensor(tokens)
        versions = torch.full_like(labels_cpu, 7)
        consumer = SamplingMaskConsumer(torch.device("cuda", device), None, 257)
        support = consumer.load_async(data, labels_cpu, versions).result()
        assert support.slot is not None
        shared = namespace is not None and namespace == shared_namespace()
        if shared:
            assert not consumer._sessions
            assert all(
                block.name and block.packed is None
                for store in consumer._stores.values()
                for block in store._blocks.values()
            )
        else:
            assert (
                sum(session.payload_bytes for session in consumer._sessions.values())
                == support.packed_ids.numel()
            )
        initial_control_bytes = consumer.control_bytes
        ids, offsets = support.chunk(0, 129, torch.device("cuda", device))
        parameter = torch.nn.Parameter(logits.clone())
        optimizer = torch.optim.SGD([parameter], lr=0.05)
        old_logp = torch.tensor(old, device=device)
        advantages = torch.linspace(-1, 1, 129, device=device)
        for epoch in range(3):
            current, _ = restricted_logprobs(
                parameter,
                labels_cpu.to(device),
                ids,
                offsets,
                temperature=0.8,
                vocab_size=257,
            )
            if epoch == 0:
                torch.testing.assert_close(
                    (current - old_logp).exp(),
                    torch.ones_like(current),
                    atol=1e-5,
                    rtol=0,
                )
            loss, _ = ppo_actor_loss_fn(
                current,
                old_logp,
                old_logp,
                advantages,
                0.2,
                torch.ones_like(current, dtype=torch.bool),
            )
            optimizer.zero_grad()
            loss.backward()
            assert torch.isfinite(parameter.grad).all()
            optimizer.step()
            permutation = torch.arange(128, -1, -1)
            reordered = {key: value[permutation] for key, value in data.items()}
            reused = consumer.load_async(
                reordered, labels_cpu[permutation], versions[permutation]
            ).result()
            assert reused.slot is None  # backward can still reference the first mask
            from areal.utils.sampling_mask import SamplingMask

            expected = SamplingMask(
                support.packed_ids, support.offsets, 257, 7
            ).select_rows(permutation)
            torch.testing.assert_close(
                reused.packed_ids, expected.packed_ids, atol=0, rtol=0
            )
        assert consumer.id_bytes == support.packed_ids.numel()
        assert consumer.control_bytes == initial_control_bytes
        with pytest.raises(ValueError, match="not aligned"):
            consumer.load_async(data, labels_cpu + 1, versions).result()
        # The failed task must not poison the next read or release the lease.
        recovered = consumer.load_async(data, labels_cpu, versions).result()
        torch.testing.assert_close(recovered.packed_ids, support.packed_ids)
        consumer.close()
        # Another shard can legitimately reference the same parent prefix.
        second = SamplingMaskConsumer(torch.device("cuda", device), None, 257)
        repeated = second.load(data, labels_cpu, versions)
        torch.testing.assert_close(
            repeated.packed_ids, support.packed_ids, atol=0, rtol=0
        )
        second.close()
        from areal.utils import logging

        logging.getLogger("RemoteInfEngine").warning(
            "Sampling mask pipeline passed: 129 tokens, 3 PPO epochs, %s logical ID bytes via %s",
            consumer.id_bytes,
            "shared memory" if shared else "GIN",
        )
    dist.barrier()
    # Processes exit only after the receiver consumed every source chunk.
    dist.destroy_process_group()
    if producer is not None:
        assert len(producer._store._blocks) == 1
        owner.close()
        assert producer._store.nbytes == 0
        producer.close()


if __name__ == "__main__":
    main()
