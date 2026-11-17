# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.infra.rpc.node_shared_vision import (
    NodeSharedVisionTransport,
)


def _broadcast_small_container(data, src_rank=0, group=None, **kwargs):
    objects = [data if dist.get_rank() == src_rank else None]
    dist.broadcast_object_list(objects, src=src_rank, group=group)
    return objects[0]


def _run_node_shared_vision(rank: int, init_method: str, shared_dir: str) -> None:
    import os

    os.environ["AREAL_VISION_SHM_DIR"] = shared_dir
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=4)
    try:
        transport = NodeSharedVisionTransport(
            _broadcast_small_container, node_name=f"node-{rank // 2}"
        )
        for multimodal in (False, True, True, False):
            payload = None
            if rank == 1:
                payload = {"input_ids": torch.tensor([[1, 2, 3]])}
                if multimodal:
                    image = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
                    payload["multi_modal_input"] = [
                        {"pixel_values": image},
                        {"pixel_values": image},
                        {"pixel_values": image.clone()},
                    ]
            result = transport(payload, src_rank=1, preserve_tensor_aliases=rank == 1)
            torch.testing.assert_close(
                result["input_ids"], torch.tensor([[1, 2, 3]]), rtol=0, atol=0
            )
            if multimodal:
                images = result["multi_modal_input"]
                shared = images[0]["pixel_values"]
                assert shared is images[1]["pixel_values"]
                assert shared is not images[2]["pixel_values"]
                expected = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
                torch.testing.assert_close(shared, expected, rtol=0, atol=0)
                # A write in the test proves that same-node ranks map the
                # same physical storage, while the other node stays isolated.
                # Finish value checks on every rank before the test mutation.
                dist.barrier()
                if rank == 0:
                    shared[0, 0] = 99
                dist.barrier()
                assert shared[0, 0].item() == (99 if rank < 2 else 0)
                torch.testing.assert_close(
                    images[2]["pixel_values"], expected, rtol=0, atol=0
                )
                dist.barrier()
                assert list(Path(shared_dir).iterdir()) == []
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_vision_broadcast_shares_storage_per_node_preserving_values_and_aliases(
    tmp_path,
):
    """Real Gloo collectives cover two node groups, caching, and text fallback."""
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    mp.spawn(
        _run_node_shared_vision,
        args=(f"file://{tmp_path / 'gloo_init'}", str(shared_dir)),
        nprocs=4,
        join=True,
    )


def _run_dp_shared_vision(
    rank: int, init_method: str, shared_dir: str, spanning: bool
) -> None:
    import os
    from datetime import timedelta

    os.environ["AREAL_VISION_SHM_DIR"] = shared_dir
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=60),
    )
    try:
        replicas = (
            [[0, 1, 4, 5], [2, 3, 6, 7]] if spanning else [[0, 1, 2, 3], [4, 5, 6, 7]]
        )
        groups = [
            dist.new_group(members, backend="gloo", timeout=timedelta(seconds=30))
            for members in replicas
        ]
        replica = next(i for i, members in enumerate(replicas) if rank in members)
        members = replicas[replica]
        group = groups[replica]
        local_members = [r for r in members if r // 4 == rank // 4]

        def broadcast_metadata(data, **kwargs):
            # Vision data must have been extracted, not silently sent by the
            # ordinary broadcast fallback (which would pass value-only tests).
            if data is not None and "multi_modal_input" in data:
                assert not isinstance(data["multi_modal_input"][0], torch.Tensor)
            return _broadcast_small_container(data, **kwargs)

        transport = NodeSharedVisionTransport(
            broadcast_metadata, node_name=f"node-{rank // 4}"
        )

        def transfer(step: int) -> None:
            source = members[step % len(members)]
            expected = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
            expected = expected + replica * 100 + step * 10
            payload = None
            if rank == source:
                payload = {
                    "input_ids": torch.tensor([replica, step]),
                    "multi_modal_input": [expected, expected, expected.clone()],
                }
            result = transport(
                payload, src_rank=source, group=group, preserve_tensor_aliases=True
            )
            torch.testing.assert_close(
                result["input_ids"], torch.tensor([replica, step]), rtol=0, atol=0
            )
            images = result["multi_modal_input"]
            assert images[0] is images[1]
            assert images[0] is not images[2]
            torch.testing.assert_close(images[0], expected, rtol=0, atol=0)
            dist.barrier(group=group)
            # Test-only mutation proves actual sharing on each node. The marker
            # differs across nodes and DP groups, exposing accidental aliasing.
            marker = 1000 + replica * 100 + rank // 4
            if rank == local_members[0]:
                images[0][0, 0] = marker
            dist.barrier(group=group)
            assert images[0][0, 0].item() == marker
            torch.testing.assert_close(images[2], expected, rtol=0, atol=0)
            dist.barrier(group=group)

        # Replica B cannot enter transport until A finishes its first call.
        # A WORLD-dependent group initialization would deadlock deterministically.
        if replica == 0:
            transfer(0)
            if rank == members[0]:
                dist.send(torch.ones(1), dst=replicas[1][0])
        else:
            if rank == members[0]:
                dist.recv(torch.empty(1), src=replicas[0][0])
            dist.barrier(group=group)
            transfer(0)
        dist.barrier()
        # Different stream lengths, nonzero/remote sources, then cached layouts.
        for step in range(1, 7 if replica == 0 else 5):
            transfer(step)
        dist.barrier()
        assert list(Path(shared_dir).iterdir()) == []
        # DP-only configurations have singleton model-parallel groups.
        singleton = dist.new_group(
            [rank], backend="gloo", use_local_synchronization=True
        )
        result = transport(
            {"multi_modal_input": [torch.tensor([float(rank)])]},
            src_rank=rank,
            group=singleton,
            preserve_tensor_aliases=True,
        )
        torch.testing.assert_close(
            result["multi_modal_input"][0],
            torch.tensor([float(rank)]),
            rtol=0,
            atol=0,
        )
        dist.barrier()
        assert list(Path(shared_dir).iterdir()) == []
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.parametrize("spanning", [False, True], ids=["local", "cross-node"])
def test_dp_vision_sharing_independent_streams_and_isolated_storage(tmp_path, spanning):
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    mp.spawn(
        _run_dp_shared_vision,
        args=(f"file://{tmp_path / 'gloo_init'}", str(shared_dir), spanning),
        nprocs=8,
        join=True,
    )
