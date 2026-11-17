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
        # Two independent DP-like subgroups must use the original broadcast.
        groups = [
            dist.new_group([0, 1], backend="gloo"),
            dist.new_group([2, 3], backend="gloo"),
        ]
        group = groups[rank // 2]
        source = 2 * (rank // 2)

        def reject_group_creation(*args, **kwargs):
            raise AssertionError("proper subgroups must not create node groups")

        transport._node_groups = reject_group_creation
        payload = {"multi_modal_input": torch.ones(2)} if rank == source else None
        result = transport(
            payload, src_rank=source, group=group, preserve_tensor_aliases=True
        )
        torch.testing.assert_close(
            result["multi_modal_input"], torch.ones(2), rtol=0, atol=0
        )
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
