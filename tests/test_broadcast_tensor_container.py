from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.utils.data import broadcast_tensor_container


class TensorPair(NamedTuple):
    left: torch.Tensor
    right: dict[str, torch.Tensor]


def _run_tuple_broadcast(rank: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo", init_method=init_method, rank=rank, world_size=2
    )
    try:
        payload = None
        if rank == 0:
            payload = (
                torch.tensor([1, 2], dtype=torch.int64),
                TensorPair(
                    left=torch.tensor([3.0]),
                    right={"value": torch.tensor([4.0, 5.0])},
                ),
                (),
                "metadata",
            )

        result = broadcast_tensor_container(payload, src_rank=0)

        assert isinstance(result, tuple)
        assert isinstance(result[1], TensorPair)
        assert result[2] == ()
        assert result[3] == "metadata"
        torch.testing.assert_close(result[0], torch.tensor([1, 2]), rtol=0, atol=0)
        torch.testing.assert_close(result[1].left, torch.tensor([3.0]), rtol=0, atol=0)
        torch.testing.assert_close(
            result[1].right["value"], torch.tensor([4.0, 5.0]), rtol=0, atol=0
        )
    finally:
        dist.destroy_process_group()


def test_broadcast_tensor_container_preserves_tuple_types(tmp_path):
    """Tuple tensor leaves use tensor collectives and preserve container types."""
    init_file = tmp_path / "gloo_init"
    mp.spawn(
        _run_tuple_broadcast,
        args=(f"file://{init_file}",),
        nprocs=2,
        join=True,
    )


def _run_alias_broadcast(rank: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo", init_method=init_method, rank=rank, world_size=2
    )
    try:
        # A nonzero source and non-contiguous tensor exercise both directions
        # and the temporary contiguous buffer used by tensor broadcast.
        expected = torch.arange(12).reshape(3, 4)[:, ::2]
        previous = None
        for preserve in (False, True, True):
            payload = None
            if rank == 1:
                shared = expected.clone().t()
                payload = (
                    shared,
                    TensorPair(
                        shared,
                        {
                            "same": shared,
                            "equal": shared.clone(),
                            "view": shared.view_as(shared),
                        },
                    ),
                    None,
                )
            result = broadcast_tensor_container(
                payload, src_rank=1, preserve_tensor_aliases=preserve
            )
            assert isinstance(result[1], TensorPair)
            torch.testing.assert_close(result[0], expected.t(), rtol=0, atol=0)
            assert (result[0] is result[1].left) is preserve
            assert (result[0] is result[1].right["same"]) is preserve
            for key in ("equal", "view"):
                assert result[0] is not result[1].right[key]
                torch.testing.assert_close(
                    result[0], result[1].right[key], rtol=0, atol=0
                )
            assert result[0] is not previous
            previous = result[0]
    finally:
        dist.destroy_process_group()


def test_broadcast_tensor_container_preserves_opt_in_aliases_per_call(tmp_path):
    mp.spawn(
        _run_alias_broadcast,
        args=(f"file://{tmp_path / 'alias_init'}",),
        nprocs=2,
        join=True,
    )
