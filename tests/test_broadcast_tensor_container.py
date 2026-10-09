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
    """Run real source/receiver collectives with receiver input absent."""
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    original_broadcast_object_list = dist.broadcast_object_list
    markers = []
    text_metadata = None

    def record_metadata(objects, *args, **kwargs):
        original_broadcast_object_list(objects, *args, **kwargs)
        if isinstance(objects[0], tuple):
            markers.append(objects[0][0])

    dist.broadcast_object_list = record_metadata
    try:
        for index, mode in enumerate(
            ("text_default", "default", "text", "empty", "mixed", "text", "mixed")
        ):
            source_rank = index % 2
            markers.clear()
            payload = None
            if rank == source_rank:
                shared = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
                other_view = shared[:, :]
                payload = {"text": TensorPair(shared, {"value": shared})}
                if mode in ("default", "mixed"):
                    payload["multi_modal_input"] = [
                        {},
                        {"pixel_values": shared},
                        {"pixel_values": shared},
                        {"pixel_values": other_view},
                        {"pixel_values": shared + 1},
                    ]
                elif mode == "empty":
                    payload["multi_modal_input"] = [
                        {},
                        {"pixel_values": torch.empty(0)},
                    ]
            result = broadcast_tensor_container(
                payload,
                # The receiver must follow source metadata, not its own flag/data.
                src_rank=source_rank,
                preserve_tensor_aliases=rank == source_rank
                and mode not in {"default", "text_default"},
            )
            torch.testing.assert_close(
                result["text"].left,
                torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
                rtol=0,
                atol=0,
            )
            if mode == "mixed":
                images = result["multi_modal_input"]
                assert markers[0] == "tensor_aliases"
                assert images[1]["pixel_values"] is images[2]["pixel_values"]
                assert images[1]["pixel_values"] is not images[3]["pixel_values"]
                torch.testing.assert_close(
                    images[4]["pixel_values"],
                    images[1]["pixel_values"] + 1,
                    rtol=0,
                    atol=0,
                )
                assert result["text"].left is result["text"].right["value"]
            else:
                if mode == "text_default":
                    text_metadata = list(markers)
                elif mode == "text":
                    assert markers == text_metadata
                assert "tensor_aliases" not in markers
                assert "tensor_ref" not in markers
                if rank != source_rank:
                    assert result["text"].left is not result["text"].right["value"]
    finally:
        dist.broadcast_object_list = original_broadcast_object_list
        dist.destroy_process_group()


def test_broadcast_aliases_are_source_selected_and_multimodal_only(tmp_path):
    """A text batch retains the old wire/alias contract, including after a VLM call."""
    mp.spawn(
        _run_alias_broadcast, args=(f"file://{tmp_path / 'alias_init'}",), nprocs=2
    )
