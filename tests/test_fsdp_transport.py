from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.models.tree_attn.tree import build_packed_tree_batch
from areal.utils.data import (
    TRANSPORT_DUMMY_KEY,
    split_padded_tensor_dict_into_mb_list,
)


def _make_batch(rank: int, dummy_rank: int, *, tree: bool, vision: bool = True):
    batch_size = 1 if rank == dummy_rank else 2
    data = {
        "input_ids": torch.arange(batch_size * 128).reshape(batch_size, 128),
        "attention_mask": torch.ones(batch_size, 128, dtype=torch.bool),
    }
    if vision and rank != dummy_rank:
        data["multi_modal_input"] = [
            {
                "pixel_values": torch.ones(1, 4),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            }
            for _ in range(batch_size)
        ]
    if tree:
        return build_packed_tree_batch(
            data, MicroBatchSpec(max_tokens_per_mb=128), dp_group=dist.group.WORLD
        )
    mb_list = split_padded_tensor_dict_into_mb_list(
        data,
        MicroBatchSpec(n_mbs=2),
        group=dist.group.WORLD,
        allow_transport_padding=True,
    )
    mb_list.padded_mbs = [
        {key: value for key, value in mb.items() if key != TRANSPORT_DUMMY_KEY}
        for mb in mb_list.mbs
    ]
    mb_list.padding_lengths = [0] * len(mb_list.mbs)
    return mb_list


def _check_fsdp_transport_ranks(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        engine = FSDPEngine.__new__(FSDPEngine)
        engine._initialized = True
        engine._cpu_group = dist.group.WORLD
        engine.is_vision_model = True
        engine.parallel_helper = SimpleNamespace(sp_size=1)
        engine.model = Mock(side_effect=AssertionError("model must not run"))

        for tree in (False, True):
            engine.enable_tree_training = tree
            for dummy_rank in (0, 1):
                mb_list = _make_batch(rank, dummy_rank, tree=tree)
                assert any(
                    mb.get(TRANSPORT_DUMMY_KEY) is True for mb in mb_list.mbs
                ) == (rank == dummy_rank)
                for forward_only in (False, True):
                    with pytest.raises(ValueError, match="FSDP transport padding"):
                        engine.forward_backward_batch(
                            mb_list, lambda *_: None, forward_only=forward_only
                        )

        # Real VLM batches remain accepted; text models may still use dummies.
        engine.enable_tree_training = False
        engine.model = Mock(return_value=SimpleNamespace(logits=torch.ones(1, 2, 3)))
        for is_vision_model, dummy_rank in ((True, -1), (False, 0)):
            engine.is_vision_model = is_vision_model
            mb_list = _make_batch(rank, dummy_rank, tree=False, vision=is_vision_model)
            engine.model.reset_mock()
            engine.forward_backward_batch(mb_list, lambda *_: None, forward_only=True)
            assert engine.model.call_count == len(mb_list.mbs)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_fsdp_vlm_transport_rejection_is_coordinated_before_forward(tmp_path):
    """A dummy rank and its image peer must both stop before model execution."""
    mp.spawn(
        _check_fsdp_transport_ranks,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )
