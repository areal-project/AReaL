import torch
import torch.distributed as dist

from areal.api.cli_args import MicroBatchSpec
from areal.engine.core.train_engine import compute_microbatch_loss_weight
from areal.models.tree_attn.tree import build_packed_tree_batch
from areal.utils.data import TRANSPORT_DUMMY_KEY


def test_tree_transport_dummy_bypasses_objective_weight(monkeypatch):
    data = {
        "input_ids": torch.arange(4).view(1, 4),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "loss_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda _group=None: 2)

    def _all_gather(outputs, local_count, group=None):
        del group
        outputs[0].copy_(local_count)
        outputs[1].fill_(2)

    monkeypatch.setattr(dist, "all_gather", _all_gather)

    mb_list = build_packed_tree_batch(
        data,
        MicroBatchSpec(max_tokens_per_mb=128),
    )
    semantic_mb, transport_mb = mb_list.mbs

    assert TRANSPORT_DUMMY_KEY not in semantic_mb
    assert transport_mb[TRANSPORT_DUMMY_KEY] is True
    assert mb_list.padded_mbs is not None
    assert TRANSPORT_DUMMY_KEY not in mb_list.padded_mbs[1]

    callback_called = False

    def _loss_weight(_microbatch):
        nonlocal callback_called
        callback_called = True
        return torch.tensor(1.0)

    torch.testing.assert_close(
        compute_microbatch_loss_weight(transport_mb, _loss_weight),
        torch.tensor(0.0),
        rtol=0.0,
        atol=0.0,
    )
    assert callback_called is False
