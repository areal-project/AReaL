# SPDX-License-Identifier: Apache-2.0

"""Expert-name and metadata-merge contracts for MoE weight transfer.

Both behaviours below decide whether a routed expert is transferred at all, and
both fail silently: the shapes stay correct either way, so a mismatch shows up
as diverging train/inference logprobs rather than an error.
"""

from types import SimpleNamespace

import pytest
import torch

from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter
from areal.v2.weight_update.gateway.app import _merge_meta_by_name

NUM_EXPERTS = 8
HIDDEN = 4
FFN = 6


def _adapter(ep_size, ep_rank):
    adapter = AwexSGLangAdapter.__new__(AwexSGLangAdapter)
    adapter._get_model = lambda: SimpleNamespace(
        config=SimpleNamespace(num_experts=NUM_EXPERTS, n_routed_experts=NUM_EXPERTS)
    )
    adapter._rank_info = SimpleNamespace(ep_size=ep_size, ep_rank=ep_rank)
    adapter._build_rank_info = lambda: adapter._rank_info
    return adapter


def _local(ep_size):
    return NUM_EXPERTS // ep_size


def _expert_ids(pairs):
    return {
        int(name.split(".experts.")[1].split(".")[0])
        for name, _ in pairs
        if ".experts." in name
    }


@pytest.mark.parametrize("fused", ["w13_weight", "w2_weight"])
@pytest.mark.parametrize("ep_rank", range(4))
def test_expert_ids_are_global_under_expert_parallelism(fused, ep_rank):
    ep_size = 4
    adapter = _adapter(ep_size, ep_rank)
    local = _local(ep_size)
    shape = (local, 2 * FFN, HIDDEN) if fused == "w13_weight" else (local, HIDDEN, FFN)

    pairs = adapter._unfuse_params(
        f"model.layers.0.mlp.experts.{fused}", torch.zeros(*shape)
    )

    assert _expert_ids(pairs) == set(range(ep_rank * local, (ep_rank + 1) * local))


def test_ranks_together_advertise_every_expert_exactly_once():
    """The transfer plan indexes by name, so a shadowed id is never sent."""
    ep_size = 4
    seen = []
    for ep_rank in range(ep_size):
        adapter = _adapter(ep_size, ep_rank)
        seen.extend(
            _expert_ids(
                adapter._unfuse_params(
                    "model.layers.0.mlp.experts.w13_weight",
                    torch.zeros(_local(ep_size), 2 * FFN, HIDDEN),
                )
            )
        )

    assert sorted(seen) == list(range(NUM_EXPERTS))


def test_expert_ids_unchanged_without_expert_parallelism():
    adapter = _adapter(ep_size=1, ep_rank=0)

    pairs = adapter._unfuse_params(
        "model.layers.0.mlp.experts.w13_weight",
        torch.zeros(NUM_EXPERTS, 2 * FFN, HIDDEN),
    )

    assert _expert_ids(pairs) == set(range(NUM_EXPERTS))


def test_shared_experts_keep_their_own_index_under_expert_parallelism():
    ep_size, ep_rank = 2, 1
    adapter = _adapter(ep_size, ep_rank)
    local = _local(ep_size)

    pairs = adapter._unfuse_params(
        "model.layers.0.mlp.experts.w13_weight",
        torch.zeros(local, 2 * FFN, HIDDEN),
    )

    assert _expert_ids(pairs) == {4, 5, 6, 7}
    assert not any("shared_experts" in name for name, _ in pairs)


def _entry(name, rank):
    return {
        "data": {
            "name": name,
            "shards": [{"rank": rank}],
            "replicas": [{"data": {"shards": [{"rank": rank}]}}],
        }
    }


def test_merging_keeps_every_rank_shard_under_one_name():
    name = "model.layers.0.mlp.experts.3.down_proj.weight"
    flat = [_entry(name, rank) for rank in range(16)]

    merged = _merge_meta_by_name(flat)

    assert len(merged) == 1
    assert len(merged[0]["data"]["shards"]) == 16


def test_merging_preserves_distinct_names():
    flat = [_entry("a.weight", 0), _entry("b.weight", 0), _entry("a.weight", 1)]

    merged = _merge_meta_by_name(flat)

    assert {e["data"]["name"] for e in merged} == {"a.weight", "b.weight"}
    assert len(merged) == 2


def test_connect_merges_the_inference_metadata_it_collected():
    """Guard the call site, not just the helper.

    The helper being correct is useless if /connect forgets to apply it, which
    is exactly the state this fix repairs.
    """
    import inspect

    from areal.v2.weight_update.gateway import app as gateway_app

    source = inspect.getsource(gateway_app)
    collect_sites = source.count("infer_params_meta.extend(meta)")
    merge_sites = source.count(
        "infer_params_meta = _merge_meta_by_name(infer_params_meta)"
    )

    assert collect_sites > 0
    assert merge_sites == collect_sites, (
        f"{collect_sites} places collect inference metadata but only "
        f"{merge_sites} merge it by name"
    )
