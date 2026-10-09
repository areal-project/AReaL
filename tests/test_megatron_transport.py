from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.cli_args import MicroBatchSpec
from areal.engine.megatron_utils.transport import (
    nonempty_microbatch_target,
    validate_transport_padding,
)
from areal.utils.data import (
    TRANSPORT_DUMMY_KEY,
    MicroBatchList,
    split_padded_tensor_dict_into_mb_list,
)


@pytest.mark.parametrize(
    ("requested", "real_rows", "expected"),
    [(128, 112, 112), (128, 74, 72), (128, 64, 64), (96, 112, 96)],
)
def test_nonempty_microbatch_target_uses_real_rows_without_dummy(
    requested, real_rows, expected
):
    """A variable-size rollout batch uses the largest real pipeline schedule."""
    target = nonempty_microbatch_target(requested, real_rows, 4, 8)
    data = {
        "input_ids": torch.arange(real_rows).view(real_rows, 1),
        "attention_mask": torch.ones(real_rows, 1, dtype=torch.bool),
    }

    mb_list = split_padded_tensor_dict_into_mb_list(
        data,
        MicroBatchSpec(n_mbs=target, n_mbs_divisor=4, max_tokens_per_mb=2),
        allow_transport_padding=True,
    )

    assert target == expected
    assert len(mb_list.mbs) == expected
    assert mb_list.transport_dummy_count == 0


def test_nonempty_microbatch_target_preserves_minimum_when_rows_too_few():
    """The existing MoE transport guard diagnoses an undersized batch."""
    assert nonempty_microbatch_target(96, 7, 4, 8) == 96


def test_nonempty_microbatch_target_respects_pair_granularity():
    """A preference batch counts pairs rather than individual rows."""
    real_rows = 148
    granularity = 2
    assert nonempty_microbatch_target(128, real_rows // granularity, 4, 8) == 72


def _microbatches(has_dummy: bool) -> MicroBatchList:
    mbs = [{"input_ids": torch.tensor([1, 2])}]
    if has_dummy:
        mbs.append({"input_ids": torch.tensor([0, 0]), TRANSPORT_DUMMY_KEY: True})
    return MicroBatchList(
        data={}, mb_spec=MicroBatchSpec(), mbs=mbs, group_lens=[2] * len(mbs)
    )


def _check_transport_ranks(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        # The rank with the auxiliary head has no dummy. Both must still stop.
        for dummy_rank in (0, 1):
            with pytest.raises(ValueError, match="All training ranks stopped"):
                validate_transport_padding(
                    _microbatches(rank == dummy_rank),
                    has_internal_objectives=rank != dummy_rank,
                    cpu_group=dist.group.WORLD,
                )

        # Internal objectives remain supported when no rank needs padding.
        validate_transport_padding(
            _microbatches(False),
            has_internal_objectives=True,
            cpu_group=dist.group.WORLD,
        )
        # Dense models still accept uneven transport padding.
        validate_transport_padding(
            _microbatches(rank == 0),
            has_internal_objectives=False,
            cpu_group=dist.group.WORLD,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_transport_rejection_is_coordinated_across_ranks(tmp_path):
    """Real Gloo peers must agree even when only one has dummy microbatches."""
    mp.spawn(
        _check_transport_ranks,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("forward_only", [False, True])
@pytest.mark.parametrize("internal_objective", ["moe", "mtp"])
def test_engine_rejects_padding_before_pipeline_execution(
    monkeypatch, forward_only, internal_objective
):
    """Train and forward-only schedules must reject before invoking MCore."""
    pytest.importorskip("megatron.core")
    from areal.engine import megatron_engine as engine_module

    engine = engine_module.MegatronEngine.__new__(engine_module.MegatronEngine)
    engine._ensure_ready = lambda: None
    engine.tf_config = SimpleNamespace(
        num_moe_experts=2 if internal_objective == "moe" else None
    )
    engine.mcore_config = SimpleNamespace(
        enable_mtp_training=internal_objective == "mtp"
    )
    engine.process_group_initialized = True
    engine._cpu_group = object()

    def all_reduce(flags, op, group):
        assert group is engine.cpu_group
        assert op == dist.ReduceOp.MAX

    def unexpected_schedule():
        pytest.fail("MCore schedule must not start for unsupported transport padding")

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(engine_module, "get_forward_backward_func", unexpected_schedule)

    with pytest.raises(ValueError, match="Megatron transport padding"):
        engine.forward_backward_batch(
            _microbatches(True), lambda *args: None, forward_only=forward_only
        )
