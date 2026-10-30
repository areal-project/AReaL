# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _broadcast_worker(rank, rendezvous):
    from areal.engine.r3 import broadcast as module

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=90),
    )
    module._TRANSFER_BYTES = 32
    for model_rank_sets, stage_rank_sets in [
        ([[0, 1, 2, 3]], [[0, 1], [2, 3]]),
        ([[0, 1], [2, 3]], [[0], [1], [2], [3]]),
    ]:
        model_group = None
        for ranks in model_rank_sets:
            group = dist.new_group(ranks, backend="gloo")
            if rank in ranks:
                model_group, own_ranks = group, ranks
        stages = []
        for index, ranks in enumerate(stage_rank_sets):
            group = dist.new_group(ranks, backend="gloo")
            if ranks[0] in own_ranks:
                stages.append(
                    {
                        "ranks": ranks,
                        "group": group,
                        "layers": (3, 1) if index % 2 == 0 else (2, 0),
                    }
                )
        src = own_ranks[0]
        for dtype in (torch.int16, torch.int32):
            expected = torch.arange(1, 73, dtype=dtype).reshape(9, 4, 2) + src * 100
            payload = None
            if rank == src:
                mask = torch.zeros(2, 5, dtype=torch.bool)
                mask[0, :4], mask[1, :5] = True, True
                payload = (
                    [
                        {
                            "routed_experts": expected,
                            "r3_cu_seqlens": torch.tensor([0, 4, 9]),
                            "r3_routing_valid": torch.ones(2, dtype=torch.bool),
                            "attention_mask": mask,
                        }
                    ],
                    {"ordinary": torch.tensor([src])},
                )
            output, kwargs = module.broadcast_packed_routing(
                payload,
                src_rank=src,
                group=model_group,
                stages=stages,
                num_experts=1024,
            )
            own_stage = next(stage for stage in stages if rank in stage["ranks"])
            layers = own_stage["layers"]
            torch.testing.assert_close(
                output[0]["routed_experts"], expected[:, layers, :], rtol=0, atol=0
            )
            assert output[0]["r3_layer_indices"].tolist() == list(layers)
            assert (
                output[0]["routed_experts"].untyped_storage().nbytes()
                == 9 * 2 * 2 * expected.element_size()
            )
            assert kwargs["ordinary"].tolist() == [src]

        # No routing retains normal RPC semantics and invokes no stage data collectives.
        output = module.broadcast_packed_routing(
            {"ordinary": torch.tensor([src])} if rank == src else None,
            src_rank=src,
            group=model_group,
            stages=stages,
        )
        assert output["ordinary"].tolist() == [src]

        # Source validation is broadcast to every rank before any data transfer.
        invalid = None
        if rank == src:
            invalid = {
                "routed_experts": torch.ones(3, 4, 2, dtype=torch.int16),
                "r3_cu_seqlens": torch.tensor([0, 3]),
                "r3_routing_valid": torch.ones(2, dtype=torch.bool),
            }
        try:
            module.broadcast_packed_routing(
                invalid,
                src_rank=src,
                group=model_group,
                stages=stages,
            )
        except ValueError as exc:
            assert "validity" in str(exc)
        else:
            raise AssertionError("Malformed routing did not fail on every rank")

        # Empty MoE stages receive a genuinely empty allocation and still agree on metadata.
        stages[1]["layers"] = ()
        payload = None
        if rank == src:
            payload = {
                "routed_experts": expected,
                "r3_cu_seqlens": torch.tensor([0, 4, 9]),
                "r3_routing_valid": torch.ones(2, dtype=torch.bool),
            }
        output = module.broadcast_packed_routing(
            payload,
            src_rank=src,
            group=model_group,
            stages=stages,
        )
        own_stage = next(stage for stage in stages if rank in stage["ranks"])
        assert output["routed_experts"].shape == (9, len(own_stage["layers"]), 2)
    dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_packed_broadcast_real_gloo_pipeline_and_data_parallel_isolation(
    tmp_path, monkeypatch
):
    """Exercise actual send/recv with remote PP leaders, DP heads and tiny chunks."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    mp.spawn(
        _broadcast_worker, args=(str(tmp_path / "rendezvous"),), nprocs=4, join=True
    )
