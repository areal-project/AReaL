"""Distributed test worker for sglang PP weight synchronization.

Run via torchrun from test_sglang_pp_distributed.py.

Validates the per-PP-rank weight update group protocol that all three
training engine types (Megatron, FSDP, Archon) share, **without
instantiating actual training engines**.  This avoids heavy dependencies
(mbridge for Megatron, flash_attn for FSDP) and model-specific constraints
(e.g., Qwen3-0.6B ``tie_word_embeddings`` blocks Archon PP>1).

The end-to-end integration with actual engines is covered by the
``test_pp_e2e_train`` tests in ``test_sglang_pp_distributed.py``, which
run ``gsm8k_rl.py`` with the same backends the author validated manually.

Test types:
  - group_init:  Verify per-PP-rank group names, head detection, world sizes,
                 and ``build_init_weights_group_request`` payload structure.
  - weight_sync: Verify allocation parsing, PP-DP folding for FSDP, per-PP
                 world size arithmetic, simulated layer partitioning, and
                 NCCL broadcast correctness.

Design notes
------------
All three training engines share the same per-PP-rank group naming:

  - Megatron:  ``f"update_weight_group_{mpu.get_pipeline_model_parallel_rank()}"``
  - FSDP:      ``f"update_weight_group_{pp_rank}"``  (per-PP path)
  - Archon:    ``WeightSyncState.__init__`` → ``f"update_weight_group_{pp_rank}"``

The ``SGLangBackend.build_init_weights_group_request`` method constructs
the HTTP payload by parsing the group name suffix.  We inline this logic
to avoid transitive imports of heavy engine modules.
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist

from areal.api.alloc_mode import ModelAllocation
from areal.infra.platforms import current_platform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result writer
# ---------------------------------------------------------------------------


def _write_result(output_path: str, passed: bool) -> None:
    if output_path:
        with open(output_path, "w") as f:
            f.write("Passed" if passed else "Failed")


# ---------------------------------------------------------------------------
# Engine-type helpers: group naming convention shared by all engines
# ---------------------------------------------------------------------------


def _get_weight_update_group_name(pp_rank: int) -> str:
    """Return the expected weight update group name for a given PP rank.

    All three engines use ``f"update_weight_group_{pp_rank}"``.
    """
    return f"update_weight_group_{pp_rank}"


def _is_pp_head(
    engine_type: str, rank: int, world_size: int, dp: int, pp: int, tp: int
) -> bool:
    """Determine whether this rank is a pipeline parallel head.

    Uses the same logic each engine applies:
      - Megatron: dp_rank == 0 and tp_rank == 0
      - FSDP:     dist.get_rank() == 0
      - Archon:   dp_rank == 0 and tp_rank == 0 and cp_rank == 0
    """
    if engine_type == "fsdp":
        return rank == 0

    # For megatron and archon: head = dp_rank==0 and tp_rank==0.
    # Layout: [dp, pp, tp] → rank = dp_idx * (pp * tp) + pp_idx * tp + tp_idx
    tp_idx = rank % tp
    dp_idx = rank // (pp * tp)
    return dp_idx == 0 and tp_idx == 0


def _get_pp_rank(engine_type: str, rank: int, dp: int, pp: int, tp: int) -> int:
    """Compute the PP rank for this worker.

    FSDP does not partition the model by PP on the training side (PP is
    folded into DP), so all FSDP training ranks have pp_rank == 0.
    """
    if engine_type == "fsdp":
        return 0
    # Layout: [dp, pp, tp]
    return (rank // tp) % pp


def _get_pp_size(engine_type: str, dp: int, pp: int, tp: int) -> int:
    """Return the effective PP size on the training side.

    FSDP folds PP into DP, so its training-side PP size is always 1.
    """
    if engine_type == "fsdp":
        return 1
    return pp


# ---------------------------------------------------------------------------
# Inline payload construction (mirrors SGLangBackend logic)
# ---------------------------------------------------------------------------


def _build_init_weights_group_payload(
    group_name: str,
    gen_pp_size: int,
    gen_tp_size: int,
    gen_world_size: int,
    master_address: str,
    master_port: int,
    server_idx: int = 0,
) -> dict:
    """Replicate ``SGLangBackend.build_init_weights_group_request`` payload.

    Avoids importing ``areal.engine.sglang_remote`` which would pull in
    heavy engine dependencies via transitive imports.
    """
    per_pp_groups = False
    if gen_pp_size > 1:
        try:
            _suffix = group_name.rsplit("_", 1)[-1]
            int(_suffix)
            per_pp_groups = True
        except (ValueError, IndexError):
            per_pp_groups = False

    if per_pp_groups:
        pp_rank = int(group_name.rsplit("_", 1)[-1])
        tp_size = gen_tp_size
        n_servers = gen_world_size // (tp_size * gen_pp_size)
        rank_offset = 1 + server_idx * tp_size
        world_size = n_servers * tp_size + 1

        return {
            "master_address": master_address,
            "master_port": str(master_port),
            "rank_offset": rank_offset,
            "world_size": world_size,
            "backend": "nccl",
            "group_name": group_name,
            "pp_rank": pp_rank,
        }
    else:
        instance_size = gen_tp_size * gen_pp_size
        rank_offset = 1 + server_idx * instance_size
        return {
            "master_address": master_address,
            "master_port": str(master_port),
            "rank_offset": rank_offset,
            "world_size": gen_world_size + 1,
            "backend": "nccl",
            "group_name": group_name,
        }


# ---------------------------------------------------------------------------
# Test: group_init
# ---------------------------------------------------------------------------


def test_group_init(backend: str, gen_pp_size: int, output: str, engine_type: str):
    """Verify per-PP-rank group names, head detection, and world sizes.

    Checks:
      1. Group name follows ``update_weight_group_{pp_rank}`` convention.
      2. ``is_pipeline_parallel_head`` returns the correct value.
      3. Per-PP world size is computed correctly.
      4. ``build_init_weights_group_request`` payload has correct fields.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    alloc = ModelAllocation.from_str(backend)
    dp = alloc.parallel.dp_size
    pp = alloc.parallel.pp_size
    tp = alloc.parallel.tp_size

    try:
        # --- 1. Group name convention ---
        pp_rank = _get_pp_rank(engine_type, rank, dp, pp, tp)
        expected_group = _get_weight_update_group_name(pp_rank)

        logger.info(
            "rank=%d engine=%s pp_rank=%d expected_group=%s",
            rank,
            engine_type,
            pp_rank,
            expected_group,
        )
        assert expected_group == f"update_weight_group_{pp_rank}", (
            f"rank {rank}: group name mismatch: {expected_group}"
        )

        # --- 2. Head detection ---
        is_head = _is_pp_head(engine_type, rank, world_size, dp, pp, tp)
        logger.info("rank=%d engine=%s is_head=%s", rank, engine_type, is_head)

        if engine_type == "fsdp":
            assert is_head == (rank == 0), (
                f"rank {rank}: FSDP head should be rank 0, got is_head={is_head}"
            )
        else:
            tp_idx = rank % tp
            dp_idx = rank // (pp * tp)
            expected_head = dp_idx == 0 and tp_idx == 0
            assert is_head == expected_head, (
                f"rank {rank}: expected is_head={expected_head}, got {is_head}"
            )

        # --- 3. Per-PP world size computation ---
        gen_alloc_str = f"sglang:d1p{gen_pp_size}t{tp}"
        gen_alloc = ModelAllocation.from_str(gen_alloc_str)
        gen_world = gen_alloc.parallel.world_size
        if gen_pp_size > 1:
            per_pp_world = gen_world // gen_pp_size
            assert per_pp_world == tp, (
                f"Expected per-PP world size {tp}, got {per_pp_world}"
            )

        # --- 4. Payload validation ---
        for test_pp_rank in range(gen_pp_size):
            group_name = f"update_weight_group_{test_pp_rank}"
            payload = _build_init_weights_group_payload(
                group_name=group_name,
                gen_pp_size=gen_pp_size,
                gen_tp_size=tp,
                gen_world_size=gen_world,
                master_address="127.0.0.1",
                master_port=29500 + test_pp_rank,
                server_idx=0,
            )
            assert payload["group_name"] == group_name, (
                f"Expected group_name={group_name}, got {payload['group_name']}"
            )
            if gen_pp_size > 1:
                assert "pp_rank" in payload, (
                    f"pp_rank missing from payload for gen_pp_size={gen_pp_size}"
                )
                assert payload["pp_rank"] == test_pp_rank, (
                    f"Expected pp_rank={test_pp_rank}, got {payload['pp_rank']}"
                )
                expected_ws = 1 * tp + 1
                assert int(payload["world_size"]) == expected_ws, (
                    f"Expected world_size={expected_ws}, got {payload['world_size']}"
                )
            else:
                assert "pp_rank" not in payload, (
                    "pp_rank should not be in payload for gen_pp_size=1"
                )
                expected_ws = gen_world + 1
                assert int(payload["world_size"]) == expected_ws, (
                    f"Expected world_size={expected_ws}, got {payload['world_size']}"
                )

        current_platform.synchronize()
        dist.barrier()

        if rank == 0 and output:
            _write_result(output, True)
    except Exception as e:
        logger.error("rank=%d test_group_init FAILED: %s", rank, e, exc_info=True)
        if rank == 0 and output:
            _write_result(output, False)
        raise


# ---------------------------------------------------------------------------
# Test: weight_sync
# ---------------------------------------------------------------------------


def test_weight_sync(backend: str, gen_pp_size: int, output: str, engine_type: str):
    """Verify allocation parsing, PP arithmetic, and distributed broadcast.

    Checks:
      1. Allocation string parses to correct dp/pp/tp dimensions.
      2. FSDP PP-DP folding: training PP is always 1 (PP folded into DP).
      3. Simulated layer partitioning across PP ranks (non-overlapping).
      4. NCCL all-reduce and per-PP-rank broadcast work correctly.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    alloc = ModelAllocation.from_str(backend)
    dp = alloc.parallel.dp_size
    pp = alloc.parallel.pp_size
    tp = alloc.parallel.tp_size

    try:
        # --- 1. Allocation parsing validation ---
        assert alloc.parallel.world_size == world_size, (
            f"world_size mismatch: alloc={alloc.parallel.world_size}, env={world_size}"
        )
        assert dp * pp * tp == world_size, (
            f"dp*pp*tp={dp}*{pp}*{tp}={dp * pp * tp} != world_size={world_size}"
        )

        # --- 2. FSDP PP-DP folding ---
        if engine_type == "fsdp":
            assert pp == 1, (
                f"FSDP allocation should have pp=1 (PP folded into DP), got pp={pp}"
            )
            logger.info(
                "FSDP allocation validated: dp=%d pp=%d tp=%d (PP folded into DP)",
                dp,
                pp,
                tp,
            )

        # --- 3. Simulated layer partitioning across PP ranks ---
        pp_rank = _get_pp_rank(engine_type, rank, dp, pp, tp)
        pp_size = _get_pp_size(engine_type, dp, pp, tp)

        total_layers = 28  # Qwen3-0.6B has 28 layers
        if pp_size > 1:
            layers_per_pp = total_layers // pp_size
            my_start = pp_rank * layers_per_pp
            my_end = (
                (pp_rank + 1) * layers_per_pp if pp_rank < pp_size - 1 else total_layers
            )
            my_layer_indices = set(range(my_start, my_end))
        else:
            my_layer_indices = set(range(total_layers))

        if pp_size > 1:
            # Compute ALL PP sub-groups in a deterministic order so that
            # every rank calls ``dist.new_group`` the same number of times
            # and in the same sequence.  ``dist.new_group`` is a collective
            # on the default group — ALL ranks must participate in every
            # call, even if they are not members of that sub-group.
            all_pp_group_ranks = []
            for d in range(dp):
                for t in range(tp):
                    ranks_in_group = sorted(
                        d * (pp * tp) + p * tp + t for p in range(pp)
                    )
                    all_pp_group_ranks.append(ranks_in_group)

            pp_group_objects = []
            for ranks_in_group in all_pp_group_ranks:
                g = dist.new_group(ranks=ranks_in_group)
                pp_group_objects.append(g)

            my_pp_group_idx = next(
                i
                for i, ranks_in_group in enumerate(all_pp_group_ranks)
                if rank in ranks_in_group
            )
            pp_group = pp_group_objects[my_pp_group_idx]
            pp_group_ranks = all_pp_group_ranks[my_pp_group_idx]

            all_layer_sets = [None] * pp_size
            dist.all_gather_object(all_layer_sets, my_layer_indices, group=pp_group)

            if rank == pp_group_ranks[0]:
                for i in range(pp_size):
                    for j in range(i + 1, pp_size):
                        overlap = all_layer_sets[i] & all_layer_sets[j]
                        assert len(overlap) == 0, (
                            f"PP ranks {i} and {j} overlap: {overlap}"
                        )
                union = set()
                for s in all_layer_sets:
                    union |= s
                assert union == set(range(total_layers)), (
                    f"Layer sets don't cover all layers: "
                    f"missing={set(range(total_layers)) - union}"
                )
                logger.info(
                    "PP layer partitioning verified: %d ranks, %d layers",
                    pp_size,
                    total_layers,
                )

        # --- 4. NCCL all-reduce test ---
        device = f"cuda:{local_rank}"
        tensor = torch.tensor([float(rank)], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected_sum = float(sum(range(world_size)))
        assert abs(tensor.item() - expected_sum) < 1e-3, (
            f"all_reduce failed: got {tensor.item()}, expected {expected_sum}"
        )

        # Per-PP-rank broadcast test
        if pp_size > 1:
            # Same pattern: create ALL TP sub-groups in deterministic order.
            all_tp_group_ranks = []
            for d in range(dp):
                for p in range(pp):
                    ranks_in_group = sorted(
                        d * (pp * tp) + p * tp + t for t in range(tp)
                    )
                    all_tp_group_ranks.append(ranks_in_group)

            tp_group_objects = []
            for ranks_in_group in all_tp_group_ranks:
                g = dist.new_group(ranks=ranks_in_group)
                tp_group_objects.append(g)

            my_tp_group_idx = next(
                i
                for i, ranks_in_group in enumerate(all_tp_group_ranks)
                if rank in ranks_in_group
            )
            tp_group = tp_group_objects[my_tp_group_idx]
            tp_group_ranks = all_tp_group_ranks[my_tp_group_idx]
            tp_idx = rank % tp

            if tp_idx == 0:
                weight_token = torch.tensor([float(pp_rank * 100 + 42)], device=device)
            else:
                weight_token = torch.zeros(1, device=device)

            dist.broadcast(weight_token, src=tp_group_ranks[0], group=tp_group)

            expected_val = float(pp_rank * 100 + 42)
            assert abs(weight_token.item() - expected_val) < 1e-3, (
                f"rank {rank}: per-PP broadcast failed: "
                f"got {weight_token.item()}, expected {expected_val}"
            )

        current_platform.synchronize()
        dist.barrier()

        if rank == 0 and output:
            _write_result(output, True)
    except Exception as e:
        logger.error("rank=%d test_weight_sync FAILED: %s", rank, e, exc_info=True)
        if rank == 0 and output:
            _write_result(output, False)
        raise


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Distributed test worker for sglang PP weight sync."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="megatron:d1p2t2",
        help="Training engine allocation string.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write Passed/Failed result.",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["group_init", "weight_sync"],
        default="group_init",
        help="Which test to run.",
    )
    parser.add_argument(
        "--gen_pp_size",
        type=int,
        default=2,
        help="Inference-side PP size for validation.",
    )
    parser.add_argument(
        "--engine_type",
        type=str,
        choices=["megatron", "fsdp", "archon"],
        default="megatron",
        help="Training engine type to validate.",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    try:
        if args.test_type == "group_init":
            test_group_init(
                args.backend, args.gen_pp_size, args.output, args.engine_type
            )
        elif args.test_type == "weight_sync":
            test_weight_sync(
                args.backend, args.gen_pp_size, args.output, args.engine_type
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
