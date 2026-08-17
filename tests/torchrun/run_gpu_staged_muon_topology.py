# SPDX-License-Identifier: Apache-2.0

"""Real MCore 0.17 TP/DP/EP topology checks for staged Muon."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from areal.engine.megatron_utils.gpu_staged_muon import (
    GPUStagedMuonConfig,
    get_megatron_optimizer_with_gpu_staged_muon,
)


def _topology() -> tuple[int, int, int]:
    name = os.environ["MUON_TOPOLOGY"]
    if name == "tp2_dp1":
        return 2, 1, 1
    if name == "tp2_dp2":
        return 2, 1, 1
    if name == "dp2":
        return 1, 1, 1
    if name == "dense_expert":
        return 1, 2, 1
    if name == "expert_only":
        return 1, 2, 1
    raise RuntimeError(f"unsupported Muon topology {name!r}")


class _TopologyMuonModel(torch.nn.Module):
    def __init__(
        self,
        *,
        tp_size: int,
        ep_size: int,
        expert_tp_size: int,
        include_dense: bool,
        include_expert: bool,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__()
        layer_config = ModelParallelConfig(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=expert_tp_size,
            bf16=True,
            params_dtype=torch.bfloat16,
            perform_initialization=True,
            use_cpu_initialization=False,
            gradient_accumulation_fusion=False,
        )

        def init_method(tensor: torch.Tensor) -> None:
            torch.nn.init.uniform_(tensor, -0.1, 0.1)

        if include_dense:
            self.dense = ColumnParallelLinear(
                8,
                8,
                config=layer_config,
                init_method=init_method,
                bias=True,
                gather_output=False,
                tp_group=pg_collection.tp,
            )
        if include_expert:
            self.experts = torch.nn.ModuleList(
                [
                    ColumnParallelLinear(
                        8,
                        8,
                        config=layer_config,
                        init_method=init_method,
                        bias=True,
                        gather_output=False,
                        is_expert=True,
                        tp_group=pg_collection.expt_tp,
                    )
                ]
            )
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=8,
        )
        self.ddp_config = DistributedDataParallelConfig()


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        optimizer="adam",
        lr=2e-3,
        min_lr=0.0,
        weight_decay=0.01,
        bf16=True,
        use_distributed_optimizer=False,
        main_grads_dtype=torch.float32,
        main_params_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        muon_split_qkv=False,
        muon_fp32_matmul_prec="highest",
        muon_num_ns_steps=3,
        muon_use_nesterov=True,
        muon_tp_mode="duplicated",
    )


def _staged_config() -> GPUStagedMuonConfig:
    return GPUStagedMuonConfig(
        buffer_count=1,
        slot_size_mb=1,
        split_qkv=False,
        fp32_matmul_prec="highest",
        num_ns_steps=3,
        use_nesterov=True,
        tp_mode="duplicated",
    )


def _group_ranks(group: dist.ProcessGroup) -> list[int]:
    return list(dist.get_process_group_ranks(group))


def _leaf_kind(leaf: Any) -> str:
    return getattr(leaf.optimizer, "optimizer_kind", "official")


def _owned_names(optimizer: Any, model: torch.nn.Module) -> set[str]:
    names = {id(param): name for name, param in model.named_parameters()}
    return {
        names[id(param)]
        for leaf in optimizer.chained_optimizers
        for group in leaf.param_groups
        for param in group["params"]
    }


def _assert_owner_and_replica_consistency(
    optimizer: Any,
    model: torch.nn.Module,
    pg_collection: ProcessGroupCollection,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    owned = _owned_names(optimizer, model)
    owner_counts: dict[str, int] = {}
    group_members: dict[str, list[int]] = {}
    for name, param in model.named_parameters():
        expert = not getattr(param, "allreduce", True)
        group = pg_collection.expt_dp if expert else pg_collection.dp_cp
        group_name = "expt_dp" if expert else "dp_cp"
        group_members[group_name] = _group_ranks(group)
        count = torch.tensor(
            [int(name in owned)], dtype=torch.int64, device=torch.cuda.current_device()
        )
        dist.all_reduce(count, group=group)
        owner_counts[name] = int(count.item())
        if owner_counts[name] != 1:
            raise AssertionError(
                f"{group_name} owner count for {name} is {owner_counts[name]}"
            )
        replicas = [torch.empty_like(param) for _ in range(group.size())]
        dist.all_gather(replicas, param, group=group)
        for replica in replicas:
            torch.testing.assert_close(replica, param, rtol=0.0, atol=0.0)
    return owner_counts, group_members


def _compare_state(staged: Any, baseline: Any) -> dict[str, float]:
    errors = {
        "master": 0.0,
        "momentum": 0.0,
        "exp_avg": 0.0,
        "exp_avg_sq": 0.0,
    }
    for staged_leaf, baseline_leaf in zip(
        staged.chained_optimizers, baseline.chained_optimizers, strict=True
    ):
        if getattr(baseline_leaf, "is_stub_optimizer", False):
            if any(group["params"] for group in staged_leaf.param_groups):
                raise AssertionError(
                    "official baseline stub does not match a non-empty staged leaf"
                )
            continue
        for staged_group, baseline_models, baseline_masters in zip(
            staged_leaf.param_groups,
            baseline_leaf.float16_groups,
            baseline_leaf.fp32_from_float16_groups,
            strict=True,
        ):
            for staged_param, baseline_param, baseline_master in zip(
                staged_group["params"], baseline_models, baseline_masters, strict=True
            ):
                staged_state = staged_leaf.state[staged_param]
                baseline_state = baseline_leaf.state[baseline_master]
                comparisons: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
                    "master": (staged_state["master_param"], baseline_master)
                }
                if "momentum_buffer" in staged_state:
                    comparisons["momentum"] = (
                        staged_state["momentum_buffer"],
                        baseline_state["momentum_buffer"],
                    )
                else:
                    comparisons["exp_avg"] = (
                        staged_state["exp_avg"],
                        baseline_state["exp_avg"],
                    )
                    comparisons["exp_avg_sq"] = (
                        staged_state["exp_avg_sq"],
                        baseline_state["exp_avg_sq"],
                    )
                for key, (actual, expected) in comparisons.items():
                    error = (actual - expected.cpu()).abs().max().item()
                    errors[key] = max(errors[key], error)
                    tolerance = 2e-3 if key in {"master", "momentum"} else 2e-6
                    torch.testing.assert_close(
                        actual, expected.cpu(), rtol=tolerance, atol=tolerance
                    )
    return errors


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    output_dir = Path(os.environ["ACCEPTANCE_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_trace: list[str] = []

    def mark_phase(phase: str) -> None:
        phase_trace.append(phase)
        (output_dir / f"rank_{rank}.phase.json").write_text(
            json.dumps(phase_trace) + "\n"
        )

    mark_phase("distributed_initialized")
    tp_size, ep_size, expert_tp_size = _topology()
    expected_world_size = (
        2 if os.environ["MUON_TOPOLOGY"] in {"tp2_dp1", "dp2", "expert_only"} else 4
    )
    if world_size != expected_world_size:
        raise RuntimeError(
            f"topology {os.environ['MUON_TOPOLOGY']} needs {expected_world_size} ranks"
        )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=expert_tp_size,
    )
    import megatron.core.optimizer.muon as mcore_muon

    original_newton_schulz_tp = mcore_muon.newton_schulz_tp
    communication_trace: list[dict[str, Any]] = []

    def traced_newton_schulz_tp(*args: Any, **kwargs: Any) -> torch.Tensor:
        group = kwargs["tp_group"]
        communication_trace.append(
            {
                "group_ranks": _group_ranks(group),
                "partition_dim": kwargs["partition_dim"],
                "mode": kwargs["tp_mode"],
            }
        )
        return original_newton_schulz_tp(*args, **kwargs)

    mcore_muon.newton_schulz_tp = traced_newton_schulz_tp
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        include_dense = os.environ["MUON_TOPOLOGY"] != "expert_only"
        include_expert = ep_size > 1
        model_parallel_cuda_manual_seed(20260819, force_reset_rng=True)
        staged_model = _TopologyMuonModel(
            tp_size=tp_size,
            ep_size=ep_size,
            expert_tp_size=expert_tp_size,
            include_dense=include_dense,
            include_expert=include_expert,
            pg_collection=pg_collection,
        )
        model_parallel_cuda_manual_seed(20260819, force_reset_rng=True)
        baseline_model = _TopologyMuonModel(
            tp_size=tp_size,
            ep_size=ep_size,
            expert_tp_size=expert_tp_size,
            include_dense=include_dense,
            include_expert=include_expert,
            pg_collection=pg_collection,
        )
        baseline_model.load_state_dict(staged_model.state_dict())

        torch.cuda.reset_peak_memory_stats()
        mark_phase("staged_build_begin")
        staged = get_megatron_optimizer_with_gpu_staged_muon(
            _optimizer_config(),
            [staged_model],
            _staged_config(),
            pg_collection=pg_collection,
        )
        mark_phase("staged_build_complete")
        baseline = get_megatron_muon_optimizer(
            _optimizer_config(),
            [baseline_model],
            use_gloo_process_groups=False,
            layer_wise_distributed_optimizer=True,
            pg_collection=pg_collection,
        )
        if os.environ["MUON_TOPOLOGY"] == "expert_only":
            # MCore 0.17 constructs the correct expert owner schema, but its stock
            # layer-wise all-gather indexes dense_params[0][0] before detecting
            # that the dense domain is globally empty. Keep the official baseline
            # update path and only mark that already-validated empty domain inactive.
            dense_params = baseline.dp_cp_params_list
            if dense_params is None or any(dense_params):
                raise AssertionError(
                    f"expected an all-empty dense owner schema, got {dense_params!r}"
                )
            baseline.dp_cp_params_list = None
        mark_phase("baseline_build_complete")
        leaf_order = [_leaf_kind(leaf) for leaf in staged.chained_optimizers]
        all_leaf_orders: list[list[str] | None] = [None] * world_size
        dist.all_gather_object(all_leaf_orders, leaf_order)
        if any(order != all_leaf_orders[0] for order in all_leaf_orders[1:]):
            raise AssertionError(
                f"rank-local optimizer trees differ: {all_leaf_orders}"
            )

        participation = os.environ.get("MUON_PARTICIPATION")
        if participation:
            before = [param.detach().clone() for param in staged_model.parameters()]
            muon_base = staged.chained_optimizers[0].optimizer
            state_before = {
                id(param): {
                    key: value.detach().clone()
                    for key, value in muon_base.state[param].items()
                    if isinstance(value, torch.Tensor)
                }
                for param in muon_base.state
            }
            muon_params = [unit.param for unit in muon_base._units]
            scalar_params = [
                param
                for leaf in staged.chained_optimizers[1:]
                for group in leaf.optimizer.param_groups
                for param in group["params"]
            ]
            scalar_before = [
                (
                    leaf.optimizer,
                    [
                        int(group.get("step", 0))
                        for group in leaf.optimizer.param_groups
                    ],
                    {
                        id(param): {
                            key: value.detach().clone()
                            for key, value in leaf.optimizer.state[param].items()
                            if isinstance(value, torch.Tensor)
                        }
                        for param in leaf.optimizer.state
                    },
                )
                for leaf in staged.chained_optimizers[1:]
            ]
            for param in staged_model.parameters():
                param.main_grad = None
                param.grad = None
            data_allgather_count = 0
            partial_tp_group: list[int] | None = None
            if participation == "storage_drift":
                original_all_gather = dist.all_gather

                def counted_all_gather(*args: Any, **kwargs: Any):
                    nonlocal data_allgather_count
                    data_allgather_count += 1
                    return original_all_gather(*args, **kwargs)

                dist.all_gather = counted_all_gather
                if rank == 0:
                    target = next(
                        param for owner in staged.dp_cp_params_list for param in owner
                    )
                    storage = target.untyped_storage()
                    old_pointer = storage.data_ptr()
                    storage.resize_(storage.nbytes() + 8 * 1024 * 1024)
                    if storage.data_ptr() == old_pointer:
                        raise AssertionError(
                            "storage drift injection did not move data"
                        )
            if participation == "partial":
                local_tp_members = _group_ranks(pg_collection.tp)
                ownership: list[tuple[list[int], bool] | None] = [None] * world_size
                dist.all_gather_object(
                    ownership,
                    (local_tp_members, bool(muon_params)),
                    group=dist.group.WORLD,
                )
                groups: dict[tuple[int, ...], dict[int, bool]] = {}
                for owner_rank, item in enumerate(ownership):
                    if item is None:
                        raise AssertionError("missing TP ownership report")
                    members, has_muon = item
                    groups.setdefault(tuple(members), {})[owner_rank] = has_muon
                candidates = [
                    members
                    for members, flags in sorted(groups.items())
                    if any(flags.get(member, False) for member in members)
                ]
                if not candidates:
                    raise AssertionError(
                        "partial participation has no Muon owner group"
                    )
                partial_tp_group = list(candidates[0])
                injection_rank = next(
                    member
                    for member in partial_tp_group
                    if groups[tuple(partial_tp_group)].get(member, False)
                )
                if rank == injection_rank:
                    for param_index, param in enumerate(muon_params):
                        param.grad = torch.full_like(param, 0.01 * (param_index + 1))
            elif participation == "muon_only":
                for param_index, param in enumerate(muon_params):
                    param.grad = torch.full_like(param, 0.01 * (param_index + 1))
            elif participation == "scalar_only":
                for param_index, param in enumerate(scalar_params):
                    param.grad = torch.full_like(param, 0.02 * (param_index + 1))
            if participation in {"muon_only", "scalar_only"}:
                for param_index, param in enumerate(baseline_model.parameters()):
                    selected = (
                        param.ndim == 2
                        if participation == "muon_only"
                        else param.ndim != 2
                    )
                    param.main_grad = (
                        torch.full_like(param, 0.01 * (param_index + 1))
                        if selected
                        # The stock mixed-precision wrapper requires main_grad
                        # for every parameter once any chained leaf steps.  A
                        # zero baseline gradient models the staged leaf skip
                        # without entering its invalid None-gradient path.
                        else torch.zeros_like(param)
                    )
            failure: str | None = None
            try:
                # Exercise the public production path so participation is proven
                # before Megatron gradient norm/clipping and Newton--Schulz.
                staged.step()
                if participation in {"muon_only", "scalar_only"}:
                    baseline.step()
            except RuntimeError as error:
                failure = str(error)
            staged.drain()
            if participation in {"all_none", "partial", "storage_drift"}:
                for actual, expected in zip(
                    staged_model.parameters(), before, strict=True
                ):
                    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
                for param, expected_state in (
                    (param, state_before[id(param)]) for param in muon_base.state
                ):
                    for key, expected in expected_state.items():
                        torch.testing.assert_close(
                            muon_base.state[param][key],
                            expected,
                            rtol=0.0,
                            atol=0.0,
                        )
            else:
                if failure is not None:
                    raise AssertionError(
                        f"{participation} public step failed: {failure}"
                    )
                for actual, expected in zip(
                    staged_model.parameters(), baseline_model.parameters(), strict=True
                ):
                    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
            if participation in {
                "all_none",
                "partial",
                "muon_only",
                "storage_drift",
            }:
                for scalar, steps, states in scalar_before:
                    if [
                        int(group.get("step", 0)) for group in scalar.param_groups
                    ] != steps:
                        raise AssertionError("inactive scalar step metadata changed")
                    for param in scalar.state:
                        for key, expected in states[id(param)].items():
                            torch.testing.assert_close(
                                scalar.state[param][key],
                                expected,
                                rtol=0.0,
                                atol=0.0,
                            )
            if participation == "scalar_only":
                for param, expected_state in (
                    (param, state_before[id(param)]) for param in muon_base.state
                ):
                    for key, expected in expected_state.items():
                        torch.testing.assert_close(
                            muon_base.state[param][key],
                            expected,
                            rtol=0.0,
                            atol=0.0,
                        )
            if participation == "all_none":
                if failure is not None:
                    raise AssertionError(f"all-none participation failed: {failure}")
            elif participation == "partial":
                if failure is None or (
                    "inconsistent gradient participation" not in failure
                    and "step preflight failed on another rank" not in failure
                ):
                    raise AssertionError(
                        f"partial TP participation was not rejected: {failure!r}"
                    )
            elif participation == "storage_drift":
                if failure is None or (
                    "owner metadata" not in failure
                    and "parameter metadata changed" not in failure
                ):
                    raise AssertionError(f"storage drift was not rejected: {failure!r}")
                if data_allgather_count != 0:
                    raise AssertionError(
                        "storage drift reached parameter data all-gather: "
                        f"count={data_allgather_count}"
                    )
            elif participation not in {"muon_only", "scalar_only"}:
                raise RuntimeError(f"unknown participation scenario {participation!r}")
            health = torch.ones(1, device="cuda")
            dist.all_reduce(health, group=dist.group.WORLD)
            if health.item() != world_size:
                raise AssertionError("post-participation NCCL health probe failed")
            (output_dir / f"rank_{rank}.json").write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "participation": participation,
                        "failure": failure,
                        "residency": staged.residency,
                        "cuda_state_numel": staged.cuda_state_numel,
                        "communication_trace": communication_trace,
                        "data_allgather_count": data_allgather_count,
                        "partial_tp_group": partial_tp_group,
                        "health": int(health.item()),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            dist.barrier(group=dist.group.WORLD)
            return

        max_model_error = 0.0
        nondefault_stream_checked = False
        for step in range(3):
            mark_phase(f"step_{step}_begin")
            accumulated: list[torch.Tensor] = []
            for param_index, (staged_param, baseline_param) in enumerate(
                zip(staged_model.parameters(), baseline_model.parameters(), strict=True)
            ):
                domain_group = (
                    pg_collection.expt_tp
                    if getattr(staged_param, "expert_tp", False)
                    else pg_collection.tp
                )
                value = (
                    0.005 * (step + 1)
                    + 0.003 * domain_group.rank()
                    + 0.0001 * param_index
                )
                grad = torch.zeros_like(staged_param, dtype=torch.float32)
                for accumulation in range(2):
                    grad.add_(value + 0.0002 * accumulation)
                accumulated.append(grad)
                baseline_param.main_grad = grad.clone()

            if step == 1:
                caller_stream = torch.cuda.Stream()
                with torch.cuda.stream(caller_stream):
                    torch.cuda._sleep(20_000_000)
                    for param, grad in zip(
                        staged_model.parameters(), accumulated, strict=True
                    ):
                        param.main_grad = grad.clone()
                    staged.step()
                    immediately_consumed = torch.stack(
                        [param.float().sum() for param in staged_model.parameters()]
                    )
                caller_stream.synchronize()
                torch.testing.assert_close(
                    immediately_consumed,
                    torch.stack(
                        [param.float().sum() for param in staged_model.parameters()]
                    ),
                    rtol=0.0,
                    atol=0.0,
                )
                nondefault_stream_checked = True
            else:
                for param, grad in zip(
                    staged_model.parameters(), accumulated, strict=True
                ):
                    param.main_grad = grad.clone()
                staged.step()
            mark_phase(f"step_{step}_staged_complete")
            baseline.step()
            mark_phase(f"step_{step}_baseline_complete")
            staged.drain()
            if staged.residency != "CPU_RESIDENT" or staged.cuda_state_numel != 0:
                raise AssertionError("staged Muon retained CUDA optimizer state")
            _assert_owner_and_replica_consistency(staged, staged_model, pg_collection)
            mark_phase(f"step_{step}_replica_check_complete")

        mark_phase("model_compare_begin")
        for staged_param, baseline_param in zip(
            staged_model.parameters(), baseline_model.parameters(), strict=True
        ):
            error = (staged_param.float() - baseline_param.float()).abs().max().item()
            max_model_error = max(max_model_error, error)
            torch.testing.assert_close(
                staged_param, baseline_param, rtol=2e-3, atol=2e-3
            )
        mark_phase("model_compare_complete")
        state_errors = _compare_state(staged, baseline)
        mark_phase("state_compare_complete")
        owner_counts, group_members = _assert_owner_and_replica_consistency(
            staged, staged_model, pg_collection
        )
        mark_phase("final_replica_check_complete")
        all_communication_traces: list[list[dict[str, Any]] | None] = [
            None
        ] * world_size
        dist.all_gather_object(
            all_communication_traces,
            communication_trace,
            group=dist.group.WORLD,
        )
        flattened_traces = [
            entry
            for rank_trace in all_communication_traces
            for entry in rank_trace or []
        ]
        if tp_size > 1 and not flattened_traces:
            raise AssertionError(
                "official newton_schulz_tp communication was not entered"
            )
        if tp_size > 1 and not any(
            entry["partition_dim"] in {0, 1} for entry in flattened_traces
        ):
            raise AssertionError(
                "official newton_schulz_tp never received TP partition metadata"
            )
        local_expected_tp_groups = {
            tuple(_group_ranks(pg_collection.tp)),
            tuple(_group_ranks(pg_collection.expt_tp)),
        }
        gathered_expected_tp_groups: list[set[tuple[int, ...]] | None] = [
            None
        ] * world_size
        dist.all_gather_object(
            gathered_expected_tp_groups,
            local_expected_tp_groups,
            group=dist.group.WORLD,
        )
        expected_tp_groups = set().union(
            *(groups or set() for groups in gathered_expected_tp_groups)
        )
        if any(
            tuple(entry["group_ranks"]) not in expected_tp_groups
            for entry in flattened_traces
        ):
            raise AssertionError(
                f"Muon used an unexpected TP group: {flattened_traces}"
            )

        mark_phase("health_probe_begin")
        health = torch.ones(1, device="cuda")
        dist.all_reduce(health)
        if health.item() != world_size:
            raise AssertionError("post-topology NCCL health probe failed")
        mark_phase("health_probe_complete")

        cpu_slab_numel = sum(
            0
            if leaf.optimizer.cpu_slabs is None
            else leaf.optimizer.cpu_slabs.master.numel()
            for leaf in staged.chained_optimizers
        )
        (output_dir / f"rank_{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "topology": os.environ["MUON_TOPOLOGY"],
                    "leaf_order": leaf_order,
                    "owner_counts": owner_counts,
                    "owned": sorted(_owned_names(staged, staged_model)),
                    "group_members": group_members,
                    "tp_group": _group_ranks(pg_collection.tp),
                    "expert_tp_group": _group_ranks(pg_collection.expt_tp),
                    "communication_trace": communication_trace,
                    "max_errors": {"model": max_model_error, **state_errors},
                    "cpu_slab_numel": cpu_slab_numel,
                    "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                    "cuda_state_numel": staged.cuda_state_numel,
                    "residency": staged.residency,
                    "nondefault_stream_checked": nondefault_stream_checked,
                    "phase_trace": phase_trace,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        mark_phase("final_barrier_begin")
        dist.barrier()
        mark_phase("final_barrier_complete")
    except BaseException:
        (output_dir / f"rank_{rank}.error.txt").write_text(traceback.format_exc())
        raise
    finally:
        mcore_muon.newton_schulz_tp = original_newton_schulz_tp
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
