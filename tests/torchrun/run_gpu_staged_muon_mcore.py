# SPDX-License-Identifier: Apache-2.0

"""Real-NCCL MCore 0.17 staged-Muon ownership and numerical comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.muon import get_megatron_muon_optimizer

from areal.engine.megatron_utils.gpu_staged_muon import (
    GPUStagedMuonConfig,
    get_megatron_optimizer_with_gpu_staged_muon,
)


class _TinyMuonModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(
            8, 8, bias=True, device="cuda", dtype=torch.bfloat16
        )
        self.norm = torch.nn.LayerNorm(8, device="cuda", dtype=torch.bfloat16)
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=8,
        )
        self.ddp_config = DistributedDataParallelConfig()


class _OnlyMuonModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = torch.nn.Parameter(
            torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
        )
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=4,
        )
        self.ddp_config = DistributedDataParallelConfig()


class _OnlyScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Parameter(
            torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
        )
        self.embedding.is_embedding_or_output_parameter = True
        self.bias = torch.nn.Parameter(
            torch.randn(4, device="cuda", dtype=torch.bfloat16)
        )
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=4,
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
    )


def _owned_param_names(optimizer, model: torch.nn.Module) -> list[str]:
    names = {id(param): name for name, param in model.named_parameters()}
    return [
        names[id(param)]
        for leaf in optimizer.chained_optimizers
        for group in leaf.param_groups
        for param in group["params"]
    ]


def _assert_rank_models_match(model: torch.nn.Module, world_size: int) -> None:
    for param in model.parameters():
        gathered = [torch.empty_like(param) for _ in range(world_size)]
        dist.all_gather(gathered, param)
        for peer in gathered:
            torch.testing.assert_close(peer, param, rtol=0.0, atol=0.0)


def _exercise_edge_model(
    model: torch.nn.Module,
    *,
    world_size: int,
    expected_muon: set[str],
    expected_scalar: set[str],
) -> dict[str, object]:
    optimizer = get_megatron_optimizer_with_gpu_staged_muon(
        _optimizer_config(),
        [model],
        GPUStagedMuonConfig(
            buffer_count=1,
            slot_size_mb=1,
            split_qkv=False,
            fp32_matmul_prec="highest",
            num_ns_steps=3,
            use_nesterov=True,
        ),
    )
    names = {id(param): name for name, param in model.named_parameters()}
    local_by_kind = {
        kind: [
            names[id(param)]
            for leaf in optimizer.chained_optimizers
            if getattr(leaf.optimizer, "optimizer_kind", None) == kind
            for group in leaf.param_groups
            for param in group["params"]
        ]
        for kind in ("muon", "scalar_adamw")
    }
    all_by_kind: list[dict[str, list[str]] | None] = [None] * world_size
    dist.all_gather_object(all_by_kind, local_by_kind)
    global_muon = {
        name for owned in all_by_kind if owned is not None for name in owned["muon"]
    }
    global_scalar = {
        name
        for owned in all_by_kind
        if owned is not None
        for name in owned["scalar_adamw"]
    }
    if global_muon != expected_muon or global_scalar != expected_scalar:
        raise AssertionError(
            "unexpected edge classification: "
            f"muon={global_muon}, scalar={global_scalar}"
        )

    empty_scalar_leaves = [
        leaf.optimizer
        for leaf in optimizer.chained_optimizers
        if getattr(leaf.optimizer, "optimizer_kind", None) == "scalar_adamw"
        and not any(group["params"] for group in leaf.param_groups)
    ]
    for inner in empty_scalar_leaves:
        if (
            inner.cpu_slabs is not None
            or inner.gpu_staging_state_numel != 0
            or inner.state
        ):
            raise AssertionError("empty scalar leaf allocated optimizer storage")

    empty_muon_leaves = [
        leaf.optimizer
        for leaf in optimizer.chained_optimizers
        if getattr(leaf.optimizer, "optimizer_kind", None) == "muon"
        and not any(group["params"] for group in leaf.param_groups)
    ]
    for inner in empty_muon_leaves:
        if inner.cpu_slabs is not None or inner.gpu_staging_numel != 0 or inner.state:
            raise AssertionError("empty Muon leaf allocated optimizer storage")

    nondefault_stream = torch.cuda.Stream()
    for step in range(3):
        for param_index, param in enumerate(model.parameters()):
            accumulated = torch.zeros_like(param, dtype=torch.float32)
            for accumulation in range(2):
                accumulated.add_(
                    0.005 * (step + 1) + 0.003 * accumulation + 0.0001 * param_index
                )
            param.main_grad = accumulated
        if step == 1:
            with torch.cuda.stream(nondefault_stream):
                optimizer.step()
            nondefault_stream.synchronize()
        else:
            optimizer.step()
        optimizer.drain()
        if optimizer.residency != "CPU_RESIDENT" or optimizer.cuda_state_numel != 0:
            raise AssertionError("edge optimizer retained CUDA optimizer state")
        _assert_rank_models_match(model, world_size)

    return {
        "owned_by_kind": local_by_kind,
        "empty_scalar_leaves": len(empty_scalar_leaves),
        "empty_muon_leaves": len(empty_muon_leaves),
        "leaf_order": [
            getattr(leaf.optimizer, "optimizer_kind", None)
            for leaf in optimizer.chained_optimizers
        ],
        "residency": optimizer.residency,
        "cuda_state_numel": optimizer.cuda_state_numel,
    }


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"this acceptance test requires DP=2, got {world_size}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    try:
        torch.manual_seed(20260816)
        staged_model = _TinyMuonModel()
        baseline_model = _TinyMuonModel()
        baseline_model.load_state_dict(staged_model.state_dict())

        staged = get_megatron_optimizer_with_gpu_staged_muon(
            _optimizer_config(),
            [staged_model],
            GPUStagedMuonConfig(
                buffer_count=1,
                slot_size_mb=1,
                split_qkv=False,
                fp32_matmul_prec="highest",
                num_ns_steps=3,
                use_nesterov=True,
            ),
        )
        baseline = get_megatron_muon_optimizer(
            _optimizer_config(),
            [baseline_model],
            use_gloo_process_groups=False,
            layer_wise_distributed_optimizer=True,
        )

        local_owned = _owned_param_names(staged, staged_model)
        names_by_id = {
            id(param): name for name, param in staged_model.named_parameters()
        }
        local_by_kind = {
            kind: [
                names_by_id[id(param)]
                for leaf in staged.chained_optimizers
                if getattr(leaf.optimizer, "optimizer_kind", None) == kind
                for group in leaf.param_groups
                for param in group["params"]
            ]
            for kind in ("muon", "scalar_adamw")
        }
        all_owned: list[list[str] | None] = [None] * world_size
        dist.all_gather_object(all_owned, local_owned)
        all_by_kind: list[dict[str, list[str]] | None] = [None] * world_size
        dist.all_gather_object(all_by_kind, local_by_kind)
        expected_names = {name for name, _ in staged_model.named_parameters()}
        owner_counts = {
            name: sum(name in (owned or []) for owned in all_owned)
            for name in expected_names
        }
        if set(owner_counts.values()) != {1}:
            raise AssertionError(f"invalid layer-wise owner map: {owner_counts}")
        global_muon = {name for owned in all_by_kind if owned for name in owned["muon"]}
        global_scalar = {
            name for owned in all_by_kind if owned for name in owned["scalar_adamw"]
        }
        if global_muon != {"linear.weight"}:
            raise AssertionError(
                f"unexpected official Muon classification: {global_muon}"
            )
        if global_scalar != {
            "linear.bias",
            "norm.weight",
            "norm.bias",
        }:
            raise AssertionError(
                f"unexpected official scalar classification: {global_scalar}"
            )

        staged_muon_units = [
            unit
            for leaf in staged.chained_optimizers
            if getattr(leaf.optimizer, "optimizer_kind", None) == "muon"
            for unit in leaf.optimizer.units
        ]
        if any(unit.numel != unit.param.numel() for unit in staged_muon_units):
            raise AssertionError("a Muon owner matrix was split into partial units")

        max_errors = {
            "model": 0.0,
            "master": 0.0,
            "momentum": 0.0,
            "exp_avg": 0.0,
            "exp_avg_sq": 0.0,
        }
        nondefault_stream_checked = False
        for step in range(3):
            accumulated_by_param: list[torch.Tensor] = []
            for param_index, (staged_param, baseline_param) in enumerate(
                zip(
                    staged_model.parameters(),
                    baseline_model.parameters(),
                    strict=True,
                )
            ):
                accumulated = torch.zeros_like(staged_param, dtype=torch.float32)
                for accumulation in range(2):
                    accumulated.add_(
                        0.005 * (step + 1) + 0.003 * accumulation + 0.0001 * param_index
                    )
                accumulated_by_param.append(accumulated)
                baseline_param.main_grad = accumulated.clone()
            if step == 1:
                caller_stream = torch.cuda.Stream()
                inputs_ready = torch.cuda.Event()
                for staged_param in staged_model.parameters():
                    staged_param.main_grad = torch.zeros_like(
                        staged_param, dtype=torch.float32
                    )
                inputs_ready.record()
                with torch.cuda.stream(caller_stream):
                    caller_stream.wait_event(inputs_ready)
                    torch.cuda._sleep(50_000_000)
                    for staged_param, accumulated in zip(
                        staged_model.parameters(), accumulated_by_param, strict=True
                    ):
                        staged_param.main_grad.copy_(accumulated)
                    staged.step()
                    immediate_model_sums = torch.stack(
                        [param.float().sum() for param in staged_model.parameters()]
                    )
                caller_stream.synchronize()
                torch.testing.assert_close(
                    immediate_model_sums,
                    torch.stack(
                        [param.float().sum() for param in staged_model.parameters()]
                    ),
                    rtol=0.0,
                    atol=0.0,
                )
                nondefault_stream_checked = True
            else:
                for staged_param, accumulated in zip(
                    staged_model.parameters(), accumulated_by_param, strict=True
                ):
                    staged_param.main_grad = accumulated.clone()
                staged.step()
            baseline.step()
            staged.drain()
            if staged.residency != "CPU_RESIDENT" or staged.cuda_state_numel != 0:
                raise AssertionError("staged Muon retained CUDA optimizer state")
            _assert_rank_models_match(staged_model, world_size)

        for staged_param, baseline_param in zip(
            staged_model.parameters(), baseline_model.parameters(), strict=True
        ):
            error = (staged_param.float() - baseline_param.float()).abs().max().item()
            max_errors["model"] = max(max_errors["model"], error)
            torch.testing.assert_close(
                staged_param,
                baseline_param,
                rtol=2e-3,
                atol=2e-3,
            )

        for staged_leaf, baseline_leaf in zip(
            staged.chained_optimizers,
            baseline.chained_optimizers,
            strict=True,
        ):
            for staged_group, baseline_group, baseline_models, baseline_masters in zip(
                staged_leaf.param_groups,
                baseline_leaf.param_groups,
                baseline_leaf.float16_groups,
                baseline_leaf.fp32_from_float16_groups,
                strict=True,
            ):
                if staged_group["lr"] != baseline_group["lr"]:
                    raise AssertionError("staged and baseline learning rates diverged")
                if (
                    getattr(staged_leaf.optimizer, "optimizer_kind", None)
                    == "scalar_adamw"
                ):
                    if (
                        staged_group["step"] != baseline_group["step"]
                        or staged_group["step"] != 3
                    ):
                        raise AssertionError("scalar AdamW step metadata diverged")
                for staged_param, baseline_param, baseline_master in zip(
                    staged_group["params"],
                    baseline_models,
                    baseline_masters,
                    strict=True,
                ):
                    staged_state = staged_leaf.state[staged_param]
                    baseline_state = baseline_leaf.state[baseline_master]
                    comparisons = {
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
                        max_errors[key] = max(max_errors[key], error)
                        tolerance = 2e-3 if key in {"master", "momentum"} else 2e-6
                        torch.testing.assert_close(
                            actual,
                            expected.cpu(),
                            rtol=tolerance,
                            atol=tolerance,
                        )

        torch.manual_seed(20260817)
        only_muon = _exercise_edge_model(
            _OnlyMuonModel(),
            world_size=world_size,
            expected_muon={"matrix"},
            expected_scalar=set(),
        )
        torch.manual_seed(20260818)
        only_scalar = _exercise_edge_model(
            _OnlyScalarModel(),
            world_size=world_size,
            expected_muon=set(),
            expected_scalar={"embedding", "bias"},
        )

        health = torch.ones(1, device="cuda")
        dist.all_reduce(health)
        if health.item() != world_size:
            raise AssertionError("post-step NCCL health probe failed")

        output_dir = Path(os.environ["ACCEPTANCE_OUTPUT_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"rank_{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "owned": local_owned,
                    "owned_by_kind": local_by_kind,
                    "owner_counts": owner_counts,
                    "muon_unit_numels": [unit.numel for unit in staged_muon_units],
                    "muon_cpu_slab_numel": sum(
                        0
                        if leaf.optimizer.cpu_slabs is None
                        else leaf.optimizer.cpu_slabs.master.numel()
                        for leaf in staged.chained_optimizers
                        if getattr(leaf.optimizer, "optimizer_kind", None) == "muon"
                    ),
                    "max_errors": max_errors,
                    "residency": staged.residency,
                    "cuda_state_numel": staged.cuda_state_numel,
                    "accumulation": 2,
                    "steps": 3,
                    "nondefault_stream_checked": nondefault_stream_checked,
                    "only_muon": only_muon,
                    "only_scalar": only_scalar,
                },
                indent=2,
            )
            + "\n"
        )
        dist.barrier()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
