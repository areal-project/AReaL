#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""torchrun worker for staged Muon layer-wise DP numerical parity."""

from __future__ import annotations

import argparse
import copy
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection

from areal.engine.megatron_utils.gpu_staged_muon import (
    GPUStagedMuonConfig,
    get_megatron_optimizer_with_gpu_staged_muon,
)


class _TinyMuonModel(torch.nn.Module):
    def __init__(self, initial: dict[str, torch.Tensor], tp_size: int) -> None:
        super().__init__()
        self.dense_weight = torch.nn.Parameter(initial["dense_weight"].clone())
        self.experts_weight = torch.nn.Parameter(initial["experts_weight"].clone())
        self.experts_weight.allreduce = False
        self.bias = torch.nn.Parameter(initial["bias"].clone())
        self.config = SimpleNamespace(
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=2,
            context_parallel_size=1,
        )
        self.ddp_config = SimpleNamespace(
            use_megatron_fsdp=False,
            use_distributed_optimizer=False,
            num_distributed_optimizer_instances=1,
        )
        if tp_size > 1:
            self.dense_weight.tensor_model_parallel = True
            self.dense_weight.partition_dim = 0
            self.dense_weight.partition_stride = 1
            self.experts_weight.tensor_model_parallel = False
            self.experts_weight.partition_dim = 0
            self.experts_weight.partition_stride = 1


def _make_pg_collection() -> ProcessGroupCollection:
    return ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=[
            "tp",
            "expt_tp",
            "pp",
            "cp",
            "ep",
            "mp",
            "tp_ep_pp",
            "dp",
            "dp_cp",
            "expt_dp",
        ]
    )


def _make_config() -> OptimizerConfig:
    return OptimizerConfig(
        optimizer="dist_muon",
        lr=0.02,
        min_lr=0.0,
        weight_decay=0.015,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_eps=1e-6,
        bf16=True,
        fp16=False,
        use_distributed_optimizer=False,
        use_precision_aware_optimizer=False,
        decoupled_weight_decay=True,
        muon_scalar_optimizer="adam",
        muon_momentum=0.82,
        muon_use_nesterov=True,
        muon_split_qkv=False,
        muon_fp32_matmul_prec="highest",
        muon_num_ns_steps=3,
        muon_tp_mode="duplicated",
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        clip_grad=0.0,
    )


def _run_dp2_outer(tp_size: int) -> None:
    if dist.get_world_size() != 2 * tp_size:
        raise ValueError("staged Muon outer worker requires DP size 2")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    generator = torch.Generator(device=device).manual_seed(20260902 + tp_rank)
    initial = {
        "dense_weight": torch.randn(
            5, 7, generator=generator, device=device, dtype=torch.bfloat16
        ),
        "experts_weight": torch.randn(
            6, 4, generator=generator, device=device, dtype=torch.bfloat16
        ),
        "bias": torch.randn(
            5, generator=generator, device=device, dtype=torch.bfloat16
        ),
    }
    staged_model = _TinyMuonModel(initial, tp_size).to(device)
    baseline_model = _TinyMuonModel(initial, tp_size).to(device)
    pg_collection = _make_pg_collection()

    staged_config = _make_config()
    staged = get_megatron_optimizer_with_gpu_staged_muon(
        staged_config,
        [staged_model],
        GPUStagedMuonConfig(buffer_count=1, slot_size_mb=1),
        pg_collection=pg_collection,
    )
    baseline_config = _make_config()
    baseline_build_config = copy.copy(baseline_config)
    baseline = get_megatron_muon_optimizer(
        baseline_build_config,
        [baseline_model],
        use_gloo_process_groups=False,
        layer_wise_distributed_optimizer=True,
        pg_collection=pg_collection,
    )

    if staged.dp_cp_params_list is None or staged.expt_dp_params_list is None:
        raise AssertionError("DP2 must materialize both dense and expert owner lists")
    if not any(len(owner) == 0 for owner in staged.expt_dp_params_list):
        raise AssertionError("expert DP case must contain an empty-owner rank")
    staged_kinds = [leaf.optimizer.optimizer_kind for leaf in staged.chained_optimizers]
    if staged_kinds != ["muon", "scalar_adamw"]:
        raise AssertionError(f"unexpected staged optimizer chain: {staged_kinds}")

    staged_by_name = dict(staged_model.named_parameters())
    baseline_by_name = dict(baseline_model.named_parameters())
    for step in range(3):
        for param_index, name in enumerate(staged_by_name):
            grad = torch.randn(
                staged_by_name[name].shape,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            ).mul_(step + param_index + 1)
            staged_by_name[name].main_grad = grad
            baseline_by_name[name].main_grad = grad.clone()

        baseline_success, _, _ = baseline.step()
        staged_success, _, _ = staged.step()
        if not baseline_success or not staged_success:
            raise AssertionError("layer-wise optimizer step failed")
        for name, staged_param in staged_by_name.items():
            torch.testing.assert_close(
                staged_param,
                baseline_by_name[name],
                rtol=4e-3,
                atol=4e-3,
            )
            replicas = [
                torch.empty_like(staged_param)
                for _ in range(dist.get_world_size(pg_collection.dp))
            ]
            dist.all_gather(replicas, staged_param.detach(), group=pg_collection.dp)
            for replica in replicas[1:]:
                torch.testing.assert_close(replica, replicas[0], rtol=0.0, atol=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=args.tp_size,
        create_gloo_process_groups=False,
    )
    try:
        _run_dp2_outer(args.tp_size)
        dist.barrier()
        if dist.get_rank() == 0:
            with open(args.output, "w") as file:
                file.write("Passed")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
