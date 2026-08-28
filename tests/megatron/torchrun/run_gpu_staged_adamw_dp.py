#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""torchrun worker for staged AdamW MCore DP-shard numerical parity."""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import megatron.core.optimizer.distrib_optimizer as distrib_optimizer_module
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig

from areal.engine.megatron_utils.gpu_staged_optimizer import (
    GPUStagedAdamW,
    GPUStagedAdamWConfig,
    bind_gpu_staged_adamw,
)


class _ModelChunk:
    def __init__(
        self,
        ddp_config: SimpleNamespace,
        param_buffer: torch.Tensor,
        expected: list[torch.Tensor],
    ) -> None:
        self.ddp_config = ddp_config
        self.param_buffer = param_buffer
        self.expected = expected
        self.param_sync_count = 0

    def start_param_sync(self) -> None:
        rank = dist.get_rank()
        shard_numel = self.param_buffer.numel() // dist.get_world_size()
        local_shard = self.param_buffer.narrow(0, rank * shard_numel, shard_numel)
        dist.all_gather_into_tensor(
            self.param_buffer, local_shard.clone(), group=dist.group.WORLD
        )
        expected = torch.cat([param.view(-1) for param in self.expected]).bfloat16()
        torch.testing.assert_close(self.param_buffer, expected, rtol=0.0, atol=0.0)
        self.param_sync_count += 1


def _run_dp2() -> None:
    if dist.get_world_size() != 2:
        raise ValueError("staged AdamW DP worker requires world size 2")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device=device).manual_seed(20260901)
    param_buffer = torch.randn(
        64, generator=generator, device=device, dtype=torch.bfloat16
    )
    first = torch.nn.Parameter(param_buffer[:19])
    second = torch.nn.Parameter(param_buffer[19:])
    model_params = [first, second]
    baseline_params = [
        torch.nn.Parameter(param.detach().float().clone()) for param in model_params
    ]
    adam_kwargs = {
        "lr": 3e-3,
        "betas": (0.8, 0.95),
        "eps": 1e-6,
        "weight_decay": 0.07,
    }
    param_groups = [
        {
            "params": model_params,
            "lr_mult": 1.0,
            "wd_mult": 1.0,
            "is_decoupled_lr": False,
        }
    ]
    inner = GPUStagedAdamW(
        param_groups,
        staged_config=GPUStagedAdamWConfig(
            buffer_count=2,
            bucket_size_mb=11 * 4 / (1024 * 1024),
            update_backend="single",
        ),
        **adam_kwargs,
    )
    ddp_config = SimpleNamespace(use_megatron_fsdp=False, overlap_param_gather=False)
    bucket = SimpleNamespace(
        grad_data=torch.empty_like(param_buffer),
        param_data=param_buffer,
        offset=0,
        numel_unpadded=param_buffer.numel(),
    )
    buffer = SimpleNamespace(
        param_dtype=torch.bfloat16,
        grad_dtype=torch.bfloat16,
        buckets=[bucket],
        param_index_map={first: (0, 19, 0), second: (19, 64, 0)},
        data_parallel_group=dist.group.WORLD,
        data_parallel_world_size=dist.get_world_size(),
        ddp_config=ddp_config,
        params=model_params,
    )
    model_chunk = _ModelChunk(ddp_config, param_buffer, baseline_params)
    config = OptimizerConfig(
        optimizer="adam",
        lr=adam_kwargs["lr"],
        min_lr=0.0,
        weight_decay=adam_kwargs["weight_decay"],
        adam_beta1=adam_kwargs["betas"][0],
        adam_beta2=adam_kwargs["betas"][1],
        adam_eps=adam_kwargs["eps"],
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        main_grads_dtype=torch.bfloat16,
        main_params_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        clip_grad=0.0,
    )

    # The test supplies the synchronous all-gather in _ModelChunk. Bucket-group
    # construction belongs to DDP and is orthogonal to the optimizer shard map.
    distrib_optimizer_module.partition_buckets = lambda _: []
    optimizer = DistributedOptimizer(
        inner,
        config,
        grad_scaler=None,
        init_state_fn=None,
        model_chunks=[model_chunk],
        per_model_buffers={0: [buffer]},
        data_parallel_group=dist.group.WORLD,
        data_parallel_group_gloo=None,
        data_parallel_group_idx=0,
        distributed_optimizer_instance_id=0,
    )
    if bind_gpu_staged_adamw(optimizer) != 1:
        raise AssertionError("staged AdamW was not bound to the MCore DP shards")
    expected_owned_params = 2 if dist.get_rank() == 0 else 1
    actual_owned_params = sum(
        len(group) for group in optimizer.model_float16_groups
    )
    if actual_owned_params != expected_owned_params:
        raise AssertionError(
            f"unexpected DP-local parameter count: {actual_owned_params}"
        )
    baseline = torch.optim.AdamW(baseline_params, **adam_kwargs)
    baseline_by_model_id = {
        id(model_param): baseline_param
        for model_param, baseline_param in zip(
            model_params, baseline_params, strict=True
        )
    }

    for step in range(4):
        for param_index, (model_param, baseline_param) in enumerate(
            zip(model_params, baseline_params, strict=True)
        ):
            grad = torch.randn(
                model_param.shape,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            ).mul_(step + param_index + 1)
            model_param.main_grad = grad
            baseline_param.grad = grad.float()
        baseline.step()

        success, grad_norm, num_zeros = optimizer.step()
        inner.drain()
        if not success or grad_norm != 0.0 or num_zeros != 0:
            raise AssertionError("MCore staged DP optimizer step failed")
        if model_chunk.param_sync_count != step + 1:
            raise AssertionError("parameter all-gather did not run after the update")

        for model_group, owned_group in zip(
            optimizer.model_float16_groups,
            optimizer.shard_float16_groups,
            strict=True,
        ):
            for model_param, owned_param in zip(model_group, owned_group, strict=True):
                baseline_param = baseline_by_model_id[id(model_param)]
                param_range = optimizer._get_model_param_range_map(model_param)["param"]
                expected_slice = slice(param_range.start, param_range.end)
                staged_state = inner.state[owned_param]
                baseline_state = baseline.state[baseline_param]
                torch.testing.assert_close(
                    staged_state["master_param"],
                    baseline_param.detach().cpu()[expected_slice],
                    rtol=2e-6,
                    atol=2e-6,
                )
                for state_name in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        staged_state[state_name],
                        baseline_state[state_name].cpu()[expected_slice],
                        rtol=2e-6,
                        atol=2e-6,
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        _run_dp2()
        dist.barrier()
        if dist.get_rank() == 0:
            with open(args.output, "w") as file:
                file.write("Passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
