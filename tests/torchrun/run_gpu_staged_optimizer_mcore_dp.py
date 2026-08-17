# SPDX-License-Identifier: Apache-2.0

"""Acceptance-only real-NCCL Megatron DistributedOptimizer comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

from areal.engine.megatron_utils.gpu_staged_optimizer import (
    GPUStagedAdamW,
    GPUStagedAdamWConfig,
    bind_gpu_staged_adamw,
)


class _ModelChunk:
    def __init__(self, ddp_config, model_param: torch.nn.Parameter):
        self.ddp_config = ddp_config
        self.model_param = model_param
        self.owned_shard: torch.Tensor | None = None
        self.sync_count = 0

    def start_param_sync(self) -> None:
        assert self.owned_shard is not None
        dist.all_gather_into_tensor(self.model_param.data, self.owned_shard)
        self.sync_count += 1


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")
    numel = 96
    assert numel % world_size == 0
    local_numel = numel // world_size

    torch.manual_seed(20260813)
    initial = torch.randn(numel, device=device, dtype=torch.bfloat16)
    dist.broadcast(initial, src=0)
    model_param = torch.nn.Parameter(initial.clone())
    baseline_param = torch.nn.Parameter(initial.float().clone())
    kwargs = {
        "lr": 3e-3,
        "betas": (0.8, 0.95),
        "eps": 1e-6,
        "weight_decay": 0.07,
    }
    inner = GPUStagedAdamW(
        [
            {
                "params": [model_param],
                "lr_mult": 1.0,
                "wd_mult": 1.0,
                "is_decoupled_lr": False,
            }
        ],
        staged_config=GPUStagedAdamWConfig(
            buffer_count=2, bucket_size_mb=7 * 4 / (1024 * 1024)
        ),
        **kwargs,
    )
    ddp_config = SimpleNamespace(
        use_megatron_fsdp=False,
        overlap_param_gather=False,
        use_distributed_optimizer=True,
        num_distributed_optimizer_instances=1,
        reduce_scatter_with_fp32_accumulation=False,
    )
    bucket = SimpleNamespace(
        grad_data=torch.empty_like(model_param),
        param_data=model_param.detach(),
        offset=0,
        numel_unpadded=numel,
        params_list=[model_param],
    )
    buffer = SimpleNamespace(
        param_dtype=torch.bfloat16,
        grad_dtype=torch.bfloat16,
        buckets=[bucket],
        param_index_map={model_param: (0, numel, 0)},
        data_parallel_group=dist.group.WORLD,
        data_parallel_world_size=world_size,
        ddp_config=ddp_config,
        params=[model_param],
    )
    model_chunk = _ModelChunk(ddp_config, model_param)
    config = OptimizerConfig(
        optimizer="adam",
        lr=kwargs["lr"],
        min_lr=0.0,
        weight_decay=kwargs["weight_decay"],
        adam_beta1=kwargs["betas"][0],
        adam_beta2=kwargs["betas"][1],
        adam_eps=kwargs["eps"],
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        main_grads_dtype=torch.bfloat16,
        main_params_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        clip_grad=0.0,
    )
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
    assert bind_gpu_staged_adamw(optimizer) == 1
    owned_shard = optimizer.optimizer.param_groups[0]["params"][0]
    model_chunk.owned_shard = owned_shard
    baseline = torch.optim.AdamW([baseline_param], **kwargs)
    param_range = optimizer._get_model_param_range_map(model_param)["param"]
    shard_start = param_range.start
    assert param_range.end - param_range.start == local_numel
    max_errors = {"model": 0.0, "master": 0.0, "exp_avg": 0.0, "exp_avg_sq": 0.0}
    use_non_default_stream = (
        os.environ.get("ACCEPTANCE_NON_DEFAULT_STREAM", "").strip() == "1"
    )
    caller_stream = torch.cuda.Stream() if use_non_default_stream else None

    for step in range(3):
        local_accum = torch.zeros(numel, device=device, dtype=torch.float32)
        for microbatch in range(2):
            local_accum.add_(
                torch.arange(numel, device=device, dtype=torch.float32) / 100
                + rank
                + microbatch
                + step
            )
        reduced_shard = torch.empty(local_numel, device=device, dtype=torch.float32)
        dist.reduce_scatter_tensor(reduced_shard, local_accum, op=dist.ReduceOp.SUM)
        reduced_shard.div_(world_size)
        full_grad = local_accum.clone()
        dist.all_reduce(full_grad)
        full_grad.div_(world_size)
        torch.testing.assert_close(
            reduced_shard.bfloat16(),
            full_grad[shard_start : shard_start + local_numel].bfloat16(),
            rtol=0.0,
            atol=0.0,
        )

        model_param.main_grad = torch.zeros_like(model_param)
        baseline_param.grad = full_grad.bfloat16().float()
        baseline.step()
        if caller_stream is None:
            model_param.main_grad[shard_start : shard_start + local_numel].copy_(
                reduced_shard
            )
            success, grad_norm, num_zeros = optimizer.step()
        else:
            input_ready = torch.cuda.Event()
            input_ready.record()
            with torch.cuda.stream(caller_stream):
                caller_stream.wait_event(input_ready)
                torch.cuda._sleep(10_000_000)
                model_param.main_grad[shard_start : shard_start + local_numel].copy_(
                    reduced_shard
                )
                success, grad_norm, num_zeros = optimizer.step()
            caller_stream.synchronize()
        inner.drain()
        assert success and grad_norm == 0.0 and num_zeros == 0
        assert model_chunk.sync_count == step + 1

        expected_model = baseline_param.detach().bfloat16()
        model_error = (model_param - expected_model).abs().max().item()
        max_errors["model"] = max(max_errors["model"], model_error)
        torch.testing.assert_close(model_param, expected_model, rtol=0.0, atol=0.0)
        state = inner.state[owned_shard]
        baseline_state = baseline.state[baseline_param]
        expected_slice = slice(shard_start, shard_start + local_numel)
        expected = {
            "master": baseline_param.detach()[expected_slice].cpu(),
            "exp_avg": baseline_state["exp_avg"][expected_slice].cpu(),
            "exp_avg_sq": baseline_state["exp_avg_sq"][expected_slice].cpu(),
        }
        actual = {
            "master": state["master_param"],
            "exp_avg": state["exp_avg"],
            "exp_avg_sq": state["exp_avg_sq"],
        }
        for key in actual:
            error = (actual[key] - expected[key]).abs().max().item()
            max_errors[key] = max(max_errors[key], error)
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-6, atol=2e-6)

    gathered = [torch.empty_like(model_param) for _ in range(world_size)]
    dist.all_gather(gathered, model_param)
    for peer in gathered:
        torch.testing.assert_close(peer, model_param, rtol=0.0, atol=0.0)

    output_dir = Path(os.environ["ACCEPTANCE_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"rank_{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "max_errors": max_errors,
                "sync_count": model_chunk.sync_count,
                "accumulation": 2,
                "owned_numel": inner.cpu_slabs.master.numel(),
                "staging_state_numel": inner.gpu_staging_state_numel,
                "non_default_caller_stream": use_non_default_stream,
            },
            indent=2,
        )
        + "\n"
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
