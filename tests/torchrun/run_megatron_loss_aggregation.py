# SPDX-License-Identifier: Apache-2.0

"""Tiny-model loss audit using runtime MCore, DDP and CP collectives.

This isolates aggregation and auxiliary scaling; it does not exercise an actual
attention layer, expert dispatch or pipeline parallelism.
"""

import argparse
import json
import os
from datetime import timedelta
from types import SimpleNamespace

try:
    from areal.utils.torch_npu_compat import import_mindspeed_adaptor

    import_mindspeed_adaptor()
except ImportError:
    pass

import megatron.core
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.utils import get_model_config

from tests.test_megatron_loss_aggregation import (
    MEAN_MODES,
    aggregate_test_loss,
    independent_loss_oracle,
    load_engine_callbacks,
    make_loss_data,
    make_loss_microbatches,
)
from tests.test_megatron_loss_normalization import (
    _bind_per_token_loss_methods,
    _MicroBatches,
)

from areal.engine.core.train_engine import compute_total_loss_weight
from areal.engine.megatron_utils.packed_context_parallel import (
    reassemble_cp_packed_logprobs,
    split_packed_seqs_for_context_parallel,
)
from areal.infra.platforms import current_platform
from areal.trainer.ppo.loss_reduction import prepare_policy_gradient_batch
from areal.utils.data import pack_tensor_dict, split_training_batch_into_microbatches
from areal.utils.functional.loss_aggregation import PolicyGradientReduction


class _TinyModel(MegatronModule):
    def __init__(self, config):
        super().__init__(config=config)
        device = current_platform.device_type
        self.theta = torch.nn.Parameter(torch.linspace(-0.2, 0.6, 128, device=device))
        self.aux_theta = torch.nn.Parameter(torch.tensor(0.3, device=device))
        self.model_type = ModelType.encoder_or_decoder
        self.share_embeddings_and_output_weights = False

    def set_input_tensor(self, value):
        pass

    def forward(self, batch):
        cu_seqlens = batch["cu_seqlens"]
        device = self.theta.device
        indices = torch.tensor(batch["rows"], device=device)[:, None] * 16
        indices = (indices + torch.arange(16, device=device)).flatten()
        local_indices = split_packed_seqs_for_context_parallel(indices, cu_seqlens)
        local_mask = split_packed_seqs_for_context_parallel(
            batch["loss_mask"], cu_seqlens
        )
        # Match the per-token router protocol: inject token-summed auxiliary loss
        # before CP gather; its backward seed is independent of main-loss weights.
        aux = 0.25 * self.aux_theta.square() * local_mask.count_nonzero()
        local_output = MoEAuxLossAutoScaler.apply(self.theta[local_indices], aux)
        return reassemble_cp_packed_logprobs(local_output, cu_seqlens).unsqueeze(-1)


def prepare_optimizer_step(data, mode, optimizer_step, split):
    """Prepare complete local prompts, then cut them using the actual PPO scheduler."""
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    device = data["loss_mask"].device
    rows = list(range(dp_rank * (8 // dp_size), (dp_rank + 1) * (8 // dp_size)))
    local = {key: value[rows] for key, value in data.items()}
    local["attention_mask"] = torch.ones_like(local["loss_mask"])
    local["input_ids"] = torch.tensor(rows, device=device)[:, None] * 16 + torch.arange(
        16, device=device
    )
    group_sizes = (
        ([3, 1, 1, 3] if dp_size == 1 else [3, 1] if dp_rank == 0 else [1, 3])
        if mode == "traj-mean"
        else ([4, 4] if dp_size == 1 else [4])
    )
    prepared = prepare_policy_gradient_batch(local, mode=mode, group_sizes=group_sizes)
    optimizer_batches = split_training_batch_into_microbatches(
        local, 2, group=mpu.get_data_parallel_group()
    )
    steps = prepared.for_steps(
        optimizer_batches,
        dp_group=mpu.get_data_parallel_group(),
        device=device,
    )
    batch = optimizer_batches[optimizer_step]
    local_rows = (batch["input_ids"][:, 0] // 16).tolist()
    gathered = [None] * dp_size
    dist.all_gather_object(gathered, local_rows, group=mpu.get_data_parallel_group())
    global_rows = [row for shard in gathered for row in shard]
    fragments = (
        [[i] for i in range(len(local_rows))]
        if split
        else [list(range(len(local_rows)))]
    )
    batches = _MicroBatches()
    for indices in fragments:
        fragment = pack_tensor_dict(
            {key: value[indices] for key, value in batch.items()}
        )
        fragment["rows"] = [local_rows[i] for i in indices]
        batches.append(fragment)
    return batches, steps[optimizer_step], global_rows, len(steps)


def run_case(mode, per_token, split, scenario, check, prepared=False, optimizer_step=0):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        num_moe_experts=2,
        calculate_per_token_loss=per_token,
        gradient_accumulation_fusion=False,
        perform_initialization=False,
    )
    model = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            overlap_grad_reduce=False,
            average_in_collective=False,
            grad_reduce_in_fp32=True,
        ),
        module=_TinyModel(config),
    )
    config.finalize_model_grads_func = finalize_model_grads
    config.grad_scale_func = lambda loss: loss
    reduction = PolicyGradientReduction(mode)
    data = make_loss_data(mode, scenario, current_platform.device_type)
    if prepared:
        batches, loss_step, global_rows, n_steps = prepare_optimizer_step(
            data, mode, optimizer_step, split
        )
    else:
        batches = make_loss_microbatches(
            data,
            mpu.get_data_parallel_rank(),
            mpu.get_data_parallel_world_size(),
            split,
        )
    namespace = {
        "torch": torch,
        "dist": dist,
        "mpu": mpu,
        "get_model_config": get_model_config,
        "compute_total_loss_weight": compute_total_loss_weight,
    }
    train, compute, build = load_engine_callbacks(namespace)
    engine = SimpleNamespace(
        _ensure_ready=lambda: None,
        optimizer_zero_grad=model.zero_grad_buffer,
        optimizer_step=lambda: {},
        _normalize_batch_input=lambda data: (data, None),
        _prepare_mb_list=lambda data: data,
        model=[model],
        config=SimpleNamespace(is_critic=True, loss_aggregation=mode),
        enable_tree_training=False,
        device=current_platform.device_type,
    )
    engine._build_per_token_loss_output = lambda *args, **kwargs: build(
        engine, *args, **kwargs
    )
    _bind_per_token_loss_methods(engine, namespace)
    engine._compute_logprobs_and_loss = lambda *args, **kwargs: compute(
        engine, *args, **kwargs
    )

    def forward_backward(microbatches, process_output, forward_only):
        def forward_step(iterator, model):
            batch = next(iterator)
            output = model(batch)

            def loss_callback(output):
                result = process_output(output, batch)
                return (*result, {}) if isinstance(result, tuple) else (result, {})

            return output, loss_callback

        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter(microbatches),
            model=model,
            num_microbatches=len(microbatches),
            seq_length=16,
            micro_batch_size=1,
            forward_only=forward_only,
        )

    engine.forward_backward_batch = forward_backward
    train(
        engine,
        batches,
        lambda values, batch: aggregate_test_loss(
            values, batch, loss_step if prepared else reduction
        ),
        (loss_step if prepared else reduction).normalizer_fn,
    )
    if check == "main":
        theta = model.module.theta.detach().clone().reshape(8, 16).requires_grad_()
        oracle_data = data
        multiplier = 1
        if prepared:
            selected = torch.zeros(
                (8, 1), dtype=torch.bool, device=current_platform.device_type
            )
            selected[global_rows] = True
            oracle_data = {
                "loss_mask": data["loss_mask"]
                if mode in ("traj-mean", "prompt-mean")
                else data["loss_mask"] & selected,
                "numerator_mask": data["numerator_mask"] & selected,
            }
            multiplier = n_steps if mode in ("traj-mean", "prompt-mean") else 1
        loss = multiplier * independent_loss_oracle(
            (theta - data["targets"]).square(), oracle_data, mode
        )
        expected = torch.autograd.grad(loss, theta)[0].flatten()
        actual = model.module.theta.main_grad.detach()
    else:
        # Same token-summed auxiliary objective for every main aggregation mode.
        expected = 0.5 * model.module.aux_theta.detach()
        actual = model.module.aux_theta.main_grad.detach()
    passed = torch.allclose(actual, expected, rtol=3e-5, atol=3e-6)
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "mode": mode,
                    "per_token": per_token,
                    "split": split,
                    "scenario": scenario,
                    "check": check,
                    "prepared": prepared,
                    "optimizer_step": optimizer_step,
                    "max_error": (actual - expected).abs().max().item(),
                    "actual_norm": actual.norm().item(),
                    "expected_norm": expected.norm().item(),
                    "result": "PASS" if passed else "FAIL",
                }
            ),
            flush=True,
        )
    return passed


def check_unequal_rank_schedule(mode):
    """Actual DP collectives cap K and retain unit means on unequal row shards."""
    rank = mpu.get_data_parallel_rank()
    size = mpu.get_data_parallel_world_size()
    n_rows = 1 if rank == 0 else 3
    device = current_platform.device_type
    mask = torch.ones(n_rows, 16, device=device, dtype=torch.bool)
    data = {
        "input_ids": torch.ones_like(mask, dtype=torch.long),
        "attention_mask": mask,
        "loss_mask": mask,
    }
    prepared = prepare_policy_gradient_batch(
        data, mode=mode, group_sizes=[1] * n_rows if mode == "traj-mean" else [n_rows]
    )
    batches = split_training_batch_into_microbatches(
        data, 4, group=mpu.get_data_parallel_group()
    )
    assert len(batches) == 1
    step = prepared.for_steps(
        batches, dp_group=mpu.get_data_parallel_group(), device=device
    )[0]
    theta = torch.tensor(1.0, device=device, requires_grad=True)
    loss = step.aggregate(
        theta.square().expand_as(mask) * (1 if rank == 0 else 10),
        mask,
        unit_weights=batches[0]["unit_weights"],
    )
    contribution = loss * step.normalizer_fn(batches[0])
    totals = torch.stack(
        (contribution.detach(), torch.autograd.grad(contribution, theta)[0])
    )
    dist.all_reduce(totals, group=mpu.get_data_parallel_group())
    expected = 1.0 if size == 1 else (31 / 4 if mode == "traj-mean" else 11 / 2)
    torch.testing.assert_close(
        totals,
        torch.tensor([expected, 2 * expected], device=device),
        rtol=1e-6,
        atol=1e-6,
    )
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "schedule": "unequal-rows",
                    "mode": mode,
                    "steps": len(batches),
                    "result": "PASS",
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-size", type=int, required=True)
    parser.add_argument("--check", choices=("main", "auxiliary"), required=True)
    parser.add_argument("--prepared", action="store_true")
    args = parser.parse_args()
    current_platform.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        current_platform.communication_backend, timeout=timedelta(seconds=60)
    )
    try:
        mpu.initialize_model_parallel(context_parallel_size=args.cp_size)
        if dist.get_rank() == 0:
            print(
                "MCORE", megatron.core.__file__, megatron.core.__version__, flush=True
            )
        if args.prepared and args.check == "main":
            for mode in ("traj-mean", "prompt-mean"):
                check_unequal_rank_schedule(mode)
        failed = []
        for mode in (
            ("seq-mean", "traj-mean", "prompt-mean") if args.prepared else MEAN_MODES
        ):
            for per_token in (False, True) if args.check == "main" else (True,):
                for split in (False, True):
                    for scenario in (
                        ("ragged", "empty-unit", "filtered")
                        if args.prepared
                        else ("ragged", "filtered")
                    ):
                        for step in range(2) if args.prepared else (0,):
                            if not run_case(
                                mode,
                                per_token,
                                split,
                                scenario,
                                args.check,
                                args.prepared,
                                step,
                            ):
                                failed.append((mode, per_token, split, scenario, step))
        # Every rank finishes the same collectives before reporting any failures.
        dist.barrier()
        assert not failed, f"Loss aggregation gradient mismatches: {failed}"
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
