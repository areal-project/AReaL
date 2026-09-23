"""Native Qwen MTP SFT smoke test with packed THD, TP, PP and CP.

Run with torchrun (world size must be divisible by 2 * TP * CP). The default
shrinks the supplied checkpoint configuration; --real-size loads its weights.
Use --full for joint backbone/MTP training and --export for HF round-trip checks.
"""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoConfig

from areal.api.cli_args import MegatronEngineConfig
from areal.engine.megatron_utils.megatron_bridge_patches import _apply_patches_on_import
from areal.engine.megatron_utils.packed_context_parallel import (
    packed_context_parallel_forward,
    split_packed_seqs_for_context_parallel,
)
from areal.models.mcore.registry import make_mcore_model
from areal.utils.logging import getLogger

logger = getLogger("MTPTrainingTest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=2)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--real-size", action="store_true")
    parser.add_argument("--export")
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    _apply_patches_on_import()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    mpu.initialize_model_parallel(
        pipeline_model_parallel_size=2,
        tensor_model_parallel_size=args.tp,
        context_parallel_size=args.cp,
    )
    model_parallel_cuda_manual_seed(17)
    hf_config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if not args.real_size:
        text = hf_config.text_config
        text.num_hidden_layers = 4
        text.layer_types = ["linear_attention", "full_attention"] * 2
        text.full_attention_interval = 2
        text.hidden_size = 256
        text.intermediate_size = 512
        text.num_attention_heads = 4
        text.num_key_value_heads = 4
        text.head_dim = 64
        text.vocab_size = 512
        text.linear_num_key_heads = 4
        text.linear_num_value_heads = 8
        text.linear_key_head_dim = 64
        text.linear_value_head_dim = 64
        vision = hf_config.vision_config
        vision.depth = 2
        vision.hidden_size = 128
        vision.intermediate_size = 256
        vision.num_heads = 4
        vision.out_hidden_size = 256
    bridge = (
        AutoBridge.from_hf_pretrained(args.checkpoint, trust_remote_code=True)
        if args.real_size
        else AutoBridge.from_hf_config(hf_config)
    )
    provider = bridge.to_megatron_provider(load_weights=False)
    for key, value in dict(
        pipeline_model_parallel_size=2,
        tensor_model_parallel_size=args.tp,
        context_parallel_size=args.cp,
        sequence_parallel=args.tp > 1,
        pipeline_dtype=torch.bfloat16,
        params_dtype=torch.bfloat16,
        bf16=True,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        deallocate_pipeline_outputs=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
        variable_seq_lengths=True,
        seq_length=96,
    ).items():
        setattr(provider, key, value)
    if not args.real_size:
        provider.mrope_section = [3, 3, 2]
    mcore_config = MegatronEngineConfig(
        bridge_type="megatron-bridge",
        enable_mtp=True,
        enable_mtp_training=True,
        mtp_only=not args.full,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    mcore_config.ddp.use_distributed_optimizer = True
    mcore_config.ddp.overlap_grad_reduce = False
    mcore_config.ddp.overlap_param_gather = False
    models = make_mcore_model(
        hf_config=hf_config,
        tf_config=provider,
        mcore_config=mcore_config,
        bridge=SimpleNamespace(to_megatron_provider=lambda **kwargs: provider),
        bridge_type="megatron-bridge",
    )
    if args.real_size:
        with torch.device("cpu"):
            bridge.load_hf_weights(models, hf_path=args.checkpoint)
    provider.finalize_model_grads_func = finalize_model_grads
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
        ),
        models,
    )
    before = {
        name: p.detach().clone()
        for model in models
        for name, p in model.named_parameters()
    }

    cu_seqlens = torch.tensor([0, 64, 96], device="cuda", dtype=torch.int32)

    def forward_step(data_iterator, model):
        ids = next(data_iterator).flatten()
        next_ids = torch.cat((ids[:64].roll(-1), ids[64:].roll(-1)))
        loss_mask = torch.ones_like(ids)
        loss_mask[[63, 95]] = 0
        labels = split_packed_seqs_for_context_parallel(next_ids, cu_seqlens)
        local_mask = split_packed_seqs_for_context_parallel(loss_mask, cu_seqlens)
        output = packed_context_parallel_forward(
            model,
            {
                "input_ids": ids,
                "cu_seqlens": cu_seqlens,
                "max_seqlen": 64,
                "mtp_kwargs": {"mtp_labels": ids, "mtp_loss_mask": loss_mask},
            },
            gather_cp_output=False,
            is_vision_model=True,
            use_model_packed_seq=True,
        )

        def loss_func(output):
            loss = (
                tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
                * local_mask
            ).sum()
            assert torch.isfinite(loss).all(), "Nonfinite main loss"
            tokens = local_mask.sum().to(dtype=torch.int)
            return loss, tokens, {"loss": loss.detach() / tokens}

        return output, loss_func

    for step in range(2):
        optimizer.zero_grad()
        for model in models:
            model.zero_grad_buffer()
        ids = torch.arange(96, device="cuda", dtype=torch.long).unsqueeze(0)
        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter([ids] * 4),
            model=models,
            num_microbatches=4,
            seq_length=96,
            micro_batch_size=1,
            forward_only=False,
        )
        finite_grad = torch.ones((), device="cuda", dtype=torch.int32)
        for model in models:
            for name, parameter in model.named_parameters():
                grad = getattr(parameter, "main_grad", None)
                if parameter.requires_grad and grad is not None:
                    finite_grad.mul_(torch.isfinite(grad).all().to(torch.int32))
        dist.all_reduce(finite_grad, op=dist.ReduceOp.MIN)
        assert finite_grad.item() == 1, f"Nonfinite gradient before step {step}"
        successful, norm, _ = optimizer.step()
        assert successful
        assert torch.isfinite(torch.as_tensor(norm)).all(), (
            f"Nonfinite norm at step {step}"
        )
        assert torch.as_tensor(norm).gt(0).all(), f"Zero gradient norm at step {step}"
        for model in models:
            for name, parameter in model.named_parameters():
                assert torch.isfinite(parameter).all(), (
                    f"Nonfinite parameter after step {step}: {name}"
                )
        logger.info("rank=%s step=%s norm=%s", dist.get_rank(), step, norm)
    updated = 0
    mtp_updated = 0
    backbone_updated = 0
    for model in models:
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), name
            if p.requires_grad:
                changed = int(not torch.equal(before[name], p))
                updated += changed
                if "mtp" in name.split("."):
                    mtp_updated += changed
                else:
                    backbone_updated += changed
            else:
                assert torch.equal(before[name], p), name
    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        assert mtp_updated > 0
    if args.full:
        assert backbone_updated > 0
    if args.full or mpu.is_pipeline_last_stage(ignore_virtual=True):
        assert updated > 0
    else:
        assert updated == 0
    logger.info("PASS rank=%s updated=%s", dist.get_rank(), updated)
    if args.export:
        if not args.real_size:
            raise ValueError("--export requires --real-size and a source checkpoint")
        bridge.save_hf_pretrained(
            models, args.export, source_path=args.checkpoint, strict=True
        )
        expected = {
            name: p.detach().cpu().clone()
            for model in models
            for name, p in model.named_parameters()
        }
        with torch.no_grad():
            for model in models:
                for p in model.parameters():
                    p.zero_()
        with torch.device("cpu"):
            bridge.load_hf_weights(models, hf_path=args.export)
        for model in models:
            for name, p in model.named_parameters():
                assert torch.equal(expected[name], p.detach().cpu()), name
        logger.info("ROUNDTRIP PASS rank=%s", dist.get_rank())
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
