# SPDX-License-Identifier: Apache-2.0
"""Run two deterministic Qwen4Exp MTP-only optimizer steps on one GPU."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from transformers import AutoConfig

from examples.mtp.qwen38_flash_next.model_build_smoke import _shrink_provider

from areal.api.cli_args import MegatronEngineConfig
from areal.engine.megatron_utils.packed_context_parallel import (
    packed_context_parallel_forward,
)
from areal.models.mcore.qwen4_exp_megatron_bridge import (
    register_qwen4_exp_megatron_bridge,
)
from areal.models.mcore.registry import make_mcore_model

SEED = 17
SEQUENCE_LENGTH = 32
NUM_MICROBATCHES = 2
NUM_STEPS = 2


def _is_mtp(name: str) -> bool:
    return "mtp" in name.split(".")[:-1]


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _digest(named_tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _parameters(models) -> dict[str, torch.nn.Parameter]:
    return {
        f"chunk_{chunk_index}.{name}": parameter
        for chunk_index, model in enumerate(models)
        for name, parameter in model.named_parameters()
    }


def main() -> None:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(SEED)
    torch.manual_seed(SEED)
    try:
        register_qwen4_exp_megatron_bridge()
        hf_config = AutoConfig.from_pretrained(
            os.environ["MODEL_PATH"], local_files_only=True
        )
        bridge = AutoBridge.from_hf_config(hf_config)
        provider = bridge.to_megatron_provider(load_weights=False)
        _shrink_provider(provider)
        # Exercise PLE freezing without allocating the checkpoint's ~100 GB
        # table. The architecture and hash path are unchanged, only the tiny
        # test table cardinality is reduced.
        provider.ple_layer_ids = [2]
        provider.ngram_size = 2
        provider.heads_per_ngram = 2
        provider.ngram_vocab_size_base = 17
        provider.make_ngram_vocab_size_divisible_by = 8
        provider.split_ngram_parts = 2
        provider.ple_conv_kernel_size = 2
        provider.eos_token_id = 1
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.deterministic_mode = True
        provider.cross_entropy_loss_fusion = False

        engine_config = MegatronEngineConfig(
            bridge_type="megatron-bridge",
            enable_mtp=True,
            enable_mtp_training=True,
            mtp_only=True,
            mtp_loss_scaling_factor=0.1,
            cross_entropy_loss_fusion=False,
        )
        engine_config.ddp.use_distributed_optimizer = True
        engine_config.ddp.overlap_grad_reduce = False
        engine_config.ddp.overlap_param_gather = False
        models = make_mcore_model(
            hf_config=hf_config,
            tf_config=provider,
            mcore_config=engine_config,
            bridge=SimpleNamespace(to_megatron_provider=lambda **kwargs: provider),
            bridge_type="megatron-bridge",
        )
        provider.finalize_model_grads_func = finalize_model_grads
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=2e-4,
                min_lr=0.0,
                weight_decay=0.0,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                clip_grad=1.0,
            ),
            models,
        )

        parameters = _parameters(models)
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in parameters.items()
        }
        before_mtp = {name: tensor for name, tensor in before.items() if _is_mtp(name)}
        before_frozen = {
            name: tensor for name, tensor in before.items() if not _is_mtp(name)
        }
        trainable = {
            name for name, parameter in parameters.items() if parameter.requires_grad
        }
        expected_trainable = {name for name in parameters if _is_mtp(name)}
        assert trainable == expected_trainable
        assert any("embedding.word_embeddings" in name for name in before_frozen)
        assert any("output_layer" in name for name in before_frozen)
        assert any(".ple." in name for name in before_frozen)

        input_ids = torch.arange(
            SEQUENCE_LENGTH, device="cuda", dtype=torch.long
        ).remainder(provider.padded_vocab_size)
        position_ids = torch.arange(SEQUENCE_LENGTH, device="cuda", dtype=torch.long)
        loss_mask = torch.ones_like(input_ids)
        loss_mask[-1] = 0
        labels = input_ids.roll(-1)
        cu_seqlens = torch.tensor(
            [0, SEQUENCE_LENGTH], device="cuda", dtype=torch.int32
        )

        def forward_step(data_iterator, model):
            ids = next(data_iterator)
            output = packed_context_parallel_forward(
                model,
                {
                    "input_ids": ids,
                    "position_ids": position_ids,
                    "cu_seqlens": cu_seqlens,
                    "max_seqlen": SEQUENCE_LENGTH,
                    "mtp_loss_mask": loss_mask,
                },
                gather_cp_output=False,
                is_vision_model=True,
            )

            def loss_func(model_output):
                token_loss = tensor_parallel.vocab_parallel_cross_entropy(
                    model_output.float(), labels
                )
                loss = (token_loss * loss_mask).sum()
                tokens = loss_mask.sum().to(torch.int)
                assert torch.isfinite(loss)
                return loss, tokens, {"loss": loss.detach() / tokens}

            return output, loss_func

        step_records = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(NUM_STEPS):
            optimizer.zero_grad()
            for model in models:
                model.zero_grad_buffer()
            started = time.perf_counter()
            losses = get_forward_backward_func()(
                forward_step_func=forward_step,
                data_iterator=iter([input_ids] * NUM_MICROBATCHES),
                model=models,
                num_microbatches=NUM_MICROBATCHES,
                seq_length=SEQUENCE_LENGTH,
                micro_batch_size=1,
                forward_only=False,
            )
            elapsed = time.perf_counter() - started
            assert "loss_values" in MTPLossLoggingHelper.tracker
            mtp_loss = (
                MTPLossLoggingHelper.tracker["loss_values"].sum().item()
                / NUM_MICROBATCHES
            )
            MTPLossLoggingHelper.clean_metrics_in_tracker()
            assert math.isfinite(mtp_loss) and mtp_loss > 0

            grad_tensors = []
            missing_grads = []
            for name, parameter in parameters.items():
                grad = getattr(parameter, "main_grad", None)
                if grad is None:
                    grad = parameter.grad
                if parameter.requires_grad:
                    if grad is None:
                        missing_grads.append(name)
                    else:
                        assert torch.isfinite(grad).all(), name
                        grad_tensors.append(grad.float().norm().square())
                else:
                    assert grad is None, f"frozen parameter received gradient: {name}"
            assert not missing_grads, missing_grads
            grad_norm_before_step = torch.stack(grad_tensors).sum().sqrt()
            assert torch.isfinite(grad_norm_before_step) and grad_norm_before_step > 0
            successful, optimizer_grad_norm, _ = optimizer.step()
            assert successful
            assert math.isfinite(float(optimizer_grad_norm))
            step_record = {
                "step": step,
                "main_loss": float(losses[0]["loss"]),
                "mtp_loss": mtp_loss,
                "grad_norm_before_step": float(grad_norm_before_step),
                "optimizer_grad_norm": float(optimizer_grad_norm),
                "elapsed_seconds": elapsed,
                "tokens_per_second": (SEQUENCE_LENGTH * NUM_MICROBATCHES / elapsed),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            step_records.append(step_record)
            print("step_record", json.dumps(step_record, sort_keys=True))

        changed_mtp = []
        changed_frozen = []
        max_frozen_abs_diff = 0.0
        for name, parameter in parameters.items():
            current = parameter.detach().cpu()
            if not torch.equal(before[name], current):
                if _is_mtp(name):
                    changed_mtp.append(name)
                else:
                    changed_frozen.append(name)
            if not _is_mtp(name):
                max_frozen_abs_diff = max(
                    max_frozen_abs_diff,
                    float((before[name].float() - current.float()).abs().max()),
                )
        assert changed_mtp
        assert not changed_frozen
        assert max_frozen_abs_diff == 0.0
        after_mtp = {
            name: parameter.detach().cpu()
            for name, parameter in parameters.items()
            if _is_mtp(name)
        }
        after_frozen = {
            name: parameter.detach().cpu()
            for name, parameter in parameters.items()
            if not _is_mtp(name)
        }
        result = {
            "seed": SEED,
            "sequence_length": SEQUENCE_LENGTH,
            "num_microbatches": NUM_MICROBATCHES,
            "num_optimizer_steps": NUM_STEPS,
            "switches": {
                "bridge_type": engine_config.bridge_type,
                "enable_mtp": engine_config.enable_mtp,
                "enable_mtp_training": engine_config.enable_mtp_training,
                "mtp_only": engine_config.mtp_only,
            },
            "mtp_parameter_count": len(before_mtp),
            "frozen_parameter_count": len(before_frozen),
            "changed_mtp_parameter_count": len(changed_mtp),
            "changed_frozen_parameter_count": len(changed_frozen),
            "max_frozen_abs_diff": max_frozen_abs_diff,
            "mtp_hash_before": _digest(before_mtp),
            "mtp_hash_after": _digest(after_mtp),
            "frozen_hash_before": _digest(before_frozen),
            "frozen_hash_after": _digest(after_frozen),
            "steps": step_records,
        }
        assert result["mtp_hash_before"] != result["mtp_hash_after"]
        assert result["frozen_hash_before"] == result["frozen_hash_after"]
        print(json.dumps(result, indent=2, sort_keys=True))
        print("mtp_only_two_step PASS")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
