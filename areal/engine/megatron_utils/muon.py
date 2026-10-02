# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from areal.api.cli_args import OptimizerConfig, TrainEngineConfig


def muon_config_kwargs(config: OptimizerConfig) -> dict[str, Any]:
    """Pass native Muon options without changing Adam/SGD configurations."""
    if config.type != "muon":
        return {}
    return {
        f.name: getattr(config, f.name)
        for f in fields(config)
        if f.name.startswith("muon_")
    }


def validate_muon_config(config: TrainEngineConfig) -> None:
    """Validate the replicated native Muon path before model allocation.

    DP uses ordinary DDP all-reduce. Tensor-parallel orthogonalization is owned
    by Megatron's TensorParallelMuon. Optimizer states are replicated across DP;
    this is intentionally not LayerWiseDistributedOptimizer.
    """
    if config.optimizer is None or config.optimizer.type != "muon":
        return
    mcore = config.megatron
    if config.dtype not in ("bfloat16", "float32"):
        raise ValueError(
            "Megatron Muon requires bfloat16 or float32; FP16 is unsupported"
        )
    if config.weight_update_mode == "awex":
        raise ValueError(
            "Replicated Muon does not support AWEX flat-parameter-buffer residency; use disk or xccl weight transfer"
        )
    if config.use_lora:
        raise ValueError("Megatron Muon does not support LoRA")
    if mcore.ddp.use_distributed_optimizer:
        raise ValueError(
            "Megatron Muon uses replicated optimizer states: set "
            "megatron.ddp.use_distributed_optimizer=false for DDP all-reduce. "
            "Layer-wise distributed Muon is not enabled by this integration."
        )
    if not mcore.wrap_with_ddp or mcore.use_custom_fsdp or mcore.use_torch_fsdp2:
        raise ValueError("Megatron Muon requires ordinary Megatron DDP wrapping")
    if mcore.ddp.overlap_param_gather or mcore.overlap_param_gather_with_optimizer_step:
        raise ValueError("Megatron Muon does not support parameter-gather overlap")
    if mcore.fp8_config is not None or mcore.use_precision_aware_optimizer:
        raise ValueError(
            "Megatron Muon does not support FP8 or precision-aware optimizer states"
        )
