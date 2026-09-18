# SPDX-License-Identifier: Apache-2.0

"""AReaL adapter for ModelScope's standalone ``mcore_bridge`` package.

The package is intentionally optional: importing AReaL does not require
``mcore_bridge`` unless ``bridge_type='mcore-bridge'`` is selected.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from transformers import PretrainedConfig


def _validate_config_fields(config_cls: type, config_kwargs: dict[str, Any]) -> None:
    accepted = {field.name for field in dataclasses.fields(config_cls)}
    unsupported = set(config_kwargs) - accepted
    if unsupported:
        raise ValueError(
            "Installed mcore-bridge ModelConfig does not accept fields: "
            f"{sorted(unsupported)}. Use a compatible mcore-bridge revision."
        )


def _configure_qwen4_exp_parameters(
    model: torch.nn.Module, *, freeze_ple_table: bool = True
) -> tuple[str, ...]:
    """Configure trainable Qwen parameters before DDP/optimizer grouping."""
    for name, module in model.named_modules():
        if name == "visual" or name.endswith(".self_attention.indexer"):
            module.requires_grad_(False)
        elif name == "embedding.word_embeddings" or name.endswith(
            ".embedding.word_embeddings"
        ):
            module.requires_grad_(True)
        elif name.endswith(".ple.ple_embedding"):
            if getattr(module, "cpu_offload", False):
                raise NotImplementedError(
                    "Qwen4-Exp training requires gradients for the PLE ngram table; "
                    "mcore-bridge's PLE_CPU_OFFLOAD host table has no backward path."
                )
            table = getattr(module, "ngram_embedding", None)
            weight = getattr(table, "weight", None)
            if (
                not isinstance(table, torch.nn.Module)
                or not isinstance(weight, torch.nn.Parameter)
                or not any(parameter is weight for parameter in table.parameters())
            ):
                raise ValueError(
                    "Qwen4-Exp PLE ngram table must expose a registered trainable Parameter."
                )
            weight.requires_grad_(not freeze_ple_table)
            if not freeze_ple_table:
                weight.no_weight_decay = True
            elif hasattr(weight, "no_weight_decay"):
                delattr(weight, "no_weight_decay")
    return tuple(
        name for name, param in model.named_parameters() if not param.requires_grad
    )


def qwen4_exp_optimizer_overrides(config: Any) -> dict[Any, Any]:
    """Preserve MCore defaults and apply Adam without decay to PLE ngram tables."""
    from megatron.core.optimizer import get_standard_config_overrides
    from megatron.core.optimizer.optimizer_config import ParamKey
    from megatron.core.optimizer_param_scheduler import ParamGroupOverride

    overrides = get_standard_config_overrides(config)
    overrides[ParamKey(attr="no_weight_decay")] = ParamGroupOverride(wd_mult=0.0)
    return overrides


class MCoreBridgeAdapter:
    """Adapt ModelScope mcore-bridge's bridge/config pair to AReaL's engine API."""

    backend_name = "mcore-bridge"

    def __init__(
        self,
        model_path: str,
        *,
        dtype: torch.dtype,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        context_parallel_size: int,
        expert_model_parallel_size: int,
        expert_tensor_parallel_size: int,
        virtual_pipeline_model_parallel_size: int | None,
        gradient_checkpointing: bool,
        recompute_granularity: str | None,
        recompute_method: str | None,
        recompute_num_layers: int | None,
        distribute_saved_activations: bool | None,
        recompute_modules: list[str] | None,
        language_model_only: bool,
        transformer_config_overrides: dict[str, Any] | None = None,
        freeze_ple_table: bool = True,
    ) -> None:
        try:
            from mcore_bridge import ModelConfig, hf_to_mcore_config
        except ImportError as exc:
            raise ImportError(
                "bridge_type='mcore-bridge' requires the ModelScope mcore-bridge "
                "package to be installed in the runtime image."
            ) from exc

        from transformers import AutoConfig

        self.hf_config: PretrainedConfig = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        config_kwargs = hf_to_mcore_config(self.hf_config)
        config_kwargs.update(
            params_dtype=dtype,
            pipeline_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            sequence_parallel=tensor_model_parallel_size > 1,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size
            if virtual_pipeline_model_parallel_size
            and virtual_pipeline_model_parallel_size > 1
            else None,
            variable_seq_lengths=True,
            language_model_only=language_model_only,
            batch_p2p_comm=False,
            overlap_p2p_comm=bool(
                pipeline_model_parallel_size > 1
                and virtual_pipeline_model_parallel_size
                and virtual_pipeline_model_parallel_size > 1
            ),
        )
        if self.hf_config.model_type == "qwen4_exp" and context_parallel_size > 1:
            config_kwargs["cp_comm_type"] = "all_gather"
        if gradient_checkpointing:
            config_kwargs.update(
                recompute_granularity=recompute_granularity,
                recompute_method=recompute_method,
                recompute_num_layers=recompute_num_layers,
                distribute_saved_activations=distribute_saved_activations,
                recompute_modules=recompute_modules,
            )
        if self.hf_config.model_type == "qwen4_exp":
            config_kwargs["apply_rope_fusion"] = False
        if transformer_config_overrides:
            config_kwargs.update(transformer_config_overrides)
        if (
            self.hf_config.model_type == "qwen4_exp"
            and config_kwargs["apply_rope_fusion"]
        ):
            raise NotImplementedError(
                "Qwen4-Exp packed QSA requires apply_rope_fusion=False so the "
                "indexer receives per-token rotary positions."
            )
        _validate_config_fields(ModelConfig, config_kwargs)
        self.config = ModelConfig(**config_kwargs)
        self.bridge = self.config.bridge
        self.model_path = model_path
        self.freeze_ple_table = freeze_ple_table
        self.frozen_parameter_names: list[tuple[str, ...]] = []
        self.ple_checkpoint_metadata: dict[str, dict[str, Any]] = {}

    def get_model(
        self,
        *,
        wrap_with_ddp: bool,
        ddp_config: dict[str, Any],
        use_torch_fsdp2: bool = False,
        use_custom_fsdp: bool = False,
        overlap_param_gather_with_optimizer_step: bool = False,
    ) -> list[torch.nn.Module]:
        if use_torch_fsdp2 or use_custom_fsdp:
            raise NotImplementedError(
                "mcore-bridge adapter currently supports Megatron DDP only; "
                "FSDP wrapping is owned by another AReaL backend."
            )
        if overlap_param_gather_with_optimizer_step:
            raise NotImplementedError(
                "mcore-bridge adapter does not yet support "
                "overlap_param_gather_with_optimizer_step."
            )
        from mcore_bridge import get_mcore_model
        from megatron.core import tensor_parallel
        from megatron.core.distributed import DistributedDataParallel as DDP
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.transformer.module import Float16Module

        models = list(get_mcore_model(self.config))
        if self.config.hf_model_type == "qwen4_exp":
            self.frozen_parameter_names = [
                _configure_qwen4_exp_parameters(
                    model, freeze_ple_table=self.freeze_ple_table
                )
                for model in models
            ]
        for model in models:
            for parameter in model.parameters():
                tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                    parameter
                )
        if self.config.fp16 or self.config.bf16:
            models = [Float16Module(self.config, model) for model in models]
        if not wrap_with_ddp:
            return models
        wrapped = []
        for chunk_index, model in enumerate(models):
            wrapped.append(
                DDP(
                    config=self.config,
                    ddp_config=DistributedDataParallelConfig(**ddp_config),
                    module=model,
                    disable_bucketing=chunk_index > 0,
                )
            )
        return wrapped

    def load_weights(self, models: list[torch.nn.Module], path: str) -> None:
        if self.config.hf_model_type == "qwen4_exp":
            from mcore_bridge.utils.qwen4_exp_checkpoint import validate_ple_checkpoint

            self.ple_checkpoint_metadata = validate_ple_checkpoint(
                path, self.config, self.bridge.hf_layers_prefix, models
            )
        self.bridge.load_weights(models, path)

    def export_hf_weights(
        self,
        models: list[torch.nn.Module],
        *,
        cpu: bool = False,
        show_progress: bool = False,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in self.bridge.export_weights(
            models,
            target_device="cpu" if cpu else None,
            disable_tqdm=not show_progress,
        ):
            if tensor is not None:
                yield name, tensor

    def save_weights(
        self,
        models: list[torch.nn.Module],
        path: str,
        *,
        cpu_group: dist.ProcessGroup | None = None,
    ) -> None:
        """Save tensor shards; the engine publishes HF metadata after validation."""
        self.bridge.save_weights(models, path)
        error = None
        try:
            if self.config.hf_model_type == "qwen4_exp":
                from mcore_bridge.utils.qwen4_exp_checkpoint import (
                    validate_ple_checkpoint,
                )

                validate_ple_checkpoint(
                    path, self.config, self.bridge.hf_layers_prefix, models
                )
        except Exception as exc:
            if cpu_group is None:
                raise
            error = f"{type(exc).__name__}: {exc}"
        if cpu_group is not None:
            errors = [None] * dist.get_world_size(group=cpu_group)
            dist.all_gather_object(errors, error, group=cpu_group)
            failures = [
                f"rank {rank}: {message}"
                for rank, message in enumerate(errors)
                if message is not None
            ]
            if failures:
                raise RuntimeError(
                    "mcore-bridge checkpoint PLE validation failed: "
                    + "; ".join(failures)
                )
