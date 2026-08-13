# SPDX-License-Identifier: Apache-2.0

import dataclasses
import types
from collections.abc import Callable
from typing import Any

import torch

import areal.utils.mbridge_compat  # noqa: F401, I001  # isort: skip  # must precede mbridge import
from mbridge.core.bridge import Bridge
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDDPConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from megatron.core.transformer import TransformerConfig
from transformers import AutoConfig, PretrainedConfig

from areal.api.cli_args import MegatronEngineConfig, MindSpeedEngineConfig
from areal.engine.core.model import is_valid_vision_model
from areal.infra.platforms import is_npu_available
from areal.models.mcore.bailing_moe import (
    hf_to_mcore_config_bailing_moe,
    make_mcore_layer_specs_bailing_moe,
)
from areal.models.mcore.glm4 import (
    hf_to_mcore_config_glm4moe,
    make_mcore_layer_specs_glm4moe,
)
from areal.models.mcore.qwen3 import (
    hf_to_mcore_config_qwen3_dense,
    make_mcore_layer_specs_qwen3_dense,
)
from areal.models.mcore.vocab_parallel_head import (
    replace_output_layer_with_areal_lm_head,
)
from areal.utils import logging

logger = logging.getLogger("MCoreRegistry")


class ValueHead(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        *,
        config: TransformerConfig,
        bias: bool = False,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits, tensor_parallel_output_grad=False
            )
        return logits, None


def _replace_output_layer_with_value_head(
    model: GPTModel,
    tf_config: TransformerConfig,
) -> None:
    """Replace model's output_layer with ValueHead.

    This function can be used on any GPTModel instance, whether created
    via mbridge or directly. After replacement:
    - model.output_layer becomes a ValueHead instance
    - model.vocab_size is set to 1

    Args:
        model: The GPTModel instance to modify
        tf_config: Transformer configuration containing hidden_size and SP settings
    """
    if not hasattr(model, "output_layer"):
        raise ValueError(
            "Model does not have output_layer. Ensure post_process=True when creating GPTModel."
        )

    dtype = tf_config.params_dtype

    model.output_layer = ValueHead(
        input_size=tf_config.hidden_size,
        output_size=1,
        config=tf_config,
        bias=False,
    ).to(dtype=dtype)

    model.vocab_size = 1


def _is_lm_head_module_name(name: str) -> bool:
    return name in ("output_layer", "lm_head") or name.endswith(
        (".output_layer", ".lm_head")
    )


def _call_with_optional_group(fn, tensor: torch.Tensor, group: Any | None):
    try:
        return fn(tensor, group=group)
    except TypeError:
        return fn(tensor)


def _fp32_lm_head_forward_impl(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    sequence_parallel: bool,
    tp_group: Any | None = None,
    **_: Any,
) -> torch.Tensor:
    """Run the lm-head projection in fp32.

    A bf16 head keeps the ``[tokens, vocab]`` logits and their gradient in bf16,
    which is the dominant rounding error in the loss for large vocabularies.
    Casting the operands here keeps the matmul and the bias add in fp32 while
    leaving the rest of the model in its configured dtype.

    Mirrors ``ColumnParallelLinear`` from megatron-core, verified against 0.17.0.
    The collective helpers gained a ``group`` keyword in newer releases, so the
    calls fall back to the positional form when it is absent.

    Upgrade considerations: this replaces the fused ``_forward_impl`` with a
    plain matmul, so the dgrad all-reduce comes from
    ``copy_to_tensor_model_parallel_region`` instead of the fused kernel -
    equivalent, not fused. If megatron-core changes the forward short-circuit
    conditions or the collective signatures, this reimplementation drifts
    silently; ``_enable_fp32_lm_head_forward`` therefore steps aside when the
    installed megatron-core exposes a native ``ColumnParallelLinearFP32``
    (absent as of 0.17.0).
    """
    if sequence_parallel:
        try:
            total_input = tensor_parallel.gather_from_sequence_parallel_region(
                input,
                tensor_parallel_output_grad=True,
                group=tp_group,
            )
        except TypeError:
            total_input = tensor_parallel.gather_from_sequence_parallel_region(
                input,
                tensor_parallel_output_grad=True,
            )
    else:
        total_input = input

    output = torch.matmul(total_input.float(), weight.t().float())
    if bias is not None:
        output = output + bias.float()
    return output


def _fp32_lm_head_forward(
    self,
    input_: torch.Tensor,
    weight=None,
    runtime_gather_output: bool | None = None,
    **kwargs,
):
    if weight is None:
        weight = self.weight
    if weight is None:
        raise RuntimeError(
            "weight was not supplied to lm_head forward pass and "
            "skip_weight_param_allocation is True."
        )

    bias = self.bias if not getattr(self, "skip_bias_add", False) else None
    tp_group = getattr(self, "tp_group", None)

    if (
        getattr(self, "async_tensor_model_parallel_allreduce", False)
        or getattr(self, "sequence_parallel", False)
        or getattr(self, "explicit_expert_comm", False)
    ):
        input_parallel = input_
    else:
        input_parallel = _call_with_optional_group(
            tensor_parallel.copy_to_tensor_model_parallel_region, input_, tp_group
        )

    output_parallel = _fp32_lm_head_forward_impl(
        input=input_parallel,
        weight=weight,
        bias=bias,
        sequence_parallel=getattr(self, "sequence_parallel", False),
        tp_group=tp_group,
    )

    runtime_gather_output = kwargs.get("runtime_gather_output", runtime_gather_output)
    gather_output = (
        getattr(self, "gather_output", False)
        if runtime_gather_output is None
        else runtime_gather_output
    )
    if gather_output:
        output = _call_with_optional_group(
            tensor_parallel.gather_from_tensor_model_parallel_region,
            output_parallel,
            tp_group,
        )
    else:
        output = output_parallel

    output_bias = self.bias if getattr(self, "skip_bias_add", False) else None
    return output, output_bias


def _enable_fp32_lm_head_forward(
    models: list[torch.nn.Module],
    *,
    enabled: bool,
) -> int:
    if not enabled:
        return 0

    native_fp32_cls = getattr(tensor_parallel, "ColumnParallelLinearFP32", None)
    patched = []
    already_fp32 = []

    for model_idx, model in enumerate(models):
        module = model.module if isinstance(model, DDP) else model
        for name, submodule in module.named_modules():
            if not _is_lm_head_module_name(name):
                continue
            if (
                native_fp32_cls is not None and isinstance(submodule, native_fp32_cls)
            ) or type(submodule).__name__ == "ColumnParallelLinearFP32":
                already_fp32.append(f"model{model_idx}:{name}")
                continue
            if getattr(submodule, "_areal_fp32_lm_head_enabled", False):
                already_fp32.append(f"model{model_idx}:{name}")
                continue
            if hasattr(submodule, "_forward_impl"):
                setattr(
                    submodule,
                    "_areal_original_forward_impl",
                    submodule._forward_impl,
                )
                setattr(submodule, "_forward_impl", _fp32_lm_head_forward_impl)
                setattr(submodule, "_areal_fp32_lm_head_enabled", True)
                patched.append(f"model{model_idx}:{name}:{type(submodule).__name__}")
            elif all(hasattr(submodule, attr) for attr in ("weight", "forward")):
                setattr(submodule, "_areal_original_forward", submodule.forward)
                setattr(
                    submodule,
                    "forward",
                    types.MethodType(_fp32_lm_head_forward, submodule),
                )
                setattr(submodule, "_areal_fp32_lm_head_enabled", True)
                patched.append(f"model{model_idx}:{name}:{type(submodule).__name__}")

    if patched:
        logger.warning(
            "Enabled FP32 lm_head/output_layer forward for modules: %s",
            ", ".join(patched),
        )
    elif already_fp32:
        logger.info(
            "FP32 lm_head/output_layer is already enabled for modules: %s",
            ", ".join(already_fp32),
        )
    else:
        logger.warning(
            "enable_fp32_lm_head=True, but no output_layer/lm_head module was found "
            "on this model chunk. This is expected for non-post-process pipeline "
            "stages."
        )

    return len(patched)


def unwrap_to_gpt_model(model: torch.nn.Module) -> GPTModel:
    """Unwraps a model to the underlying GPTModel instance.

    Handles both plain GPTModel (possibly wrapped in DDP) and VLM models
    (e.g., Qwen2_5VLModel) where GPTModel lives at ``model.language_model``.
    """
    _model = model
    while not isinstance(_model, GPTModel) and hasattr(_model, "module"):
        _model = _model.module
    if isinstance(_model, GPTModel):
        return _model
    # VLM models wrap GPTModel as language_model (e.g., Qwen2_5VLModel)
    if hasattr(_model, "language_model") and isinstance(
        _model.language_model, GPTModel
    ):
        return _model.language_model
    raise TypeError(f"Model could not be unwrapped to GPTModel. Got {type(_model)}")


def _replace_actor_output_layers(
    models: list[GPTModel | DDP],
    *,
    enabled: bool,
    fp32_operands: bool = False,
) -> None:
    if not enabled:
        return
    for model in models:
        gpt_model = unwrap_to_gpt_model(model)
        replace_output_layer_with_areal_lm_head(
            gpt_model,
            fp32_output=True,
            fp32_operands=fp32_operands,
        )


def _configure_actor_output_layers(
    models: list[GPTModel | DDP],
    mcore_config: MegatronEngineConfig | None,
) -> None:
    if mcore_config is None:
        return
    if mcore_config.enable_chunked_logits:
        _replace_actor_output_layers(
            models,
            enabled=True,
            fp32_operands=mcore_config.enable_fp32_lm_head,
        )
    else:
        _enable_fp32_lm_head_forward(
            models,
            enabled=mcore_config.enable_fp32_lm_head,
        )


# Model registry for different architectures
def make_hf_and_mcore_config(
    hf_path: str,
    dtype: torch.dtype,
    bridge=None,
    bridge_type: str = "mbridge",
    moe_permute_fusion: bool | None = None,
    moe_token_dispatcher_type: str | None = None,
) -> tuple[PretrainedConfig, TransformerConfig]:
    if bridge is not None and bridge_type == "mbridge":
        hf_config = bridge.hf_config
        hf_config._name_or_path = hf_path
        return hf_config, bridge.config
    elif bridge is not None and bridge_type == "megatron-bridge":
        if is_npu_available and moe_token_dispatcher_type is not None:
            MindSpeedEngineConfig(
                moe_token_dispatcher_type=moe_token_dispatcher_type
            ).validate_moe_token_dispatcher(bridge_type)
        if not is_npu_available and moe_token_dispatcher_type is not None:
            if moe_token_dispatcher_type != "alltoall":
                logger.warning(
                    "Ignoring megatron.moe_token_dispatcher_type=%r for Megatron Bridge; using 'alltoall'.",
                    moe_token_dispatcher_type,
                )
            moe_token_dispatcher_type = "alltoall"
        hf_config = getattr(bridge.hf_pretrained, "config", bridge.hf_pretrained)
        if hasattr(hf_config, "_name_or_path"):
            hf_config._name_or_path = hf_path
        if moe_token_dispatcher_type is not None or (
            is_npu_available
            and (
                mpu.get_expert_model_parallel_world_size() > 1
                or moe_permute_fusion is not None
            )
        ):
            # MindSpeed validates overlap during config extraction. Set expert
            # dimensions before finalizing; pipeline layout is configured later.
            provider = bridge.to_megatron_provider(load_weights=False)
            provider.expert_model_parallel_size = (
                mpu.get_expert_model_parallel_world_size()
            )
            provider.expert_tensor_parallel_size = (
                mpu.get_expert_tensor_parallel_world_size()
            )
            if moe_token_dispatcher_type is not None:
                provider.moe_token_dispatcher_type = moe_token_dispatcher_type
            if moe_permute_fusion is not None:
                if moe_token_dispatcher_type is None:
                    provider.moe_token_dispatcher_type = "alltoall"
                provider.moe_permute_fusion = moe_permute_fusion
                provider.use_fused_moe_token_permute_and_unpermute = moe_permute_fusion
            provider.finalize()
            tf_config = TransformerConfig(
                **{
                    field.name: getattr(provider, field.name)
                    for field in dataclasses.fields(TransformerConfig)
                    if hasattr(provider, field.name)
                }
            )
            return hf_config, tf_config
        return hf_config, bridge.transformer_config
    else:
        hf_config: PretrainedConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=hf_path,
            trust_remote_code=True,
        )
        assert len(hf_config.architectures) == 1
        architecture = hf_config.architectures[0]
        if architecture == "Qwen3ForCausalLM":
            return hf_config, hf_to_mcore_config_qwen3_dense(hf_config, dtype)
        elif architecture == "Glm4MoeForCausalLM":
            return hf_config, hf_to_mcore_config_glm4moe(hf_config, dtype)
        elif architecture in (
            "BailingMoeV2_5ForCausalLM",
            "BailingMoeLinearForCausalLM",
            "BailingHybridForCausalLM",
        ):
            return hf_config, hf_to_mcore_config_bailing_moe(hf_config, dtype)
        else:
            raise ValueError(
                f"Architecture not registered for config conversion: {architecture}."
            )


# These hybrid specs are for lora related unfused layers
def make_hybrid_spec_qwen3_dense(base_spec):
    from copy import deepcopy

    from mindspeed.core.megatron_basic.megatron_basic import PTNorm

    spec = deepcopy(base_spec)
    for layer in spec.layer_specs:
        sm = layer.submodules

        # norms
        sm.input_layernorm = PTNorm
        sm.pre_mlp_layernorm = PTNorm

        # attention
        attn = sm.self_attention.submodules
        attn.linear_qkv = ColumnParallelLinear
        attn.linear_proj = RowParallelLinear

        # mlp
        mlp = sm.mlp.submodules
        mlp.linear_fc1 = ColumnParallelLinear
    return spec


# # Works for Qwen3-30BB
def make_hybrid_spec_qwen3_moe(base_spec):
    from copy import deepcopy

    from mindspeed.core.megatron_basic.megatron_basic import PTNorm

    spec = deepcopy(base_spec)
    for layer in spec.layer_specs:
        sm = layer.submodules

        # norms
        sm.input_layernorm = PTNorm

        # attention
        attn = sm.self_attention.submodules
        attn.linear_qkv = ColumnParallelLinear
        attn.linear_proj = RowParallelLinear

    return spec


def make_hybrid_spec_glm4moe(base_spec):
    """
    Unfuse layer norms for GLM-4 MoE to ensure PEFT compatibility.

    GLM-4 has additional layernorms compared to Qwen3:
    - input_layernorm (pre-attention)
    - post_self_attn_layernorm (after attention residual)
    - post_attention_layernorm / pre_mlp_layernorm (pre-MLP)
    - post_mlp_layernorm (after MLP residual)
    - q_layernorm, k_layernorm (QK normalization)
    """
    from copy import deepcopy

    from mindspeed.core.megatron_basic.megatron_basic import PTNorm

    spec = deepcopy(base_spec)
    for layer in spec.layer_specs:
        sm = layer.submodules

        # Replace all layer norms with PTNorm (no fused implementations)
        sm.input_layernorm = PTNorm

        # GLM-4 specific: post-attention layernorm
        if hasattr(sm, "post_self_attn_layernorm"):
            sm.post_self_attn_layernorm = PTNorm

        # Pre-MLP layernorm (can be named differently in GLM-4)
        if hasattr(sm, "pre_mlp_layernorm"):
            sm.pre_mlp_layernorm = PTNorm
        if hasattr(sm, "post_attention_layernorm"):
            sm.post_attention_layernorm = PTNorm

        # GLM-4 specific: post-MLP layernorm
        if hasattr(sm, "post_mlp_layernorm"):
            sm.post_mlp_layernorm = PTNorm

        # Attention: unfuse QKV and replace QK layernorms
        attn = sm.self_attention.submodules
        attn.linear_qkv = ColumnParallelLinear
        attn.linear_proj = RowParallelLinear

        # GLM-4 QK layernorms (if present)
        if hasattr(attn, "q_layernorm"):
            # attn.q_layernorm = PTNorm
            attn.q_layernorm = None
        if hasattr(attn, "k_layernorm"):
            # attn.k_layernorm = PTNorm
            attn.k_layernorm = None

        # MoE layers are kept as-is (handled by Megatron's MoE)
        # No need to modify expert layers

    return spec


def make_mcore_layer_specs(
    hf_config: PretrainedConfig, tf_config: TransformerConfig, use_lora: bool
):
    assert len(hf_config.architectures) == 1
    architecture = hf_config.architectures[0]
    if architecture == "Qwen3ForCausalLM":
        transformer_layer_spec = make_mcore_layer_specs_qwen3_dense(
            tf_config, use_te=True
        )
        if use_lora:
            transformer_layer_spec = make_hybrid_spec_qwen3_dense(
                transformer_layer_spec
            )
    elif architecture == "Qwen3MoeForCausalLM":
        transformer_layer_spec = make_mcore_layer_specs_qwen3_dense(
            tf_config, use_te=True
        )
        if use_lora:
            transformer_layer_spec = make_hybrid_spec_qwen3_moe(transformer_layer_spec)
    elif architecture == "Glm4MoeForCausalLM":
        transformer_layer_spec = make_mcore_layer_specs_glm4moe(tf_config, use_te=True)
        if use_lora:
            transformer_layer_spec = make_hybrid_spec_glm4moe(transformer_layer_spec)
    elif architecture in (
        "BailingMoeV2_5ForCausalLM",
        "BailingMoeLinearForCausalLM",
        "BailingHybridForCausalLM",
    ):
        transformer_layer_spec = make_mcore_layer_specs_bailing_moe(
            tf_config, hf_config, use_te=True
        )
    else:
        raise ValueError(
            f"Architecture not registered for config conversion: {architecture}."
        )
    return transformer_layer_spec


def make_mcore_model(
    hf_config: PretrainedConfig,
    tf_config: TransformerConfig,
    mcore_config: MegatronEngineConfig | None = None,
    bridge: Bridge | Any | None = None,
    bridge_type: str = "mbridge",
    is_critic: bool = False,
    use_lora: bool = False,
    pre_wrap_hook: Callable[[list[Any]], list[Any]] | None = None,
    mindspeed_config: MindSpeedEngineConfig | None = None,
) -> list[GPTModel | DDP]:
    if bridge is not None and bridge_type == "mbridge":
        models = bridge.get_model(
            # TODO: Add DDP options when supporting training
            wrap_with_ddp=mcore_config.wrap_with_ddp,
            ddp_config=dataclasses.asdict(mcore_config.ddp),
            use_torch_fsdp2=mcore_config.use_torch_fsdp2,
            use_custom_fsdp=mcore_config.use_custom_fsdp,
            fp16=tf_config.fp16,
            bf16=tf_config.bf16,
            use_precision_aware_optimizer=mcore_config.use_precision_aware_optimizer,
            overlap_param_gather_with_optimizer_step=mcore_config.overlap_param_gather_with_optimizer_step,
        )
        models = list(models)

        # Replace output_layer with ValueHead for critic models
        if is_critic:
            for model in models:
                _model = unwrap_to_gpt_model(model)
                _replace_output_layer_with_value_head(_model, tf_config)
        else:
            _configure_actor_output_layers(models, mcore_config)

        return models

    if bridge is not None and bridge_type == "megatron-bridge":
        dispatcher = (
            (mindspeed_config or MindSpeedEngineConfig()).validate_moe_token_dispatcher(
                bridge_type
            )
            if is_npu_available
            else "alltoall"
        )
        if (
            not is_npu_available
            and mcore_config.moe_token_dispatcher_type != "alltoall"
        ):
            logger.warning(
                "Ignoring megatron.moe_token_dispatcher_type=%r for Megatron Bridge; using 'alltoall'.",
                mcore_config.moe_token_dispatcher_type,
            )
        if is_npu_available and hf_config.model_type in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
        ):
            from areal.engine.megatron_utils.megatron_bridge_patches import (
                patch_qwen3_vl_vision_config,
            )

            patch_qwen3_vl_vision_config()
        provider = bridge.to_megatron_provider(load_weights=False)
        provider.vision_dp_when_cp = mcore_config.vision_dp_when_cp
        if use_lora and hf_config.model_type == "qwen3_5":
            from megatron.bridge.models.qwen_vl import qwen35_vl_provider

            # The Qwen3.5 provider stores its spec callable on the dataclass as
            # well as calling the module-level builder from the VLM path. The
            # runtime patch updates the latter; keep the stored callable in sync
            # for language-only and alternate provider construction paths.
            provider.transformer_layer_spec = qwen35_vl_provider.get_transformer_block_with_experimental_attention_variant_spec
        vpp_size = mcore_config.virtual_pipeline_parallel_size or 0

        provider.tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
        provider.pipeline_model_parallel_size = (
            mpu.get_pipeline_model_parallel_world_size()
        )
        provider.virtual_pipeline_model_parallel_size = (
            vpp_size if vpp_size > 1 else None
        )
        provider.context_parallel_size = mpu.get_context_parallel_world_size()
        provider.expert_model_parallel_size = mpu.get_expert_model_parallel_world_size()
        provider.expert_tensor_parallel_size = (
            mpu.get_expert_tensor_parallel_world_size()
        )
        provider.sequence_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        provider.pipeline_dtype = tf_config.params_dtype

        provider.recompute_granularity = mcore_config.recompute_granularity
        provider.recompute_method = mcore_config.recompute_method
        provider.recompute_num_layers = mcore_config.recompute_num_layers
        provider.distribute_saved_activations = (
            mcore_config.distribute_saved_activations
        )
        provider.recompute_modules = mcore_config.recompute_modules

        provider.freeze_vision_model = mcore_config.freeze_vision_model
        provider.freeze_vision_projection = mcore_config.freeze_vision_projection

        from megatron.bridge.models.gpt_provider import (
            default_layer_spec,
            local_layer_spec,
        )

        uses_qwen3_moe_lora_spec = (
            use_lora
            and is_npu_available
            and hf_config.model_type == "qwen3_moe"
            and hf_config.architectures == ["Qwen3MoeForCausalLM"]
            and bool(getattr(provider, "num_moe_experts", None))
            and provider.transformer_layer_spec is default_layer_spec
            and not getattr(provider, "fp8", None)
            and not getattr(provider, "use_transformer_engine_full_layer_spec", False)
            and not getattr(provider, "use_transformer_engine_op_fuser", False)
        )
        uses_local_layer_spec = (
            use_lora
            and is_npu_available
            and provider.transformer_layer_spec is default_layer_spec
            and not getattr(provider, "num_moe_experts", None)
            and not getattr(provider, "fp8", None)
            and not getattr(provider, "use_transformer_engine_full_layer_spec", False)
            and not is_valid_vision_model(hf_config.model_type)
        )
        if uses_qwen3_moe_lora_spec:
            from areal.engine.megatron_utils.megatron_bridge_patches import (
                patch_qwen3_moe_lora_spec,
            )

            patch_qwen3_moe_lora_spec(provider)
        elif uses_local_layer_spec:
            # Megatron-Bridge applies LoRA to fused TE QKV/FC1 layers by asking
            # the base layer for its normalized input via `return_layernorm_output`.
            # MindSpeed's TELayerNormColumnParallelLinear does not implement that
            # contract and only returns `(linear_output, bias)`. The generic local
            # GPT spec keeps normalization separate and uses MCore row/column
            # linears, so LoRA consumes the correct normalized activation.
            provider.transformer_layer_spec = local_layer_spec
            logger.info(
                "Using the local Megatron GPT layer spec for %s LoRA on NPU.",
                hf_config.architectures,
            )
        elif use_lora and is_npu_available:
            logger.warning(
                "Keeping transformer layer spec %r for NPU LoRA because the generic "
                "local GPT spec cannot safely replace MoE, FP8/full-TE, vision, or "
                "custom model blocks.",
                provider.transformer_layer_spec,
            )

        if (
            hasattr(tf_config, "pipeline_model_parallel_layout")
            and tf_config.pipeline_model_parallel_layout is not None
        ):
            provider.pipeline_model_parallel_layout = (
                tf_config.pipeline_model_parallel_layout
            )
        else:
            pipeline_split_attrs = (
                "num_layers_in_first_pipeline_stage",
                "num_layers_in_last_pipeline_stage",
                "account_for_embedding_in_pipeline_split",
                "account_for_loss_in_pipeline_split",
            )

            for attr in pipeline_split_attrs:
                if hasattr(tf_config, attr):
                    setattr(provider, attr, getattr(tf_config, attr))

        has_mtp = bool(getattr(provider, "mtp_num_layers", None))
        if mcore_config.enable_mtp:
            if not has_mtp:
                raise ValueError(
                    "megatron.enable_mtp=True but the model has no MTP layers."
                )
        elif has_mtp:
            logger.warning(
                "Dropping MTP head (mtp_num_layers=%s -> None); not used in RL and not "
                "exportable for Qwen3.6. Set megatron.enable_mtp=True to keep it.",
                provider.mtp_num_layers,
            )
            provider.mtp_num_layers = None

        # Disable fused weight-grad accumulation: LoRA params lack the main_grad
        # buffers it needs; on NPU the fused wgrad op is a no-op stub that
        # corrupts gradients (NaN).
        if use_lora or is_npu_available:
            provider.gradient_accumulation_fusion = False

        # Keep runtime-sensitive provider flags aligned with AReaL's config.
        provider.variable_seq_lengths = True
        provider.moe_token_dispatcher_type = dispatcher
        provider.batch_p2p_comm = False
        provider.overlap_p2p_comm = (
            vpp_size > 1 and provider.pipeline_model_parallel_size > 1
        )

        # NPU full-attention layers hit MindSpeed's DotProductAttention, whose
        # get_attention_mask() requires use_flash_attn (else it demands a
        # micro_batch_size). Force it on for bridge providers.
        if is_npu_available and hasattr(provider, "experimental_attention_variant"):
            provider.use_flash_attn = True

        # Aligning tf config settings with provider for consistency.
        tf_config.variable_seq_lengths = provider.variable_seq_lengths
        tf_config.moe_token_dispatcher_type = provider.moe_token_dispatcher_type

        provider.moe_permute_fusion = (
            (mindspeed_config or MindSpeedEngineConfig()).moe_permute_fusion
            if is_npu_available
            else mcore_config.moe_permute_fusion
        )
        tf_config.moe_permute_fusion = provider.moe_permute_fusion
        if is_npu_available:
            provider.use_fused_moe_token_permute_and_unpermute = (
                provider.moe_permute_fusion
            )
            tf_config.use_fused_moe_token_permute_and_unpermute = (
                provider.moe_permute_fusion
            )

        if is_npu_available and mindspeed_config is not None:
            provider.moe_aux_loss_coeff = mindspeed_config.moe_aux_loss_coeff
            provider.moe_router_load_balancing_type = (
                mindspeed_config.moe_router_load_balancing_type
            )
            tf_config.moe_aux_loss_coeff = provider.moe_aux_loss_coeff
            tf_config.moe_router_load_balancing_type = (
                provider.moe_router_load_balancing_type
            )

        tf_config.batch_p2p_comm = provider.batch_p2p_comm
        tf_config.overlap_p2p_comm = provider.overlap_p2p_comm

        provider.finalize()

        # GLM DSA features need the finalized provider that AReaL constructs
        # without Bridge's ConfigContainer. Limit reapplication to NPU GLM DSA.
        if is_npu_available and hf_config.model_type in ("glm_moe_dsa", "glm53_flash"):
            from mindspeed_bridge.features.adaptor import FeatureAdaptor, bind_config

            FeatureAdaptor.apply_finalized(bind_config(provider), reapply=True)

        ddp_config = MCoreDDPConfig(**dataclasses.asdict(mcore_config.ddp))
        if use_lora:
            ddp_config.use_distributed_optimizer = False
            ddp_config.overlap_grad_reduce = False
            ddp_config.overlap_param_gather = False

        models = provider.provide_distributed_model(
            ddp_config=ddp_config,
            fp16=tf_config.fp16,
            bf16=tf_config.bf16,
            use_megatron_fsdp=mcore_config.use_custom_fsdp,
            use_torch_fsdp2=mcore_config.use_torch_fsdp2,
            wrap_with_ddp=mcore_config.wrap_with_ddp,
            overlap_param_gather_with_optimizer_step=mcore_config.overlap_param_gather_with_optimizer_step,
            pre_wrap_hook=pre_wrap_hook,
        )
        models = list(models)

        if is_critic:
            for model in models:
                _model = unwrap_to_gpt_model(model)
                _replace_output_layer_with_value_head(_model, tf_config)
        else:
            _configure_actor_output_layers(models, mcore_config)

        return models

    else:
        if (
            mcore_config is not None
            and mcore_config.virtual_pipeline_parallel_size is not None
            and mcore_config.virtual_pipeline_parallel_size > 1
        ):
            raise NotImplementedError(
                "Virtual pipeline parallelism requires mbridge-backed models."
            )
        transformer_layer_spec = make_mcore_layer_specs(
            hf_config, tf_config, use_lora=use_lora
        )

        rope_scaling_args = {}
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling:
            rope_type = rope_scaling.get(
                "type", rope_scaling.get("rope_type", "default")
            )

            if rope_type == "linear":
                rope_scaling_args["seq_len_interpolation_factor"] = rope_scaling[
                    "factor"
                ]
            elif rope_type != "default":
                raise NotImplementedError(
                    f"Rope scaling type {rope_type} not supported yet."
                )

        rotary_base = (
            getattr(hf_config, "rope_theta", None)
            or (rope_scaling or {}).get("rope_theta")
            or 10000
        )

        pp_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_size > 1:
            pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False)
            post_process = mpu.is_pipeline_last_stage(ignore_virtual=False)
        else:
            pre_process = True
            post_process = True

        model = GPTModel(
            config=tf_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=hf_config.vocab_size,
            max_sequence_length=hf_config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=False,  # TODO: implement share output weights
            position_embedding_type="rope",
            rotary_base=rotary_base,
            **rope_scaling_args,
            # vp_stage=None TODO: virtual pipeline parallel
        )

        # Replace output_layer with ValueHead for critic models
        if is_critic:
            _replace_output_layer_with_value_head(model, tf_config)
        else:
            _configure_actor_output_layers([model], mcore_config)

        if mcore_config.wrap_with_ddp:
            ddp_config = MCoreDDPConfig(**dataclasses.asdict(mcore_config.ddp))
            wrapped = DDP(
                config=tf_config,
                ddp_config=ddp_config,
                module=model,
                disable_bucketing=False,
            )
            return [wrapped]
        return [model]
