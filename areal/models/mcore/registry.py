# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import inspect
import types
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from mbridge.core.bridge import Bridge
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDDPConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank
from transformers import AutoConfig, PretrainedConfig

from areal.api.cli_args import MegatronEngineConfig
from areal.engine.megatron_utils.deterministic import set_deterministic_algorithms
from areal.engine.megatron_utils.mtp_only import freeze_non_mtp_parameters
from areal.engine.megatron_utils.qwen3_vl_token_mask import install_qwen3_vl_token_mask
from areal.models.mcore.bailing_moe import (
    hf_to_mcore_config_bailing_moe,
    make_mcore_layer_specs_bailing_moe,
)
from areal.models.mcore.bailing_v3 import (
    hf_to_mcore_config_bailing_v3,
    make_mcore_layer_specs_bailing_v3,
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

        # Zero init (as in OpenAI's lm-human-preferences): V starts at 0 so early
        # advantages are not dominated by noise from an untrained head, and the
        # replicated head starts identical across tensor-parallel ranks.
        torch.nn.init.zeros_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

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

    Mirrors ``ColumnParallelLinear`` from megatron-core, verified against 0.19.0.
    The collective helpers gained a ``group`` keyword in newer releases, so the
    calls fall back to the positional form when it is absent.

    Upgrade considerations: this replaces the fused ``_forward_impl`` with a
    plain matmul, so the dgrad all-reduce comes from
    ``copy_to_tensor_model_parallel_region`` instead of the fused kernel -
    equivalent, not fused. If megatron-core changes the forward short-circuit
    conditions or the collective signatures, this reimplementation drifts
    silently; ``_enable_fp32_lm_head_forward`` therefore steps aside when the
    installed megatron-core exposes a native ``ColumnParallelLinearFP32``
    (absent as of 0.19.0).
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


def _unwrap_wrapper_modules(model: torch.nn.Module) -> torch.nn.Module:
    raw_model = model
    while not isinstance(raw_model, GPTModel) and hasattr(raw_model, "module"):
        raw_model = raw_model.module
    return raw_model


def _untie_critic_embeddings(model: torch.nn.Module, gpt_model: GPTModel) -> None:
    gpt_model.share_embeddings_and_output_weights = False
    raw_model = _unwrap_wrapper_modules(model)
    if raw_model is not gpt_model and hasattr(
        raw_model, "share_embeddings_and_output_weights"
    ):
        if not isinstance(
            getattr(type(raw_model), "share_embeddings_and_output_weights", None),
            property,
        ):
            raw_model.share_embeddings_and_output_weights = False
    embedding = getattr(gpt_model, "embedding", None)
    word_embeddings = getattr(embedding, "word_embeddings", None)
    emb_weight = getattr(word_embeddings, "weight", None)
    if emb_weight is not None:
        for attr in ("shared_embedding", "shared", "zero_out_wgrad"):
            if getattr(emb_weight, attr, False):
                setattr(emb_weight, attr, False)


@contextlib.contextmanager
def _untie_hf_config_for_critic(hf_config: Any):
    targets = []
    for cfg in (hf_config, getattr(hf_config, "text_config", None)):
        if cfg is not None and hasattr(cfg, "tie_word_embeddings"):
            targets.append((cfg, cfg.tie_word_embeddings))
            cfg.tie_word_embeddings = False
    try:
        yield
    finally:
        for cfg, prev in targets:
            cfg.tie_word_embeddings = prev


def _install_value_head_on_model(
    model: torch.nn.Module,
    tf_config: TransformerConfig,
) -> None:
    gpt_model = unwrap_to_gpt_model(model)
    _untie_critic_embeddings(model, gpt_model)
    if getattr(gpt_model, "post_process", True):
        _replace_output_layer_with_value_head(gpt_model, tf_config)


def _value_head_creation_callback(
    tf_config: TransformerConfig,
) -> Callable[..., None]:
    """mbridge post_model_creation_callbacks entry that installs ValueHead pre-DDP."""

    def install_value_head(model: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        _install_value_head_on_model(model, tf_config)

    return install_value_head


def _value_head_pre_wrap_hook(
    tf_config: TransformerConfig,
) -> Callable[[list[torch.nn.Module]], list[torch.nn.Module]]:
    """megatron-bridge pre-wrap hook that installs ValueHead pre-DDP."""

    def install_value_heads(
        models: list[torch.nn.Module],
    ) -> list[torch.nn.Module]:
        for model in models:
            _install_value_head_on_model(model, tf_config)
        return models

    return install_value_heads


def _verify_critic_value_heads(models: list[GPTModel | DDP]) -> None:
    for model in models:
        raw_model = _unwrap_wrapper_modules(model)
        gpt_model = unwrap_to_gpt_model(model)
        if getattr(gpt_model, "share_embeddings_and_output_weights", False) or getattr(
            raw_model, "share_embeddings_and_output_weights", False
        ):
            raise RuntimeError(
                "Critic model still ties its output layer to the word embeddings."
            )
        if not getattr(gpt_model, "post_process", True):
            continue
        output_layer = getattr(gpt_model, "output_layer", None)
        if not isinstance(output_layer, ValueHead):
            raise RuntimeError(
                "Critic post-process stage was built without a ValueHead."
            )
        if isinstance(model, DDP) and not hasattr(output_layer.weight, "main_grad"):
            raise RuntimeError(
                "Critic ValueHead is not registered in Megatron DDP gradient buffers; "
                "it must be installed before DDP wrapping."
            )


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
) -> tuple[PretrainedConfig, TransformerConfig]:
    if bridge is not None and bridge_type == "mbridge":
        hf_config = bridge.hf_config
        hf_config._name_or_path = hf_path
        return hf_config, bridge.config
    elif bridge is not None and bridge_type == "megatron-bridge":
        hf_config = getattr(bridge.hf_pretrained, "config", bridge.hf_pretrained)
        if hasattr(hf_config, "_name_or_path"):
            hf_config._name_or_path = hf_path
        return hf_config, bridge.transformer_config
    elif bridge is not None and bridge_type == "mcore-bridge":
        return bridge.hf_config, bridge.config
    else:
        hf_config: PretrainedConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=hf_path,
            trust_remote_code=True,
        )
        assert len(hf_config.architectures) == 1
        architecture = hf_config.architectures[0]
        if architecture == "Qwen3ForCausalLM":
            return hf_config, hf_to_mcore_config_qwen3_dense(hf_config, dtype)
        elif architecture in (
            "BailingMoeV2_5ForCausalLM",
            "BailingMoeLinearForCausalLM",
            "BailingHybridForCausalLM",
        ):
            return hf_config, hf_to_mcore_config_bailing_moe(hf_config, dtype)
        elif architecture == "BailingMoeV3ForCausalLM":
            return hf_config, hf_to_mcore_config_bailing_v3(hf_config, dtype)
        else:
            raise ValueError(
                f"Architecture not registered for config conversion: {architecture}."
            )


def make_mcore_layer_specs(hf_config: PretrainedConfig, tf_config: TransformerConfig):
    assert len(hf_config.architectures) == 1
    architecture = hf_config.architectures[0]
    if architecture == "Qwen3ForCausalLM":
        return make_mcore_layer_specs_qwen3_dense(tf_config, use_te=True)
    elif architecture in (
        "BailingMoeV2_5ForCausalLM",
        "BailingMoeLinearForCausalLM",
        "BailingHybridForCausalLM",
    ):
        return make_mcore_layer_specs_bailing_moe(tf_config, hf_config, use_te=True)
    elif architecture == "BailingMoeV3ForCausalLM":
        return make_mcore_layer_specs_bailing_v3(tf_config, hf_config, use_te=True)
    else:
        raise ValueError(
            f"Architecture not registered for config conversion: {architecture}."
        )


def make_mcore_model(
    hf_config: PretrainedConfig,
    tf_config: TransformerConfig,
    mcore_config: MegatronEngineConfig | None = None,
    bridge: Bridge | Any | None = None,
    bridge_type: str = "mbridge",
    is_critic: bool = False,
    use_lora: bool = False,
) -> list[GPTModel | DDP]:
    if mcore_config is not None and mcore_config.mtp_only:
        if bridge_type != "megatron-bridge" or bridge is None:
            raise ValueError("mtp_only requires a megatron-bridge model")
        if use_lora or is_critic:
            raise ValueError("mtp_only does not support LoRA or critic models")

    if is_critic and mcore_config is not None and mcore_config.enable_mtp:
        raise ValueError(
            "Megatron critic models do not support MTP; set megatron.enable_mtp=False."
        )

    if bridge is not None and bridge_type == "mbridge":
        extra_kwargs: dict[str, Any] = {}
        if is_critic:
            if (
                "post_model_creation_callbacks"
                not in inspect.signature(bridge.get_model).parameters
            ):
                raise NotImplementedError(
                    "Megatron critic requires an mbridge version whose get_model() "
                    "accepts post_model_creation_callbacks."
                )
            for cfg in (tf_config, getattr(bridge, "config", None)):
                if cfg is not None and getattr(cfg, "mtp_num_layers", None):
                    logger.warning(
                        "Dropping MTP head (mtp_num_layers=%s -> None) for critic model.",
                        cfg.mtp_num_layers,
                    )
                    cfg.mtp_num_layers = None
            extra_kwargs["post_model_creation_callbacks"] = [
                _value_head_creation_callback(tf_config)
            ]

        untie_ctx = (
            _untie_hf_config_for_critic(getattr(bridge, "hf_config", hf_config))
            if is_critic
            else contextlib.nullcontext()
        )
        with untie_ctx:
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
                **extra_kwargs,
            )
        models = list(models)

        if is_critic:
            _verify_critic_value_heads(models)
        else:
            _configure_actor_output_layers(models, mcore_config)

        return models

    if bridge is not None and bridge_type == "megatron-bridge":
        provider = bridge.to_megatron_provider(load_weights=False)
        # Native VLM providers apply these policies in provide(), before DDP
        # allocates gradient buffers and the optimizer selects parameters.
        is_qwen_vl = getattr(hf_config, "model_type", None) in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
        )
        for option in ("freeze_vision_model", "freeze_vision_projection"):
            freeze = getattr(mcore_config, option)
            if freeze is None and is_qwen_vl:
                freeze = True
            if freeze is not None:
                if not hasattr(provider, option):
                    raise ValueError(
                        f"{type(provider).__name__} does not support {option}; "
                        "use a compatible vision-language model provider"
                    )
                setattr(provider, option, freeze)
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

        # Bridge defaults to fused cross entropy, whereas AReaL defaults to the
        # unfused implementation. Honor the explicit engine setting for native
        # internal objectives such as MTP as well as the main model loss.
        provider.cross_entropy_loss_fusion = mcore_config.cross_entropy_loss_fusion

        provider.recompute_granularity = mcore_config.recompute_granularity
        provider.recompute_method = mcore_config.recompute_method
        provider.recompute_num_layers = mcore_config.recompute_num_layers
        provider.distribute_saved_activations = (
            mcore_config.distribute_saved_activations
        )
        provider.recompute_modules = mcore_config.recompute_modules

        provider.account_for_embedding_in_pipeline_split = False
        provider.account_for_loss_in_pipeline_split = False

        has_mtp = bool(getattr(provider, "mtp_num_layers", None))
        if mcore_config.enable_mtp:
            if not has_mtp:
                raise ValueError(
                    "megatron.enable_mtp=True but the model has no MTP layers."
                )
            if mcore_config.enable_mtp_training:
                if provider.mtp_num_layers != 1:
                    raise ValueError(
                        "MTP training currently supports exactly one prediction "
                        f"layer, got mtp_num_layers={provider.mtp_num_layers}."
                    )
                # Weight of the auxiliary MTP loss; consumed by Megatron-Core's
                # process_mtp_loss via config.mtp_loss_scaling_factor.
                provider.mtp_loss_scaling_factor = mcore_config.mtp_loss_scaling_factor
                # MCore 0.19 natively isolates the auxiliary MTP head from the
                # backbone, embedding, and tied or untied LM head.
                provider.mtp_detach_heads = True
        elif has_mtp:
            logger.warning(
                "Dropping MTP head (mtp_num_layers=%s -> None); not used in RL and not "
                "exportable for Qwen3.6. Set megatron.enable_mtp=True to keep it.",
                provider.mtp_num_layers,
            )
            provider.mtp_num_layers = None

        # LoRA params are injected after model materialization and do not carry
        # Megatron main_grad buffers required by fused grad accumulation kernels.
        if use_lora:
            provider.gradient_accumulation_fusion = False
        elif provider.gradient_accumulation_fusion:
            # Bridge also enables fusion when only TE is available, but the
            # native ColumnParallelLinear output head requires the Apex kernel.
            try:
                import fused_weight_gradient_mlp_cuda  # noqa: F401
            except ImportError:
                provider.gradient_accumulation_fusion = False
                logger.warning(
                    "Disabling gradient accumulation fusion: the native output "
                    "head requires fused_weight_gradient_mlp_cuda."
                )

        # Keep these four flags aligned with mbridge base defaults.
        provider.variable_seq_lengths = True
        logger.warning(
            "Ignoring mcore_config.moe_token_dispatcher_type=%s for bridge_type='megatron-bridge'; "
            "using 'alltoall' and variable_seq_lengths=True.",
            mcore_config.moe_token_dispatcher_type,
        )
        provider.moe_token_dispatcher_type = "alltoall"
        provider.batch_p2p_comm = False
        provider.overlap_p2p_comm = (
            vpp_size > 1 and provider.pipeline_model_parallel_size > 1
        )
        if mcore_config.enable_router_replay:
            if bool(getattr(provider, "moe_router_fusion", False)):
                raise ValueError(
                    "megatron.enable_router_replay=True is incompatible with "
                    "effective Megatron moe_router_fusion=True. Disable "
                    "moe_router_fusion to use Megatron-Core native RouterReplay."
                )
            provider.moe_enable_routing_replay = True

        # Aligning tf config settings with provider for consistency.
        tf_config.variable_seq_lengths = provider.variable_seq_lengths
        tf_config.moe_token_dispatcher_type = provider.moe_token_dispatcher_type
        tf_config.batch_p2p_comm = provider.batch_p2p_comm
        tf_config.overlap_p2p_comm = provider.overlap_p2p_comm
        if mcore_config.enable_router_replay:
            tf_config.moe_enable_routing_replay = True

        # Megatron-Bridge creates a new provider instead of building from
        # ``tf_config`` directly. Apply the prebuild deterministic settings to
        # the actual provider so construction-time consumers (TP layers and TE
        # attention) see the same configuration.
        if mcore_config.use_deterministic_algorithms:
            set_deterministic_algorithms(provider, prebuild=True)

        if is_critic and hasattr(provider, "share_embeddings_and_output_weights"):
            provider.share_embeddings_and_output_weights = False

        provider.finalize()

        if getattr(hf_config, "model_type", None) in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
        ):
            provider.register_pre_wrap_hook(install_qwen3_vl_token_mask)

        if mcore_config.mtp_only:
            if getattr(provider, "moe_router_enable_expert_bias", False):
                raise ValueError(
                    "mtp_only does not support MoE router expert-bias updates"
                )
            # Preserve existing Bridge hooks and freeze before DDP allocates
            # buffers. Do not disable autograd: the main loss still triggers
            # Megatron-Core's auxiliary MTP loss through MTPLossAutoScaler.
            # Stages without MTP are fully frozen, including all their virtual
            # chunks. Keep the missing-MTP check on ranks expected to own MTP.
            if "config" in inspect.signature(mtp_on_this_rank).parameters:
                owns_mtp = mtp_on_this_rank(provider, ignore_virtual=True)
            else:
                owns_mtp = mtp_on_this_rank(
                    layout=getattr(provider, "pipeline_model_parallel_layout", None),
                    mtp_num_layers=provider.mtp_num_layers,
                    ignore_virtual=True,
                )
            provider.register_pre_wrap_hook(
                partial(
                    freeze_non_mtp_parameters,
                    allow_missing_mtp=not owns_mtp,
                )
            )

        if is_critic:
            provider.register_pre_wrap_hook(_value_head_pre_wrap_hook(tf_config))

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
        )
        models = list(models)

        if is_critic:
            _verify_critic_value_heads(models)
        else:
            _configure_actor_output_layers(models, mcore_config)

        return models

    if bridge is not None and bridge_type == "mcore-bridge":
        if is_critic or use_lora:
            raise NotImplementedError(
                "mcore-bridge currently supports the base actor model only; "
                "critic heads and LoRA are not adapted."
            )
        models = bridge.get_model(
            wrap_with_ddp=mcore_config.wrap_with_ddp,
            ddp_config=dataclasses.asdict(mcore_config.ddp),
            use_torch_fsdp2=mcore_config.use_torch_fsdp2,
            use_custom_fsdp=mcore_config.use_custom_fsdp,
            overlap_param_gather_with_optimizer_step=mcore_config.overlap_param_gather_with_optimizer_step,
        )
        models = list(models)
        _configure_actor_output_layers(models, mcore_config)
        return models

    else:
        if is_critic and getattr(tf_config, "mtp_num_layers", None):
            logger.warning(
                "Dropping MTP head (mtp_num_layers=%s -> None) for critic model.",
                tf_config.mtp_num_layers,
            )
            tf_config.mtp_num_layers = None
        if mcore_config is not None and mcore_config.enable_router_replay:
            tf_config.moe_enable_routing_replay = True
        if (
            mcore_config is not None
            and mcore_config.virtual_pipeline_parallel_size is not None
            and mcore_config.virtual_pipeline_parallel_size > 1
        ):
            raise NotImplementedError(
                "Virtual pipeline parallelism requires mbridge-backed models."
            )
        transformer_layer_spec = make_mcore_layer_specs(hf_config, tf_config)
        rope_scaling_args = {}
        if hf_config.rope_scaling is not None:
            if hf_config.rope_scaling["type"] != "linear":
                raise NotImplementedError(
                    f"Rope scaling type {hf_config.rope_scaling['type']} not supported yet."
                )
            rope_scaling_args["seq_len_interpolation_factor"] = hf_config.rope_scaling[
                "factor"
            ]

        model = GPTModel(
            config=tf_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=hf_config.vocab_size,
            max_sequence_length=hf_config.max_position_embeddings,
            pre_process=True,  # TODO: pipeline parallel
            post_process=True,  # TODO: pipeline parallel
            share_embeddings_and_output_weights=False,  # TODO: implement share output weights
            position_embedding_type="rope",
            rotary_base=hf_config.rope_theta,
            **rope_scaling_args,
            # vp_stage=None TODO: virtual pipeline parallel
        )

        # Install ValueHead before DDP wrapping for critic models
        if is_critic:
            _install_value_head_on_model(model, tf_config)
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
            result = [wrapped]
        else:
            result = [model]

        if is_critic:
            _verify_critic_value_heads(result)
        return result
