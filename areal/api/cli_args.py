# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import warnings
from dataclasses import MISSING as dataclass_missing
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import uvloop
import yaml
from hydra import compose as hydra_compose
from hydra import initialize as hydra_init
from hydra.core.global_hydra import GlobalHydra
from omegaconf import MISSING, DictConfig, OmegaConf

from areal.engine.fsdp_utils.attn_impl import (
    BUILTIN_ATTN_IMPLS,
    get_attn_impl_validation_error,
    is_valid_attn_impl,
)
from areal.utils import logging, name_resolve, pkg_version
from areal.utils.constants import (
    PROX_LOGP_METHOD_RECOMPUTE,
    PROX_LOGP_METHODS_ALL,
)
from areal.utils.seqpack import PACKING_ALGORITHMS

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

uvloop.install()

logger = logging.getLogger("CLIArgs")

ConfigT = TypeVar("ConfigT")


@dataclass
class NormConfig:
    """Configuration for reward/advantage normalization."""

    mean_level: str | None = field(
        default="batch",
        metadata={
            "help": "Mean level for normalization. None for no mean normalization.",
            "choices": ["batch", "group", None],
        },
    )
    mean_leave1out: bool = field(
        default=False,
        metadata={"help": "Whether to use leave-one-out average."},
    )
    std_level: str | None = field(
        default="batch",
        metadata={
            "help": "Standard deviation level for normalization. None for no std normalization.",
            "choices": ["batch", "group", None],
        },
    )
    std_unbiased: bool = field(
        default=True,
        metadata={
            "help": "Whether to use unbiased standard deviation computation. Defaults to True (changed from False in v0.3.4)."
        },
    )
    eps: float = field(
        default=1e-5,
        metadata={
            "help": "The eps when dividing by standard deviation to avoid numerical issues."
        },
    )
    group_size: int = field(
        default=1, metadata={"help": "Group size for group-level normalization"}
    )

    def __post_init__(self):
        """Validate normalization configuration."""
        valid_levels = {"batch", "group", None}
        if self.mean_level not in valid_levels:
            raise ValueError(
                f"mean_level must be 'batch', 'group' or None, got {self.mean_level}"
            )
        if self.std_level not in valid_levels:
            raise ValueError(
                f"std_level must be 'batch', 'group', or None, got {self.std_level}"
            )
        if (
            self.mean_level == "group" or self.std_level == "group"
        ) and self.group_size < 1:
            raise ValueError(
                f"group_size must be a positive integer when using group normalization, got {self.group_size}"
            )


@dataclass
class MicroBatchSpec:
    """Specification for splitting micro-batches during training."""

    n_mbs: int | None = field(
        default=1,
        metadata={
            "help": "Number of micro-batches (or minimum number if max_tokens_per_mb is set). Used when max_tokens_per_mb is None or as minimum count",
        },
    )
    granularity: int = field(
        default=1,
        metadata={
            "help": "Granularity of each micro-batch. Adjacent sequences are grouped by this size when dividing microbatches.",
        },
    )
    max_tokens_per_mb: int | None = field(
        default=None,
        metadata={
            "help": "Maximum tokens per micro-batch for each forward pass. When set, n_mbs becomes the minimum number of micro-batches.",
        },
    )
    n_mbs_divisor: int = field(
        default=1,
        metadata={
            "help": "Divisor for the number of micro-batches. The final number of micro-batches will be adjusted to be divisible by this value.",
        },
    )
    packing_algorithm: str = field(
        default="ffd",
        metadata={
            "help": (
                "Sequence packing algorithm for micro-batch allocation. "
                "Supported values: 'ffd' (First Fit Decreasing, default), "
                "'kk' (Karmarkar-Karp, better balance but slightly slower). "
                "KK is recommended when workload balance across DP ranks is "
                "critical (e.g., large-scale RL training with variable-length sequences)."
            ),
            "choices": ["ffd", "kk"],
        },
    )

    def __post_init__(self):
        """Validate packing algorithm configuration."""
        if self.packing_algorithm not in PACKING_ALGORITHMS:
            raise ValueError(
                f"packing_algorithm must be one of {sorted(PACKING_ALGORITHMS)}, "
                f"got '{self.packing_algorithm}'"
            )

    @classmethod
    def new(cls, mb_spec: "MicroBatchSpec", **kwargs):
        """Create new spec with updated fields while maintaining Omegaconf compatibility."""
        fields = dict(
            n_mbs=mb_spec.n_mbs,
            granularity=mb_spec.granularity,
            max_tokens_per_mb=mb_spec.max_tokens_per_mb,
            n_mbs_divisor=mb_spec.n_mbs_divisor,
            packing_algorithm=mb_spec.packing_algorithm,
        )
        fields.update(kwargs)
        return cls(**fields)


@dataclass
class GenerationHyperparameters:
    """Controls text generation behavior for rollout."""

    n_samples: int = field(
        default=1, metadata={"help": "Number of sequences to generate per prompt."}
    )
    max_new_tokens: int = field(
        default=16384, metadata={"help": "Maximum number of tokens to generate."}
    )
    min_new_tokens: int = field(
        default=0, metadata={"help": "Minimum number of tokens to generate."}
    )
    max_tokens: int = field(
        default=32768,
        metadata={
            "help": "Maximum number of tokens including prompt and generated tokens."
        },
    )
    greedy: bool = field(
        default=False,
        metadata={"help": "Whether to use greedy decoding (max probability)."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Nucleus sampling probability threshold (0.0, 1.0]."},
    )
    top_k: int = field(
        default=int(1e8),
        metadata={"help": "Number of highest probability tokens to consider."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature. Higher values increase diversity."},
    )
    stop_token_ids: list[int] = field(
        default_factory=list,
        metadata={"help": "Stop generation when encountering these token IDs."},
    )
    ignore_eos: bool = field(
        default=False,
        metadata={"help": "Do not stop generation when EOS is encountered."},
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={"help": "Skip special tokens when decoding/displaying outputs."},
    )
    stop: list[str] | None = field(
        default=None,
        metadata={
            "help": "One or multiple stop words. Generation will stop if one of these words is sampled."
        },
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Penalizes tokens based on their frequency in generation so far. "
                "Must be between -2 and 2 where negative numbers encourage repetition."
            )
        },
    )
    lora_name: str = field(
        default="default_lora",
        metadata={"help": "Lora name to be used for this generation."},
    )
    use_beam_search: bool = field(
        default=False,
        metadata={
            "help": "Enable beam search in the vLLM engine. When enabled, sampling parameters like temperature, top-p, and top-k are auto ignored."
        },
    )
    # NOTE: to add new parameters, please correctly handle them in the `to_openai_args_dict` method.

    def new(self, **kwargs):
        args = asdict(self)
        args.update(kwargs)
        return GenerationHyperparameters(**args)

    def new_with_stop_and_pad_token_ids(self, tokenizer: "PreTrainedTokenizerFast"):
        """Create a new generation hyperparameters with stop and pad token ids added."""
        new_stop_token_ids = self.stop_token_ids.copy()
        if tokenizer.pad_token_id not in new_stop_token_ids:
            new_stop_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id not in new_stop_token_ids:
            new_stop_token_ids.append(tokenizer.eos_token_id)
        return self.new(stop_token_ids=new_stop_token_ids)

    def to_openai_completions_args_dict(
        self, exclude_args: list[str] | None = None
    ) -> dict[str, Any]:
        return self.to_openai_args_dict(
            exclude_args=exclude_args, api_format="completions"
        )

    def to_openai_responses_args_dict(
        self, exclude_args: list[str] | None = None
    ) -> dict[str, Any]:
        return self.to_openai_args_dict(
            exclude_args=exclude_args, api_format="responses"
        )

    def to_openai_agents_model_settings_dict(
        self, exclude_args: list[str] | None = None
    ) -> dict[str, Any]:
        return self.to_openai_args_dict(
            exclude_args=exclude_args, api_format="openai-agents"
        )

    _OPENAI_UNSUPPORTED_ARGS: ClassVar[set[str]] = {
        "min_new_tokens",  # Not supported by OpenAI
        "greedy",  # Not directly supported by OpenAI
        "top_k",  # Not supported by OpenAI
        "stop_token_ids",  # Not supported by OpenAI
        "ignore_eos",  # Not supported by OpenAI
        "skip_special_tokens",  # Not supported by OpenAI
        "lora_name",  # Not supported by OpenAI
        "use_beam_search",  # Not supported by OpenAI
        "max_tokens",  # deprecated by "completions", not used in "responses", should be `max_new_tokens` in "openai-agents"
    }

    def to_openai_args_dict(
        self, exclude_args: list[str] | None = None, api_format: str = "completions"
    ) -> dict[str, Any]:
        """Convert the generation hyperparameters to a dictionary of arguments for OpenAI client."""
        final_exclude_args = set(exclude_args) if exclude_args is not None else set()
        final_exclude_args.update(self._OPENAI_UNSUPPORTED_ARGS)
        # TODO: move the excluded args into extra body, so they can be passed through the client request

        mapping = {"n_samples": "n"}
        if api_format == "completions":
            mapping["max_new_tokens"] = "max_completion_tokens"
        elif api_format == "responses":
            mapping["max_new_tokens"] = "max_output_tokens"
        elif api_format == "openai-agents":
            # NOTE: max_tokens in openai-agents means `max_new_tokens` in sglang/vllm. This is not a bug
            mapping["max_new_tokens"] = "max_tokens"
        else:
            raise ValueError(f"Unsupported API format: {api_format}")

        res = {}
        for k, v in asdict(self).items():
            if k in final_exclude_args:
                should_warn = False

                current_value = getattr(self, k)
                f = next(_field for _field in fields(self) if _field.name == k)

                # Check if equal to the default value
                if f.default is not dataclass_missing:
                    if current_value != f.default:
                        should_warn = True
                elif f.default_factory is not dataclass_missing:
                    if current_value != f.default_factory():
                        should_warn = True

                if should_warn:
                    logger.warning(
                        f"Unsupported arg for openai format: `{k}` with value {current_value}"
                    )
                continue
            key = mapping.get(k, k)
            if key in res:
                logger.warning(f"Overriding key: {key} from {k} with value: {v}")
            res[key] = v

        return res


# Train Engine Configs


@dataclass
class OptimizerConfig:
    """Configuration for model optimization during training."""

    type: str = field(
        default="adam",
        metadata={
            "help": "Optimizer type. For FSDP Engine, adam_bf16 enables memory-efficient BF16 optimizer states. "
            "For Megatron Engine, adam_bf16 requires dtype=bfloat16 and is automatically converted to adam "
            "with precision-aware optimizer enabled.",
            "choices": ["adam", "sgd", "adam_bf16"],
        },
    )
    lr: float = field(default=1e-3, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})
    beta1: float = field(
        default=0.9,
        metadata={
            "help": "Adam beta1 parameter. Only effective when optimizer_type is adam/adam_bf16"
        },
    )
    beta2: float = field(
        default=0.999,
        metadata={
            "help": "Adam beta2 parameter. Only effective when optimizer_type is adam/adam_bf16"
        },
    )
    eps: float = field(
        default=1e-8,
        metadata={
            "help": "Adam epsilon parameter. Only effective when optimizer_type is adam/adam_bf16"
        },
    )
    min_lr_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Minimum learning rate ratio after annealing",
        },
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate scheduler type",
            "choices": ["linear", "cosine", "constant"],
        },
    )
    warmup_steps_proportion: float = field(
        default=0.001,
        metadata={
            "help": "Proportion of training steps for warmup",
        },
    )
    initial_loss_scale: float = field(
        default=2**32, metadata={"help": "Initial loss scaling factor"}
    )
    min_loss_scale: float = field(
        default=1.0, metadata={"help": "Minimum loss scaling factor"}
    )
    loss_scale_window: float = field(
        default=5, metadata={"help": "Window size for loss scaling adjustment"}
    )
    hysteresis: int = field(
        default=2, metadata={"help": "Hysteresis (scaling factor) for loss scaling"}
    )
    gradient_clipping: float = field(
        default=1.0, metadata={"help": "Gradient clipping threshold"}
    )


@dataclass
class FSDPWrapPolicy:
    """Policy configuration for FSDP model layer wrapping. None defaults to wrapping transformer decoder layers defined by transformers."""

    transformer_layer_cls_to_wrap: list[str] | None = field(
        default=None,
        metadata={"help": "A list of transformer layer names for FSDP to wrap."},
    )


@dataclass
class FSDPEngineConfig:
    """Configuration for Fully Sharded Data Parallel (FSDP) training backend."""

    wrap_policy: FSDPWrapPolicy | None = field(
        default=None,
        metadata={"help": "FSDP wrap policy, specifying model layers to wrap."},
    )
    offload_params: bool = field(
        default=False,
        metadata={"help": "Whether to offload FSDP parameters to CPU."},
    )
    memory_efficient_load: bool = field(
        default=False,
        metadata={
            "help": "Enable memory-efficient model loading. When enabled, model weights "
            "are initialized on CPU and only rank 0 loads pretrained weights, which are "
            "then broadcast to all ranks after FSDP sharding. This reduces peak GPU memory "
            "during initialization for large models. Note: For VLMs, rank 0 broadcast is "
            "not used; each rank loads weights independently on CPU."
        },
    )
    per_layer_optim_step: bool = field(
        default=False,
        metadata={
            "help": "Run Adam step on GPU by streaming optimizer states layer-by-layer "
            "with async prefetching, instead of running on CPU. Optimizer states are "
            "automatically managed on CPU by the per-layer wrapper regardless of "
            "offload_params setting. Requires optimizer type 'adam' (AdamW)."
        },
    )
    optim_step_prefetch_layers: int = field(
        default=1,
        metadata={"help": "Number of layers to prefetch during per-layer optim step."},
    )

    def __post_init__(self):
        if self.optim_step_prefetch_layers < 0:
            raise ValueError(
                f"optim_step_prefetch_layers must be >= 0, got {self.optim_step_prefetch_layers}"
            )

    shard_vision_across_sp: bool = field(
        default=False,
        metadata={
            "help": "Shard vision encoder across SP ranks by image. "
            "Only effective when context_parallel_size > 1."
        },
    )


@dataclass
class ArchonFP8Config:
    """Archon FP8 training configuration."""

    mode: str = field(
        default="disabled",
        metadata={
            "help": "FP8 precision mode. "
            "'disabled': FP8 training off (default). "
            "'blockwise': blockwise 128x128 FP8 e4m3fn matmuls (requires Hopper GPU).",
            "choices": ["disabled", "blockwise"],
        },
    )

    exclude_modules: list[str] = field(
        default_factory=lambda: ["output", "router", "score"],
        metadata={
            "help": (
                "FQN substrings of nn.Linear modules to keep in BF16 (not converted to FP8). "
                "Any module whose fully-qualified name contains one of these strings is skipped. "
                "Meaningful values for Archon models: "
                "'output' (LM head, logit precision sensitive), "
                "'router' (MoE router gate, routing stability sensitive), "
                "'score' (critic head, value precision sensitive). "
                "Note: nn.Embedding modules (e.g. tok_embeddings) are never converted "
                "regardless of this list. "
                "WARNING: Setting this in YAML replaces the entire default list "
                "(does not extend it). Include ALL modules you want to keep in BF16."
            )
        },
    )

    include_experts: bool = field(
        default=False,
        metadata={
            "help": "Apply FP8 to MoE expert computation. "
            "Uses per-expert blockwise FP8 matmuls via torchao."
        },
    )

    use_triton: bool = field(
        default=True,
        metadata={
            "help": (
                "Use Triton GEMM kernel for FP8 blockwise matmuls instead of cuBLAS. "
                "Currently must be True: torchao's blockwise FP8 is a prototype that uses "
                "mixed per-operand scaling (1x128 activations + 128x128 weights), which "
                "torch._scaled_mm does not support. The Triton kernel "
                "(triton_fp8_gemm_1x128_128x128) handles this natively. "
                "Revisit when torchao stabilizes mixed-mode cuBLAS dispatch."
            ),
        },
    )

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    def __post_init__(self):
        valid_modes = {"disabled", "blockwise"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"fp8_config.mode must be one of {valid_modes}, got {self.mode!r}"
            )
        if self.enabled and not self.use_triton:
            raise ValueError(
                "fp8_config.use_triton must be True when FP8 is enabled. "
                "torchao blockwise FP8 uses mixed per-operand scaling "
                "(1x128 activations + 128x128 weights) which "
                "torch._scaled_mm does not support."
            )


@dataclass
class ArchonEngineConfig:
    """Configuration for Archon Engine training backend."""

    # Attention backend
    attn_type: str = field(
        default="varlen",
        metadata={
            "help": "Attention backend type. Use 'tree' for tree training.",
            "choices": ["varlen", "sdpa", "tree"],
        },
    )

    # CPU offloading for FSDP
    offload_params: bool = field(
        default=False,
        metadata={"help": "Whether to offload FSDP parameters to CPU."},
    )

    # Whether to enable torch.compile
    enable_compile: bool = field(
        default=False,
        metadata={"help": "Enable torch.compile for TransformerBlocks."},
    )

    # Activation Checkpointing (enabled when gradient_checkpointing=True)
    ac_mode: str = field(
        default="selective",
        metadata={
            "help": "Activation checkpointing mode. "
            "'memory_budget' requires enable_compile=True.",
            "choices": ["none", "full", "selective", "memory_budget"],
        },
    )
    selective_ac_option: str = field(
        default="op",
        metadata={
            "help": "Selective AC option: 'op' for op-level, "
            "or integer string (e.g., '2') for every Nth layer."
        },
    )
    ac_memory_budget: float = field(
        default=0.5,
        metadata={
            "help": "Memory budget for 'memory_budget' AC mode. "
            "0.0 = minimum memory (max recompute), 1.0 = default behavior (no recompute)."
        },
    )
    ac_preserve_rng_state: bool = field(
        default=False,
        metadata={
            "help": "Preserve RNG state during checkpointing for deterministic output. "
            "Enabling this may slow down training."
        },
    )
    ac_debug: bool = field(
        default=False,
        metadata={
            "help": "(Testing only) Capture AC debug information. Will be slower."
        },
    )

    # Pipeline Parallel Schedule
    pp_schedule: str = field(
        default="Interleaved1F1B",
        metadata={
            "help": "Pipeline parallel schedule type.",
            "choices": [
                "1F1B",
                "Interleaved1F1B",
                "InterleavedZeroBubble",
                "ZBVZeroBubble",
            ],
        },
    )
    # NOTE: The following three PP layer distribution parameters are advanced options
    # that most users do not need to configure. The defaults work well for typical cases.
    # TODO: Consider simplifying or refactoring these parameters in the future.
    # Currently kept for consistency with Megatron's pipeline parallel configuration.
    pp_layers_per_stage: int | None = field(
        default=None,
        metadata={
            "help": "Number of transformer layers per (virtual) pipeline stage. "
            "If set, num_virtual_stages is calculated from num_layers. "
            "If None, stages are inferred from schedule type "
            "(1 stage/rank for 1F1B, 2 stages/rank for Interleaved1F1B/InterleavedZeroBubble/ZBVZeroBubble).",
        },
    )
    pp_first_stage_less_layers: int = field(
        default=1,
        metadata={
            "help": "Number of layers to reduce in the first pipeline stage. "
            "Accounts for embedding layer overhead.",
        },
    )
    pp_last_stage_less_layers: int = field(
        default=1,
        metadata={
            "help": "Number of layers to reduce in the last pipeline stage. "
            "Accounts for output layer overhead.",
        },
    )

    # FSDP reshard policy after forward pass
    reshard_after_forward_policy: str = field(
        default="default",
        metadata={
            "help": "FSDP reshard policy after forward pass. "
            "'default': reshard when pipeline parallelism is off; keep unsharded when on to avoid repeated all-gather per microbatch. "
            "'always': always reshard after forward (saves memory). "
            "'never': never reshard after forward.",
            "choices": ["default", "always", "never"],
        },
    )

    # FP8 Training
    fp8_config: ArchonFP8Config = field(
        default_factory=ArchonFP8Config,
        metadata={
            "help": "FP8 training configuration. Set mode='blockwise' to enable."
        },
    )

    # Deterministic mode
    use_deterministic_algorithms: bool = field(
        default=False,
        metadata={
            "help": "Enable deterministic algorithms for training reproducibility. "
            "Sets torch.use_deterministic_algorithms(True, warn_only=True), "
            "CUBLAS_WORKSPACE_CONFIG, NCCL_ALGO, and TORCH_COMPILE_DETERMINISTIC. "
            "May reduce performance.",
        },
    )

    # MoE
    moe_router_dtype: str | None = field(
        default="fp32",
        metadata={
            "help": "Data type for MoE router gate GEMM computation. "
            "'fp32' runs gate linear in float32 for numerical stability. "
            "None uses model dtype (no override).",
            "choices": ["fp32", None],
        },
    )

    def __post_init__(self):
        if self.pp_layers_per_stage is not None and self.pp_layers_per_stage < 1:
            raise ValueError(
                f"pp_layers_per_stage must be >= 1, got {self.pp_layers_per_stage}"
            )
        if self.pp_first_stage_less_layers < 0:
            raise ValueError(
                f"pp_first_stage_less_layers must be >= 0, "
                f"got {self.pp_first_stage_less_layers}"
            )
        if self.pp_last_stage_less_layers < 0:
            raise ValueError(
                f"pp_last_stage_less_layers must be >= 0, "
                f"got {self.pp_last_stage_less_layers}"
            )
        valid_reshard_policies = ("default", "always", "never")
        if self.reshard_after_forward_policy not in valid_reshard_policies:
            raise ValueError(
                f"reshard_after_forward_policy must be one of {valid_reshard_policies}, "
                f"got '{self.reshard_after_forward_policy}'"
            )
        valid_router_dtypes = ("fp32", None)
        if self.moe_router_dtype not in valid_router_dtypes:
            raise ValueError(
                f"moe_router_dtype must be one of {valid_router_dtypes}, "
                f"got '{self.moe_router_dtype}'"
            )


# These configurations are used by Megatron Bridge to build Megatron models.
@dataclass
class DistributedDataParallelConfig:
    """Configuration for Megatron's DistributedDataParallel.
    Refer to Megatron-LM documentation for details.
    """

    grad_reduce_in_fp32: bool = True
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    align_param_gather: bool = False
    use_distributed_optimizer: bool = True
    check_for_nan_in_grad: bool = False
    bucket_size: int | None = None
    average_in_collective: bool = False
    fp8_param_gather: bool = False


@dataclass
class FP8EngineConfig:
    """Configuration for FP8 (8-bit floating point) training.

    This configuration encapsulates all FP8-related parameters and can be reused
    across different engines (e.g., Megatron, FSDP). When None in the parent config,
    FP8 training is disabled.
    """

    mode: str = field(
        default="e4m3",
        metadata={
            "help": "FP8 precision mode. Options: "
            "'e4m3' (uniform e4m3), "
            "'hybrid' (e4m3 for activations/weights, e5m2 for output activation gradients)."
        },
    )

    recipe: str = field(
        default="delayed",
        metadata={
            "help": "FP8 scaling recipe. Options: 'tensorwise', 'delayed', 'mxfp8' (Blackwell only), 'blockwise'."
        },
    )

    param: bool = field(
        default=False,
        metadata={
            "help": "Keep parameters in FP8 precision to save memory. "
            "Not all parameters will be converted to fp8; for example, biases will remain unchanged."
        },
    )

    margin: int = field(
        default=0,
        metadata={"help": "Margin for FP8 scaling factor computation."},
    )

    amax_history_len: int = field(
        default=1,
        metadata={
            "help": "Length of amax history window for scaling factor computation."
        },
    )

    amax_compute_algo: str = field(
        default="most_recent",
        metadata={
            "help": "Algorithm for choosing amax value. Options: 'max' (largest in history window), 'most_recent'."
        },
    )

    wgrad: bool = field(
        default=True,
        metadata={
            "help": "When False, override FP8 config and compute weight gradients in higher precision."
        },
    )

    dot_product_attention: bool = field(
        default=False,
        metadata={"help": "Use FP8 implementation of Dot Product Attention."},
    )

    multi_head_attention: bool = field(
        default=False,
        metadata={"help": "Use FP8 implementation of Multi Head Attention."},
    )

    tp_only_amax_red: bool = field(
        default=False,
        metadata={"help": "Reduce FP8 AMAX only in TP or TP-CP domain."},
    )

    first_last_layers_bf16: bool = field(
        default=False,
        metadata={
            "help": "Retain first and last N TransformerBlocks in BF16 instead of FP8."
        },
    )

    num_layers_at_start_in_bf16: int = field(
        default=1,
        metadata={
            "help": "Number of layers at start to keep in BF16 when first_last_layers_bf16 is True."
        },
    )

    num_layers_at_end_in_bf16: int = field(
        default=1,
        metadata={
            "help": "Number of layers at end to keep in BF16 when first_last_layers_bf16 is True."
        },
    )

    direct_convert: bool = field(
        default=True,
        metadata={
            "help": "Whether to use direct FP8 conversion during weight updates and save/load. "
            "When True, FP8 parameters are directly converted between TE FP8 and PyTorch FP8 "
            "without intermediate dequantization/quantization."
        },
    )


@dataclass
class MegatronEngineConfig:
    """Configuration for Megatron-LM training framework.
    Refer to Megatron-LM documentation for implementation details.
    """

    # Distributed Training Configuration
    wrap_with_ddp: bool = True
    use_torch_fsdp2: bool = False  # TODO: pending test
    use_custom_fsdp: bool = False  # TODO: pending test
    ddp: DistributedDataParallelConfig = field(
        default_factory=DistributedDataParallelConfig
    )
    virtual_pipeline_parallel_size: int = field(
        default=1,
        metadata={
            "help": (
                "Virtual pipeline parallel size for Megatron interleaved schedule. "
                "Set to >1 to enable VPP. Default is 1 (disabled)."
            )
        },
    )
    # Don't use MegatronOptimizerConfig here because OmegaConf
    # does not recognize the annotation "torch.dtype"
    overlap_param_gather_with_optimizer_step: bool = False

    # Precision Configuration
    use_precision_aware_optimizer: bool = field(
        default=False,
        metadata={
            "help": "Enable precision-aware optimizer for Megatron. "
            "When using adam_bf16 optimizer type with Megatron Engine, "
            "this is automatically enabled with exp_avg_dtype=bfloat16 and exp_avg_sq_dtype=bfloat16."
        },
    )
    main_grads_dtype: str = "float32"
    main_params_dtype: str = "float32"
    exp_avg_dtype: str = "float32"
    exp_avg_sq_dtype: str = "float32"

    # Checkpointing Configuration
    async_save: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, Megatron checkpoint saves run in background processes and "
                "save_checkpoint() returns immediately after weights are durably "
                "staged off the GPU. Pending saves are drained before the next "
                "load_checkpoint() and during engine.destroy(). Reduces per-save "
                "sync wait on large MoE checkpoints."
            ),
        },
    )
    use_checkpoint_opt_param_scheduler: bool = True

    # Deterministic Option
    # NOTE: This option forces torch to use deterministic algorithms,
    # which makes sure that two forward passes with the same input
    # will produce the same output. However, it may have a performance impact.
    # It is recommended to set this option to True for RL training on MoE models for stability.
    use_deterministic_algorithms: bool = False

    # Gradient checkpointing options, only effective when gradient_checkpointing=True
    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1
    distribute_saved_activations: bool | None = None
    recompute_modules: list[str] | None = None

    # MoE
    moe_router_dtype: str | None = "fp32"
    moe_shared_expert_overlap: bool = field(
        default=False,
        metadata={
            "help": "Enable overlapping between shared expert computations and dispatcher communications. "
            "Without this, the shared experts execute after the routed experts."
        },
    )
    moe_enable_deepep: bool = False
    moe_token_dispatcher_type: str = field(
        default="alltoall",
        metadata={
            "help": "Type of token dispatcher. Options: 'allgather','alltoall' and 'flex'."
        },
    )
    moe_permute_fusion: bool = field(
        default=False,
        metadata={"help": "Fuse token rearrangement ops during token dispatching."},
    )

    # FP8 Training Configuration
    fp8_config: FP8EngineConfig | None = None

    # Bridge backend used for HF<->Megatron conversion/model creation.
    bridge_type: str = field(
        default="mbridge",
        metadata={
            "help": "Bridge backend for MegatronEngine. Choices: 'mbridge' or 'megatron-bridge'.",
            "choices": ["mbridge", "megatron-bridge"],
        },
    )

    use_mbridge_save: bool = field(
        default=False,
        metadata={
            "help": "Use mbridge's save method to save gpu memory when saving weights."
        },
    )

    use_bridge_for_update_weights: bool = field(
        default=False,
        metadata={
            "help": "When True and bridge_type='megatron-bridge', delegate live "
            "weight sync to bridge.export_hf_weights instead of the hand-rolled "
            "convert_to_hf registry. Required for models without a registry entry "
            "(e.g. Qwen3.5). FP8 paths fall back to the registry automatically.",
        },
    )

    disable_grad_buffers_cpu_backup: bool = field(
        default=False,
        metadata={
            "help": (
                "When offloading with torch_memory_saver, skip CPU backup for "
                "Megatron gradient buffers (they are recomputed each step). "
            )
        },
    )


class SchedulingStrategyType(str, Enum):
    separation = "separation"
    colocation = "colocation"


@dataclass
class SchedulingStrategy:
    type: str = field(
        default="separation",
        metadata={"choices": ["separation", "colocation"]},
    )
    target: str | None = field(
        default=None, metadata={"help": "The target role to be colocated with"}
    )
    fork: bool = field(
        default=True,
        metadata={
            "help": "When True with colocation, the target worker spawns a new "
            "process on the same node/GPUs instead of sharing its process. "
            "Provides process isolation while sharing GPU resources."
        },
    )


@dataclass
class SchedulingSpec:
    cpu: int = field(
        default=8, metadata={"help": "Number of CPU cores required per GPU"}
    )
    gpu: int = field(
        default=0,
        metadata={
            "help": "Number of GPU units required. Used only when allocating pods."
        },
    )
    mem: int = field(
        default=32, metadata={"help": "Amount of memory (GB) required per GPU"}
    )
    port_count: int = field(default=2, metadata={"help": "Number of ports to expose"})
    image: str = field(
        default="/storage/openpsi/images/areal-latest.sif",
        metadata={
            "help": "Docker/Singularity container image to use. "
            "Currently only used by Slurm. Will be potentially used by Kubernetes in the future."
        },
    )
    task_type: str = field(
        default="worker",
        metadata={
            "help": "Task type (e.g., worker, engine)",
            "choices": ["worker", "engine"],
        },
    )
    env_vars: dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Environment variables for the container"},
    )
    cmd: str | None = field(
        default=None,
        metadata={
            "help": "Command to execute inside the container. Defaults to AReaL's RPC server."
        },
    )
    # Slurm specific options
    srun_additional_args: str = field(
        default="--unbuffered --mpi=pmi2 -K --chdir $PWD",
        metadata={
            "help": "Additional arguments to pass to the srun command. Only used by slurm."
        },
    )
    additional_bash_cmds: list[str] | None = field(
        default=None,
        metadata={
            "help": "Additional bash commands to setup the container before running "
            "the torchrun command. Only used by slurm."
        },
    )
    container_type: str = field(
        default="apptainer",
        metadata={
            "help": "Type of containers used in slurm",
            "choices": ["apptainer", "none"],
        },
    )
    mount: str = field(
        default="/storage:/storage", metadata={"help": "Mount path for slurm."}
    )
    nodelist: str | None = field(
        default=None, metadata={"help": "sbatch/srun's `--nodelist` option for slurm."}
    )
    exclude: str | None = field(
        default=None, metadata={"help": "sbatch/srun's `--exclude` option for slurm."}
    )
    ray_placement_strategy: str = field(
        default="shared",
        metadata={
            "help": "Which placement strategy to use for Ray scheduling. "
            "Shared will produce 1 placement group for all workers in the role (training). "
            "Separate will 1 placement group per worker (rollout). "
            "Deferred will do the same as separate but defers accelerator scheduling (multinode rollout). ",
            "choices": ["shared", "separate", "deferred"],
        },
    )

    def __post_init__(self):
        """Validate scheduling spec configuration."""
        valid_strategies = {"shared", "separate", "deferred"}
        if self.ray_placement_strategy not in valid_strategies:
            raise ValueError(
                f"ray_placement_strategy must be one of {valid_strategies}, "
                f"got '{self.ray_placement_strategy}'"
            )


@dataclass
class TrainEngineConfig:
    """Core configuration for model training, including optimization and backend settings."""

    experiment_name: str = MISSING
    trial_name: str = MISSING
    path: str = field(default="", metadata={"help": "Path to HuggingFace checkpoint"})
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={
            "help": "Attention implementation for huggingface transformers model. "
            "Accepts builtin transformers backends or a Hugging Face kernels repo ID "
            "formatted as org/repo[@revision][:entrypoint].",
            "choices": list(BUILTIN_ATTN_IMPLS),
        },
    )
    use_kernels: bool = field(
        default=False,
        metadata={
            "help": "Enable Hugging Face kernels model kernelization after model creation."
        },
    )
    init_from_scratch: bool = field(
        default=False, metadata={"help": "Initialize model weights randomly"}
    )
    is_critic: bool = field(
        default=False,
        metadata={"help": "Whether to use a critic/reward model"},
    )
    temperature: float = field(
        default=1.0, metadata={"help": "Temperature during generation."}
    )
    # Runtime microbatch limit
    mb_spec: MicroBatchSpec = field(default_factory=MicroBatchSpec)
    pad_to_maximum: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad each microbatch to the length upper bound specified by mb_spec. "
                "Can reduce memory fragmentation but slows down training."
            )
        },
    )

    # Training Backend Configuration
    disable_dropout: bool = field(
        default=False, metadata={"help": "Disable dropout layers during training"}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Enable gradient checkpointing"}
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "Forward/backward compute dtype."},
    )
    grad_reduce_dtype: str = field(
        default="float32", metadata={"help": "Gradient reduction data type."}
    )
    optimizer_dtype: str = field(
        default="float32",
        metadata={
            "help": (
                "Underlying parameter storage dtype, also the dtype of optimizer "
                "states (exp_avg, exp_avg_sq) since torch.optim.AdamW inherits "
                "dtype from model.parameters(). "
                "Default 'float32' maintains fp32 master weights matching "
                "DeepSpeed ZeRO-3 and Megatron precision-aware optimizer behavior. "
                "FSDP2's MixedPrecisionPolicy(param_dtype=`dtype`) will still "
                "cast forward/backward computation to `dtype` (e.g. bfloat16). "
                "Set to 'bfloat16' together with optimizer.type='adam_bf16' to "
                "reduce memory at the cost of needing Kahan summation for stability. "
                "Currently FSDP-only; Megatron uses use_precision_aware_optimizer "
                "instead and ignores this field."
            )
        },
    )
    optimizer: OptimizerConfig | None = field(
        default=None,
        metadata={"help": "Optimizer configuration. None means no training."},
    )

    weight_update_mode: str = field(
        default="xccl",
        metadata={"help": "Weight update backend type.", "choices": ["disk", "xccl"]},
    )
    fsdp: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    archon: ArchonEngineConfig = field(default_factory=ArchonEngineConfig)
    megatron: MegatronEngineConfig = field(default_factory=MegatronEngineConfig)

    # offload
    offload: bool = field(
        default=False,
        metadata={
            "help": "Whether to offload model parameters and optimizer states to CPU. "
        },
    )

    # Lora
    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use LoRA. Only support FSDP. Note that should be enabled together with vLLM/SGLang."
        },
    )
    lora_rank: int = field(default=32, metadata={"help": "lora rank"})
    lora_alpha: int = field(default=16, metadata={"help": "lora alpha"})
    target_modules: list[str] = field(
        default_factory=list,
        metadata={"help": "lora target_modules."},
    )
    peft_type: str = field(
        default="lora",
        metadata={"help": "peft method type. Only LoRA is supported for now."},
    )

    # Tree training
    enable_tree_training: bool = field(
        default=False,
        metadata={"help": "Enable tree training with flex attention module."},
    )

    # Scheduling
    scheduling_spec: tuple[SchedulingSpec, ...] = field(
        default_factory=lambda: (
            SchedulingSpec(cmd="python -m areal.infra.rpc.rpc_server"),
        ),
        metadata={
            "help": "Train engine schedule specs. Can accept 1 or 2 SchedulingSpec: "
            "if 1 spec provided, it's used for both worker and engine, engine is embedded in the worker; "
            "if 2 specs provided, first one is for worker, second one is for engine. "
            "Currently only used by the TrainController."
        },
    )
    # Backend and parallelism (new per-engine config)
    backend: str = field(
        default=MISSING,
        metadata={
            "help": "Backend and parallelism strategy. Must include an explicit backend prefix, "
            "e.g. 'fsdp:d4', 'megatron:d4t2p2', 'archon:d2'. Required."
        },
    )

    # v2 controller options
    _version: str = field(
        default="v1",
        metadata={
            "help": "Train controller implementation version. Use 'v1' for legacy TrainController, 'v2' for GatewayTrainController.",
            "choices": ["v1", "v2"],
        },
    )
    admin_api_key: str = field(
        default="areal-admin-key",
        metadata={
            "help": "Admin API key used by gateway/router/data-proxy in controller v2."
        },
    )
    log_level: str = field(
        default="warning",
        metadata={"help": "Gateway stack log level for controller v2."},
    )
    request_timeout: float = field(
        default=3600.0,
        metadata={"help": "Gateway request timeout in seconds for controller v2."},
    )
    setup_timeout: float = field(
        default=3600.0,
        metadata={"help": "Gateway setup timeout in seconds for controller v2."},
    )
    workers_ready_timeout: float = field(
        default=30.0,
        metadata={
            "help": "Timeout (seconds) for initialize() to wait for guards to be ready."
        },
    )
    scheduling_strategy: SchedulingStrategy = field(
        default_factory=SchedulingStrategy,
        metadata={
            "help": "The scheduling strategy of this TrainEngine, either separation or colocation. "
            "Currently only used by the TrainController."
        },
    )

    def __post_init__(self):
        """Validate scheduling_spec length and config combinations."""
        if len(self.scheduling_spec) not in (1, 2):
            raise ValueError(
                f"scheduling_spec must contain 1 or 2 SchedulingSpec, "
                f"got {len(self.scheduling_spec)}"
            )
        if not is_valid_attn_impl(self.attn_impl):
            raise ValueError(get_attn_impl_validation_error(self.attn_impl))
        if self.fsdp.memory_efficient_load and self.init_from_scratch:
            raise ValueError(
                "memory_efficient_load cannot be used with init_from_scratch=True. "
                "memory_efficient_load is for loading pretrained weights on CPU, "
                "but init_from_scratch creates a model without loading any weights."
            )
        if self._version not in ("v1", "v2"):
            raise ValueError(
                f"_version must be either 'v1' or 'v2', got '{self._version}'"
            )

        # Canonicalize common aliases so getattr(torch, ...) works at runtime.
        # Storage map omits fp16 since float16 is not a valid optimizer_dtype;
        # leaving "fp16" un-canonicalized makes the validation error below
        # echo what the user typed instead of a silently rewritten value.
        _compute_aliases = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
        _storage_aliases = {"fp32": "float32", "bf16": "bfloat16"}
        if self.optimizer_dtype in _storage_aliases:
            self.optimizer_dtype = _storage_aliases[self.optimizer_dtype]
        if self.dtype in _compute_aliases:
            self.dtype = _compute_aliases[self.dtype]

        if self.optimizer_dtype not in ("float32", "bfloat16"):
            raise ValueError(
                f"optimizer_dtype must be 'float32' or 'bfloat16', "
                f"got {self.optimizer_dtype!r}"
            )
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError(
                f"dtype must be one of float32/bfloat16/float16, got {self.dtype!r}"
            )


@dataclass
class RejectionSamplingConfig:
    """Unified configuration for sample filtering based on policy divergence.

    Filters tokens/sequences where the divergence between proximal policy
    and behavior policy exceeds a threshold, via two action modes:
    - 'mask': zero out loss_mask (rejection, exclude from gradient)
    - 'clamp': clamp importance weight to bounds (truncation, bounded gradient)

    Supports direct ratio bounds and KL divergence estimators (K1/K2/K3),
    at both token-level and sequence-level granularity.

    Replaces the removed ``behave_imp_weight_cap`` and ``behave_imp_weight_mode``.

    Attributes:
        level: Filtering granularity ('token' or 'sequence'). When ``level='sequence'``
            and ``metric='ratio'``, both the filtering decision and the correction
            weight (behave_imp_weight) use the sequence-level geometric mean,
            matching the old ``sequence_mask``/``sequence_truncate`` semantics.
        action: Action mode ('mask' or 'clamp').
        metric: Divergence metric ('ratio', 'kl_k1', 'kl_k2', 'kl_k3').
        agg: Aggregation method for sequence-level ('sum', 'mean', 'max').
            For 'ratio' metric, aggregation is performed in log space (geometric
            mean/sum) to avoid the "length trap" and match GSPO semantics.
            For KL metrics, aggregation is arithmetic.
        upper: Upper bound for filtering.
        lower: Lower bound for filtering (optional).
    """

    level: str = field(
        default="token",
        metadata={
            "help": "Filtering granularity. "
            "'token': per-token filtering (each token judged independently). "
            "'sequence': per-sequence filtering (all tokens in a sequence share the same fate). "
            "When metric='ratio', both the filtering decision and the correction weight "
            "(behave_imp_weight) operate at sequence level using the geometric mean.",
            "choices": ["token", "sequence"],
        },
    )
    action: str = field(
        default="mask",
        metadata={
            "help": "Action to take when metric exceeds threshold. "
            "'mask': zero out loss_mask for filtered tokens/sequences (rejection, "
            "completely excludes from gradient computation). "
            "'clamp': clamp importance weight to [lower, upper] bounds (truncation, "
            "tokens still participate in gradient but with bounded weight).",
            "choices": ["mask", "clamp"],
        },
    )
    metric: str = field(
        default="ratio",
        metadata={
            "help": "Divergence metric for filtering. "
            "'ratio': direct importance ratio π_proximal/π_behave. "
            "'kl_k1': KL estimator k1 = log(r), forward KL unbiased estimator (can be negative). "
            "'kl_k2': KL estimator k2 = 0.5 * (log r)^2, non-negative quadratic approximation. "
            "'kl_k3': KL estimator k3 = r - log(r) - 1, non-negative exact forward KL estimator. "
            "'binary_kl': KPop (symmetric binary KL divergence) — masks tokens where either "
            "KL(proximal||behave) or KL(behave||proximal) exceeds the upper bound.",
            "choices": ["ratio", "kl_k1", "kl_k2", "kl_k3", "binary_kl"],
        },
    )
    agg: str = field(
        default="mean",
        metadata={
            "help": "Aggregation method for sequence-level filtering. "
            "Only used when level='sequence'. "
            "For 'ratio' metric, aggregation is in log space: "
            "'sum' = exp(sum(log(r_i))), 'mean' = exp(mean(log(r_i))) = geometric mean "
            "(length-invariant, consistent with GSPO). "
            "For KL metrics, aggregation is arithmetic: "
            "'sum' = sum(kl_i), 'mean' = mean(kl_i). "
            "'max': max of per-token metric values (most conservative).",
            "choices": ["sum", "mean", "max"],
        },
    )
    upper: float = field(
        default=5.0,
        metadata={
            "help": "Upper bound for filtering. "
            "Tokens/sequences with metric > upper are filtered out (loss_mask zeroed). "
            "For 'ratio' metric: must be > 1.0, typical values are 2.0 or 5.0. "
            "For 'kl_k2'/'kl_k3' metrics: typical values are 0.5-2.0."
        },
    )
    lower: float | None = field(
        default=None,
        metadata={
            "help": "Lower bound for filtering (optional). "
            "None means no lower bound. "
            "For 'ratio' metric: typical value is 0.5 (filter out tokens where policy "
            "probability dropped significantly). Must be > 0. "
            "For 'kl_k1' metric: can be used to filter negative KL estimates."
        },
    )

    def __post_init__(self):
        """Validate configuration."""
        import warnings

        _VALID_LEVELS = ("token", "sequence")
        _VALID_ACTIONS = ("mask", "clamp")
        _VALID_METRICS = ("ratio", "kl_k1", "kl_k2", "kl_k3")
        _VALID_AGGS = ("sum", "mean", "max")

        # Validate enum-like fields.
        if self.level not in _VALID_LEVELS:
            raise ValueError(
                f"level must be one of {_VALID_LEVELS}, got '{self.level}'"
            )
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {_VALID_ACTIONS}, got '{self.action}'"
            )
        if self.metric not in _VALID_METRICS:
            raise ValueError(
                f"metric must be one of {_VALID_METRICS}, got '{self.metric}'"
            )
        if self.agg not in _VALID_AGGS:
            raise ValueError(f"agg must be one of {_VALID_AGGS}, got '{self.agg}'")

        # Validate lower <= upper when both are set.
        if self.lower is not None and self.lower > self.upper:
            raise ValueError(
                f"lower ({self.lower}) cannot be greater than upper ({self.upper})"
            )

        # For ratio metric, upper must be > 1.0 (otherwise all non-identical policy tokens are filtered).
        if self.metric == "ratio":
            if self.upper <= 1.0:
                raise ValueError(
                    f"upper must be > 1.0 for 'ratio' metric (otherwise all non-identical "
                    f"policy tokens will be filtered), got {self.upper}"
                )
            if self.lower is not None and self.lower <= 0:
                raise ValueError(
                    f"lower must be positive for 'ratio' metric, got {self.lower}"
                )
        # For KL metrics, upper must be > 0.
        # Note: kl_k1 is excluded because it is a forward KL unbiased estimator that
        # can produce negative values, so requiring upper > 0 would be too restrictive.
        if self.metric in ("kl_k2", "kl_k3") and self.upper <= 0:
            raise ValueError(
                f"upper must be positive for '{self.metric}' metric, got {self.upper}"
            )
        # Clamp action only supports ratio metric (direct importance weight truncation).
        if self.action == "clamp" and self.metric != "ratio":
            raise ValueError(
                f"action='clamp' only supports metric='ratio' (direct importance weight "
                f"truncation). Got metric='{self.metric}'. "
                f"Use action='mask' for KL-based filtering."
            )
        # Clamp action defaults lower to 0.0 (consistent with old truncate behavior).
        if self.action == "clamp" and self.lower is None:
            self.lower = 0.0
        # Validate sequence-level aggregation.
        if self.level == "token" and self.agg != "mean":
            warnings.warn(
                f"agg='{self.agg}' is ignored when level='token'. "
                "Aggregation is only used for sequence-level filtering.",
                UserWarning,
                stacklevel=2,
            )


@dataclass
class PPOActorConfig(TrainEngineConfig):
    """Configuration for PPO actor model, a subclass of a TrainEngine."""

    # Core PPO/GRPO Parameters
    ppo_n_minibatches: int = field(
        default=4, metadata={"help": "Number of minibatches for each PPO update"}
    )
    eps_clip: float = field(
        default=0.2, metadata={"help": "Clipping factor for policy ratio"}
    )
    eps_clip_higher: float | None = field(
        default=None,
        metadata={
            "help": "Clipping factor (higher value) for policy ratio. Default is None. When eps_clip_higher is set (decoupled), eps_clip will be used as the lower value."
        },
    )
    c_clip: float | None = field(
        default=None,
        metadata={
            "help": "Dual clipping factor for policy ratio, must be > 1.0. None disables dual clipping."
        },
    )
    # M2PO
    m2_threshold: float | None = field(
        default=None, metadata={"help": "The second momentum threshold for M2PO."}
    )
    # Reward
    reward_norm: NormConfig | None = field(
        default=None,
        metadata={"help": "Normalization configuration for rewards"},
    )
    reward_scaling: float = field(
        default=1.0, metadata={"help": "Reward scaling factor"}
    )
    reward_bias: float = field(default=0.0, metadata={"help": "Reward bias"})
    reward_clip: float = field(
        default=20.0, metadata={"help": "Maximum absolute value for reward clipping"}
    )
    overlong_reward_penalty: bool = field(
        default=False,
        metadata={"help": "Penalty for overlong sequences. Used within DAPO."},
    )
    overlong_tokens: int | None = field(
        default=None,
        metadata={"help": "Number of tokens in the tail that will receive a penalty"},
    )
    overlong_penalty_factor: float | None = field(
        default=None,
        metadata={"help": "Penalty factor for tokens in the tail"},
    )
    mask_no_eos_with_zero: bool = field(
        default=False,
        metadata={
            "help": "Mask truncated generations (no EOS token) and exclude from training"
        },
    )

    # Advantage Estimation
    discount: float = field(
        default=1.0, metadata={"help": "Discount factor for future rewards"}
    )
    gae_lambda: float = field(
        default=1.0, metadata={"help": "Lambda parameter for GAE"}
    )
    adv_norm: NormConfig | None = field(
        default=None, metadata={"help": "Normalization configuration for advantages."}
    )

    # KL Control
    kl_ctl: float = field(default=0.1, metadata={"help": "KL divergence coefficient"})
    kl_estimator: str = field(
        default="k1",
        metadata={"help": "KL divergence estimator", "choices": ["k1", "k2", "k3"]},
    )

    # SAPO (Soft Adaptive Policy Optimization) - https://arxiv.org/abs/2511.20347
    use_sapo_loss: bool = field(
        default=False,
        metadata={"help": "Use SAPO loss (mutually exclusive with PPO clipping)"},
    )
    sapo_tau_pos: float = field(
        default=1.0,
        metadata={"help": "SAPO temperature for positive advantages"},
    )
    sapo_tau_neg: float = field(
        default=1.05,
        metadata={"help": "SAPO temperature for negative advantages"},
    )

    # Asynchronous RL
    recompute_logprob: bool = field(
        default=False,
        metadata={
            "help": "Recompute log probability and replace the log probability returned by inference."
        },
    )
    use_decoupled_loss: bool = field(
        default=False,
        metadata={
            "help": "Use the decoupled loss. Implicitly enables recompute_logprob."
        },
    )
    rejection_sampling: RejectionSamplingConfig | None = field(
        default=None,
        metadata={
            "help": "Rejection sampling configuration for filtering stale samples. "
            "None disables filtering (equivalent to old behave_imp_weight_mode='disabled'). "
            "Only effective when use_decoupled_loss=True."
        },
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": "Level at which to compute importance sampling ratios. 'token': per-token ratios (standard PPO). 'sequence': sequence-level geometric mean of per-token ratios (GSPO).",
            "choices": ["token", "sequence"],
        },
    )
    # Proximal Log-Probability Computation Method
    prox_logp_method: str = field(
        default=PROX_LOGP_METHOD_RECOMPUTE,
        metadata={
            "help": "Method for computing proximal policy log-probabilities in decoupled PPO. "
            "Only effective when use_decoupled_loss=True. Options: "
            "'recompute' (default): Standard decoupled PPO, recompute proximal policy via forward pass. "
            "'loglinear': Use log-linear interpolation to approximate proximal policy (skip forward pass). "
            "'metrics': Like 'recompute', but also compute approximation metrics for evaluation.",
            "choices": PROX_LOGP_METHODS_ALL,
        },
    )

    # Logging Agent Trajectories
    log_agent_stats: bool = field(
        default=False,
        metadata={"help": "Log statistics for agent trajectories"},
    )
    log_agent_stats_keys: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Keys for logging agent trajectory statistics"},
    )
    # Others
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens to generate"},
    )

    def should_compute_prox_logp(self) -> bool:
        """Determine if forward pass is needed for proximal log-probabilities.

        Returns:
            True if compute_logp() should be called, False to skip.
        """
        from areal.utils.constants import ProxLogpMethod

        method = ProxLogpMethod(self.prox_logp_method)
        return (self.use_decoupled_loss and not method.skips_forward_pass()) or (
            not self.use_decoupled_loss and self.recompute_logprob
        )

    def __post_init__(self):
        """Validate PPO actor configuration."""
        # Warn if rejection_sampling is configured but use_decoupled_loss is False
        if not self.use_decoupled_loss and self.rejection_sampling is not None:
            logger.warning(
                "rejection_sampling is configured but use_decoupled_loss=False. "
                "Filtering will be ignored. Set use_decoupled_loss=True to enable."
            )
        # Warn if decoupled loss is enabled but no rejection sampling configured.
        # The old default (behave_imp_weight_cap=5.0, mode=token_mask) enabled
        # filtering implicitly; the new default (rejection_sampling=None) disables
        # it. This warning helps users who relied on the old defaults.
        if self.use_decoupled_loss and self.rejection_sampling is None:
            logger.warning(
                "use_decoupled_loss=True with rejection_sampling=None: "
                "staleness filtering is disabled. If you previously relied on "
                "the default behave_imp_weight_cap=5.0 with token_mask mode, "
                "restore equivalent behavior with:\n"
                "  rejection_sampling:\n"
                "    level: token\n"
                "    action: mask\n"
                "    metric: ratio\n"
                "    upper: 5.0"
            )

        # Validate SAPO configuration
        if self.use_sapo_loss:
            if self.sapo_tau_pos <= 0 or self.sapo_tau_neg <= 0:
                raise ValueError(
                    f"SAPO temperatures (sapo_tau_pos, sapo_tau_neg) must be positive. "
                    f"Got sapo_tau_pos={self.sapo_tau_pos}, sapo_tau_neg={self.sapo_tau_neg}."
                )
            if self.use_decoupled_loss:
                raise ValueError(
                    "SAPO is not compatible with `use_decoupled_loss=True`. "
                    "Please set `actor.use_decoupled_loss=false` in your configuration."
                )

        super().__post_init__()


@dataclass
class PPOCriticConfig(TrainEngineConfig):
    """Configuration for PPO critic model, a subclass of a TrainEngine."""

    ppo_n_minibatches: int = field(
        default=4, metadata={"help": "Number of minibatches for each PPO update"}
    )
    eps_clip: float = field(
        default=0.5, metadata={"help": "Clipping factor for value loss"}
    )
    mask_no_eos_with_zero: bool = field(
        default=False,
        metadata={
            "help": "Mask truncated generations (no EOS token) and exclude from training"
        },
    )


def get_py_cmd(module: str, args: dict[str, Any]):
    # convert to flags
    cmd = ["python3", "-m", module]
    for k, v in args.items():
        if v is None or v is False or v == "" or (isinstance(v, list) and not v):
            continue
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            cmd.append(flag)
        elif isinstance(v, list):
            cmd.append(flag)
            cmd.extend(map(str, v))
        else:
            cmd.append(flag)
            cmd.append(str(v))
    return cmd


@dataclass
class vLLMConfig:
    """Configuration for vLLM runtime. Refer to:
    https://docs.vllm.ai/en/stable/api/index.html for detailed documentation.
    """

    model: str = ""
    seed: int = 1
    skip_tokenizer_init: bool = False
    enforce_eager: bool = False
    dtype: str = "bfloat16"
    distributed_executor_backend: str = "mp"
    # original
    max_num_seqs: int = 256
    # kv_cache_type: str = "auto"
    block_size: int = 16
    cpu_offload_gb: float = 0
    disable_sliding_window: bool = True
    max_model_len: int | None = 32768
    # NOTE: We use no_enable_* prefix (instead of enable_*) because get_py_cmd()
    # ignores parameters with False values. Setting enable_chunked_prefill=False
    # or enable_prefix_caching=False has NO effect - vLLM will use its default
    # values (True). Using no_enable_*=True correctly passes --no-enable-* flags
    # to vLLM, achieving enable_*=False behavior.
    #
    # IMPORTANT: vLLM V1 engine forces enable_chunked_prefill=True by default
    # for non-pooling tasks (generation tasks). And no_enable_chunked_prefill=True
    # has NO effect for generation tasks in vLLM v0.11.0.
    #
    no_enable_chunked_prefill: bool = False
    # NOTE: Disables prefix caching (vLLM default is enabled) because it will
    # make RL training corrupted in single controller mode.
    no_enable_prefix_caching: bool = True
    gpu_memory_utilization: float = 0.9
    worker_extension_cls: str = (
        "areal.engine.vllm_ext.vllm_worker_extension.VLLMWorkerExtension"
    )
    enable_sleep_mode: bool = False
    uvicorn_log_level: str = "warning"
    # GDN prefill backend for hybrid models like Qwen3.5; "triton" avoids the
    # FlashInfer GDN-kernel hang (vLLM #38916). None leaves vLLM's default, so
    # no flag is emitted and non-GDN models are unaffected.
    gdn_prefill_backend: str | None = field(
        default=None,
        metadata={
            "help": "GDN prefill backend for hybrid models like Qwen3.5.",
            "choices": ["triton", "flashinfer"],
        },
    )
    # lora
    enable_lora: bool = False
    max_lora_rank: int = 16  # vllm's default
    max_loras: int = 8  # override default
    lora_modules: list[str] | None = None  # lora_modules is automatically filled

    @staticmethod
    def build_args(
        vllm_config: "vLLMConfig",
        tp_size: int,
        pp_size: int,
        host: str | None = None,
        port: int | None = None,
        dist_init_addr: str | None = None,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):
        args: dict = conf_as_dict(vllm_config)
        args = dict(
            # Model and tokenizer
            tokenizer=vllm_config.model,
            load_format="auto",
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            **args,
        )
        if port is not None:
            args["port"] = port
        if host is not None:
            args["host"] = host
        # Multi-node support
        if n_nodes > 1:
            args["nnodes"] = n_nodes
            args["node_rank"] = node_rank
            if dist_init_addr is not None:
                from areal.utils.network import split_hostport

                master_host, master_port = split_hostport(dist_init_addr)
                args["master_addr"] = master_host
                args["master_port"] = str(master_port)
            if node_rank > 0:
                args["headless"] = True
        return args

    @staticmethod
    def build_cmd_from_args(args: dict[str, Any]):
        return get_py_cmd("areal.engine.vllm_ext.areal_vllm_server", args)

    @staticmethod
    def build_cmd(
        vllm_config: "vLLMConfig",
        tp_size: int,
        pp_size: int,
        host: str | None = None,
        port: int | None = None,
        dist_init_addr: str | None = None,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):
        args = vLLMConfig.build_args(
            vllm_config=vllm_config,
            tp_size=tp_size,
            pp_size=pp_size,
            host=host,
            port=port,
            dist_init_addr=dist_init_addr,
            n_nodes=n_nodes,
            node_rank=node_rank,
        )
        return vLLMConfig.build_cmd_from_args(args)


@dataclass
class SGLangConfig:
    """Configuration for SGLang runtime. Refer to:
    https://github.com/sgl-project/sglang for detailed documentation.
    """

    model_path: str = ""
    random_seed: int = 1
    skip_tokenizer_init: bool = False
    disable_cuda_graph: bool = False
    disable_radix_cache: bool = True
    disable_cuda_graph_padding: bool = False
    enable_nccl_nvls: bool = False
    disable_outlines_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    disable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_dp_attention: bool = False
    enable_ep_moe: bool = False
    enable_torch_compile: bool = False
    torch_compile_max_bs: int = 32
    cuda_graph_max_bs: int | None = None
    cuda_graph_bs: list[int] | None = None
    torchao_config: str = ""
    enable_nan_detection: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    num_continuous_decode_steps: int = 1
    enable_memory_saver: bool = False
    allow_auto_truncate: bool = False
    attention_backend: str | None = "fa3"
    enable_multimodal: bool = False
    sampling_backend: str | None = None
    context_length: int | None = 32768
    mem_fraction_static: float | None = 0.9
    max_running_requests: int | None = None
    # NOTE: chunked_prefill_size is by default 8192 on GPUs with 80GB mem in SGLang,
    # but we disable it to avoid precision issues
    chunked_prefill_size: int | None = -1
    max_prefill_tokens: int = 32768
    schedule_policy: str = "lpm"
    schedule_conservativeness: float = 1.0
    cpu_offload_gb: int = 0
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    dp_size: int = 1  # only used for dp attention
    ep_size: int = 1
    # lora
    enable_lora: bool | None = None
    max_lora_rank: int | None = None
    max_loaded_loras: int = 8  # override default
    lora_paths: list[str] | None = None  # lora_paths is automatically filled
    lora_backend: str = "triton"
    # logging
    log_level: str = "warning"
    log_level_http: str | None = "warning"
    log_requests: bool = False
    log_requests_level: int = 0
    show_time_cost: bool = False
    enable_metrics: bool = True  # Exports Prometheus-like metrics
    # The interval (in decoding iterations) to log throughput
    # and update prometheus metrics
    decode_log_interval: int = 1
    # Extra loader arguments
    # NOTE: These arguments will be parsed into a dict json-string
    # and passed as `model_loader_extra_config` to SGLang.
    enable_multithread_load: bool = False

    # Internal field, not exposed to users.
    enable_return_routed_experts: bool = False

    # Use staticmethod to make OmegaConf happy.
    @staticmethod
    def build_cmd(
        sglang_config: "SGLangConfig",
        tp_size,
        base_gpu_id,
        host: str | None = None,
        port: int | None = None,
        dist_init_addr: str | None = None,
        n_nodes: int = 1,
        node_rank: int = 0,
        pp_size: int = 1,
    ):
        args = SGLangConfig.build_args(
            sglang_config=sglang_config,
            tp_size=tp_size,
            base_gpu_id=base_gpu_id,
            host=host,
            port=port,
            dist_init_addr=dist_init_addr,
            n_nodes=n_nodes,
            node_rank=node_rank,
            pp_size=pp_size,
        )

        return SGLangConfig.build_cmd_from_args(args)

    @staticmethod
    def build_cmd_from_args(args: dict[str, Any]):
        return get_py_cmd(
            "areal.experimental.inference_service.sglang.launch_server", args
        )

    @staticmethod
    def build_args(
        sglang_config: "SGLangConfig",
        tp_size: int,
        base_gpu_id: int,
        host: str | None = None,
        port: int | None = None,
        dist_init_addr: str | None = None,
        n_nodes: int = 1,
        node_rank: int = 0,
        pp_size: int = 1,
    ):
        # Map "all-linear" to "all"
        args: dict = conf_as_dict(sglang_config)
        if sglang_config.enable_multithread_load:
            model_loader_extra_config = dict(
                enable_multithread_load=sglang_config.enable_multithread_load,
            )
            args["model_loader_extra_config"] = json.dumps(
                model_loader_extra_config, separators=(",", ":")
            )
        args.pop("enable_multithread_load", None)

        args = dict(
            # Model and tokenizer
            tokenizer_path=sglang_config.model_path,
            tokenizer_mode="auto",
            load_format="auto",
            trust_remote_code=True,
            is_embedding=False,
            # Other runtime options
            tp_size=tp_size,
            # Because we have set CUDA_VISIBLE_DEVICES to a single GPU in each process
            base_gpu_id=base_gpu_id,
            nnodes=n_nodes,
            node_rank=node_rank,
            # initialization addresses and ports
            dist_init_addr=dist_init_addr,
            **args,
        )
        if pp_size > 1:
            args["pp_size"] = pp_size
        if host is not None:
            args["host"] = host
        if port is not None:
            args["port"] = port
        if not pkg_version.is_version_greater_or_equal("sglang", "0.5.10.post1"):
            raise RuntimeError("Needs sglang>=0.5.10.post1 to run the code.")
        return args


@dataclass
class AgentConfig:
    """Configuration for agent workflows and the experimental agent service controller.

    Consolidates proxy settings (mode, parsers, export) with agent-service
    orchestration (scheduling, auth) into a single flat dataclass.
    """

    agent_cls_path: str = field(
        default="",
        metadata={
            "help": "Fully-qualified import path for the AgentRunnable implementation."
        },
    )
    admin_api_key: str = field(
        default="areal-admin-key",
        metadata={
            "help": (
                "Admin API key for the proxy server and agent-service inter-service auth. "
                "Used to authenticate management operations (grant_capacity, start_session). "
                "Cannot be used for chat completions. Each session gets a unique "
                "API key allocated via start_session. "
                "WARNING: Change this from the default for non-local deployments."
            ),
        },
    )
    scheduling_spec: tuple[SchedulingSpec, ...] = field(
        default_factory=lambda: (
            SchedulingSpec(
                gpu=0,
                cmd="python -m areal.experimental.agent_service.guard",
            ),
        ),
        metadata={
            "help": "Scheduling spec for agent-service guard workers. Must contain exactly one SchedulingSpec. Use scheduling_spec[0].env_vars for child-process environment variables."
        },
    )

    # -- Proxy / workflow settings (formerly OpenAIProxyConfig) ----------------
    mode: str = field(
        default="inline",
        metadata={
            "help": (
                "OpenAI proxy mode: 'inline' (in-process), 'subproc' (subprocess), "
                "or 'online' (external user sessions for online RL training). "
                "`inline` mode runs the provided agent workflow directly in the same process. "
                "`subproc` mode launches a separate process to run the agent. "
                "`online` mode waits for external users to complete sessions via "
                "the proxy gateway URL, enabling online RL training."
            ),
            "choices": ["inline", "subproc", "online"],
        },
    )
    tool_call_parser: str = field(
        default="qwen",
        metadata={"help": "Parser for tool calls in model output."},
    )
    reasoning_parser: str = field(
        default="qwen3",
        metadata={"help": "Parser for reasoning content (<think> tags)."},
    )
    chat_template_type: str = field(
        default="hf",
        metadata={
            "help": "Chat template type: 'hf' (standard) or 'concat' (multi-turn concatenation).",
            "choices": ["hf", "concat"],
        },
    )
    engine_max_tokens: int | None = field(
        default=None,
        metadata={"help": "Maximum total tokens for the engine (prompt + completion)."},
    )
    turn_discount: float = field(
        default=1.0,
        metadata={"help": "Discount factor for multi-turn reward propagation."},
    )
    export_style: str = field(
        default="individual",
        metadata={
            "help": "Export style: 'individual' (all interactions) or 'concat' (leaf nodes only). "
            "The 'individual' style exports each interaction (input-output-reward) step separately, "
            "and treats them as independent samples to train the model. "
            "The 'concat' style exports only the final concatenated trajectory from the root. "
            "It is only suitable for linear conversation histories without token mismatching (whether valid depends on the tokenizer).",
            "choices": ["individual", "concat"],
        },
    )
    subproc_max_workers: int = field(
        default=4,
        metadata={
            "help": "Maximum number of worker processes for subprocess mode execution pool."
        },
    )
    session_timeout_seconds: int = field(
        default=3600,
        metadata={
            "help": "Session timeout in seconds. Sessions inactive longer than this will be garbage collected."
        },
    )
    set_reward_finish_timeout: float = field(
        default=0.0,
        metadata={
            "help": "Timeout in seconds to wait for additional reward updates before finalizing a session."
        },
    )

    def __post_init__(self) -> None:
        if not self.agent_cls_path:
            raise ValueError("agent_cls_path must be a non-empty import path")
        if len(self.scheduling_spec) != 1:
            raise ValueError(
                f"scheduling_spec must contain exactly 1 SchedulingSpec, got {len(self.scheduling_spec)}"
            )
        if not self.admin_api_key or not self.admin_api_key.strip():
            raise ValueError("admin_api_key must not be empty or whitespace-only")
        if self.set_reward_finish_timeout < 0:
            raise ValueError(
                "set_reward_finish_timeout must be non-negative, "
                f"got {self.set_reward_finish_timeout}"
            )


@dataclass
class InferenceEngineConfig:
    """Configuration for inference servers, including offpolicyness control."""

    experiment_name: str | None = None
    trial_name: str | None = None
    fileroot: str | None = field(
        default=None,
        metadata={"help": "Root directory for logs and trajectory dumps."},
    )
    max_concurrent_rollouts: None | int = field(
        default=None,
        metadata={
            "help": "Maximum number of concurrent rollouts to "
            "the inference engine. Defaults to consumer_batch_size."
        },
    )
    queue_size: None | int = field(
        default=None,
        metadata={"help": "Input/Output queue size for async rollout."},
    )
    consumer_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for consuming rollouts from the queue."},
    )
    max_head_offpolicyness: int = field(
        default=0,
        metadata={
            "help": "Maximum off-policyness for the head. "
            "If the current version is more than this many versions behind, "
            "the request will not be accepted.",
        },
    )
    enable_rollout_tracing: bool = field(
        default=False,
        metadata={
            "help": "Whether to output verbose tracing messages for each generation request."
        },
    )
    check_trajectory_format: bool = field(
        default=False,
        metadata={
            "help": "Whether to check the format of produced trajectories of a customized workflow. Useful when debugging the workflow in isolation. Should be False during RL training."
        },
    )
    tokenizer_path: str = field(
        default="",
        metadata={"help": "Path to tokenizer for trajectory text decoding."},
    )
    dump_to_file: bool = field(
        default=False,
        metadata={"help": "Whether to dump the trajectories to files under fileroot."},
    )
    setup_timeout: float = field(
        default=300.0,
        metadata={
            "help": "Timeout in seconds of connecting to remote servers or launching local servers."
        },
    )
    workers_ready_timeout: float = field(
        default=30.0,
        metadata={
            "help": "Timeout (seconds) for initialize() to wait for guards to be ready."
        },
    )
    request_timeout: float = field(
        default=3600, metadata={"help": "Timeout for HTTP requests."}
    )
    request_retries: int = field(
        default=3, metadata={"help": "Number of retries for failed requests."}
    )
    pause_grace_period: float = field(
        default=0.0,
        metadata={
            "help": "The grace period after calling /pause_generation. Wait until all requests have been dropped."
        },
    )
    scheduling_spec: tuple[SchedulingSpec, ...] = field(
        default_factory=lambda: (
            SchedulingSpec(cmd="python -m areal.infra.rpc.rpc_server"),
        ),
        metadata={
            "help": "inference engine schedule specs. Can accept 1 or 2 SchedulingSpec: "
            "if 1 spec provided, it's used for both worker and engine, engine is embedded in the worker; "
            "if 2 specs provided, first one is for worker, second one is for engine. "
            "Currently only used by the RolloutController."
        },
    )
    # Backend and parallelism (new per-engine config)
    backend: str = field(
        default=MISSING,
        metadata={
            "help": "Backend and parallelism strategy. Must include an explicit backend prefix, "
            "e.g. 'sglang:d4', 'vllm:d2t4'. Required."
        },
    )
    scheduling_strategy: SchedulingStrategy = field(
        default_factory=SchedulingStrategy,
        metadata={
            "help": "The scheduling strategy of this InferenceEngine, either separation or colocation. "
            "Currently only used by the RolloutController."
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA. Should be same as actors LORA option."},
    )
    lora_name: str = field(
        default="",
        metadata={
            "help": "LoRA adapter name the rollout backend serves. Generation "
            "requests select the adapter by this name (plus the weight version). "
            "Usually left empty and auto-filled from gconfig.lora_name by "
            "PPOConfig.__post_init__ so load and request sides stay in sync."
        },
    )
    agent: AgentConfig = field(
        default_factory=lambda: AgentConfig(
            agent_cls_path="areal.experimental.openai.proxy.online_agent._OnlineAgent"
        ),
        metadata={
            "help": "Agent workflow configuration used by inference-service rollouts."
        },
    )
    return_routed_experts: bool = field(
        default=False,
        metadata={
            "help": "Return routed expert indices for MoE models. Effective only when using SGLang engine with MoE models."
        },
    )

    # v2 controller options
    _version: str = field(
        default="v1",
        metadata={
            "help": "Rollout controller implementation version. Use 'v1' for legacy RolloutController, 'v2' for RolloutControllerV2.",
            "choices": ["v1", "v2"],
        },
    )
    model: str = field(
        default="default",
        metadata={"help": "Model name exposed through the inference-service gateway."},
    )
    routing_strategy: str = field(
        default="round_robin",
        metadata={"help": "Routing strategy for the inference-service router."},
    )
    poll_interval: float = field(
        default=5.0,
        metadata={
            "help": "Health-poll interval in seconds for the inference-service router."
        },
    )
    admin_api_key: str = field(
        default="areal-admin-key",
        metadata={
            "help": "Admin API key used by the inference-service gateway, router, and data proxies."
        },
    )
    api_url: str | None = field(
        default=None,
        metadata={
            "help": "External OpenAI-compatible base URL for inference-service external model mode."
        },
    )
    provider_api_key: str | None = field(
        default=None,
        metadata={"help": "API key for the external OpenAI-compatible provider."},
    )

    def __post_init__(self):
        """Validate scheduling_spec length."""
        if len(self.scheduling_spec) not in (1, 2):
            raise ValueError(
                f"scheduling_spec must contain 1 or 2 SchedulingSpec, "
                f"got {len(self.scheduling_spec)}"
            )
        if self._version not in ("v1", "v2"):
            raise ValueError(
                f"_version must be either 'v1' or 'v2', got '{self._version}'"
            )
        if not self.admin_api_key or not self.admin_api_key.strip():
            raise ValueError("admin_api_key must not be empty or whitespace-only")
        if (
            self._version == "v2"
            and self.agent is not None
            and self.agent.admin_api_key != "areal-admin-key"
        ):
            logger.warning(
                "rollout.agent.admin_api_key is ignored by rollout controller v2; "
                "use rollout.admin_api_key instead."
            )


@dataclass
class _Timer:
    experiment_name: str = MISSING
    trial_name: str = MISSING
    fileroot: str = MISSING
    freq_epochs: int | None = field(
        default=None,
        metadata={
            "help": "Trigger frequency in epochs. None disables epoch-based saving."
        },
    )
    freq_steps: int | None = field(
        default=None,
        metadata={
            "help": "Trigger frequency in steps. None disables step-based saving."
        },
    )
    freq_secs: int | None = field(
        default=None,
        metadata={
            "help": "Trigger frequency in seconds. None disables time-based saving."
        },
    )


@dataclass
class EvaluatorConfig(_Timer):
    """Configuration for model evaluation scheduling and timing."""

    eval_before_train: bool = field(
        default=False,
        metadata={
            "help": "Run one evaluation before training begins, then continue with the configured evaluation frequency.",
        },
    )


@dataclass
class SaverConfig(_Timer):
    """Configuration for model checkpoint saving scheduling and timing."""

    mode: str = field(
        default="auto",
        metadata={
            "help": "Checkpoint save mode for HF saves. "
            "'auto': use async for Archon engine, sync for others (default). "
            "'sync': always synchronous. "
            "'async': always process-based async with pinned memory staging, "
            "extra CPU pinned memory "
            "proportional to per-rank model shard size "
            "(e.g., ~17.5GB/rank for 70B model on 8 GPUs). "
            "Non-Archon engines fall back to sync with a warning.",
            "choices": ["auto", "sync", "async"],
        },
    )

    def __post_init__(self):
        valid_modes = {"auto", "sync", "async"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Valid: {valid_modes}")


@dataclass
class RecoverConfig(_Timer):
    """Configuration for experiment recovery and fault tolerance."""

    mode: str = field(
        default="disabled",
        metadata={
            "help": "Recovery mode for the launcher. "
            "Options: "
            "'on' or 'auto': Automatically recover from previous runs if recover info and checkpoints are available. "
            "'off' or 'disabled': Never recover from previous runs."
        },
    )
    retries: int = field(
        default=3,
        metadata={"help": "Number of recovery retries when recovery is enabled."},
    )
    no_save_optim: bool = field(
        default=False,
        metadata={
            "help": "Do not save optimizer state in recovery checkpoints. "
            "Required when using use_distributed_optimizer with Megatron "
            "(flattened_range incompatibility)."
        },
    )
    no_load_optim: bool = field(
        default=False,
        metadata={
            "help": "Do not load optimizer state when recovering from checkpoint."
        },
    )

    def __post_init__(self):
        valid_modes = {"on", "off", "auto", "disabled"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid recover mode '{self.mode}'. "
                f"Valid options: {valid_modes}. "
                f"Note: 'fault' and 'resume' modes have been removed."
            )


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases experiment tracking."""

    mode: str = field(
        default="disabled",
        metadata={
            "help": "Tracking mode. One of 'online', 'offline', 'disabled', or 'shared'.",
            "choices": ["online", "offline", "disabled", "shared"],
        },
    )
    wandb_base_url: str = ""
    wandb_api_key: str = ""
    entity: str | None = None
    project: str | None = None
    name: str | None = None
    job_type: str | None = None
    group: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    config: dict | None = None
    id_suffix: str | None = "train"

    def __post_init__(self):
        """Validate WandB configuration."""
        valid_modes = ("online", "offline", "disabled", "shared")
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid wandb mode: '{self.mode}'. Must be one of: {', '.join(valid_modes)}."
            )


@dataclass
class SwanlabConfig:
    """Configuration for SwanLab experiment tracking and monitoring."""

    project: str | None = None
    name: str | None = None
    config: dict | None = None
    logdir: str | None = None
    mode: str = field(
        default="disabled",
        metadata={
            "help": "Tracking mode. One of 'cloud', 'local', 'disabled', or 'offline'.",
            "choices": ["cloud", "local", "disabled", "offline"],
        },
    )
    # set None to prevent info-leak in docs
    api_key: str | None = None

    def __post_init__(self):
        """Validate SwanLab configuration."""
        valid_modes = ("cloud", "local", "disabled", "offline")
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid swanlab mode: '{self.mode}'. Must be one of: {', '.join(valid_modes)}."
            )
        if self.api_key is None:
            self.api_key = os.getenv("SWANLAB_API_KEY")


@dataclass
class TensorBoardConfig:
    """Configuration for TensorBoard logging and visualization."""

    path: str | None = None


@dataclass
class TrackioConfig:
    """Configuration for Trackio experiment tracking (Hugging Face).

    Trackio is a lightweight, local-first experiment tracking library
    with a wandb-compatible API. Dashboards can be viewed locally or
    deployed to Hugging Face Spaces.

    See: https://github.com/gradio-app/trackio
    """

    mode: str = "disabled"
    """Tracking mode. One of "disabled", "online", or "local"."""
    project: str | None = None
    """Project name. Defaults to experiment_name if not set."""
    name: str | None = None
    """Run name. Defaults to trial_name if not set."""
    space_id: str | None = None
    """HF Space ID for remote dashboard deployment (e.g. "user/my-space").
    When set, metrics are also pushed to the specified Hugging Face Space."""

    def __post_init__(self):
        """Validate Trackio configuration."""
        valid_modes = {"disabled", "online", "local"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid trackio mode: '{self.mode}'. Must be one of {valid_modes}."
            )


@dataclass
class StatsLoggerConfig:
    """Configuration for experiment statistics logging and tracking services."""

    experiment_name: str = MISSING
    trial_name: str = MISSING
    fileroot: str = MISSING
    wandb: WandBConfig = field(
        default_factory=WandBConfig,
        metadata={"help": "Weights & Biases configuration."},
    )
    swanlab: SwanlabConfig = field(
        default_factory=SwanlabConfig,
        metadata={"help": "SwanLab configuration."},
    )
    tensorboard: TensorBoardConfig = field(
        default_factory=TensorBoardConfig,
        metadata={"help": "TensorBoard configuration. Only 'path' field required."},
    )
    trackio: TrackioConfig = field(
        default_factory=TrackioConfig,
        metadata={"help": "Trackio configuration (Hugging Face experiment tracking)."},
    )


@dataclass
class SessionTracerConfig:
    """Configuration for per-session lifecycle tracing."""

    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable per-session lifecycle tracing alongside perf events. "
                "When true, session metadata is captured to sessions.jsonl."
            )
        },
    )
    flush_threshold: int = field(
        default=256,
        metadata={
            "help": (
                "Flush session trace records once this many entries are ready. "
                "Values <= 0 fall back to 1."
            )
        },
    )


@dataclass
class MemoryProfilerConfig:
    """CUDA memory snapshot profiling configuration.

    Attributes:
        profile_steps: Steps at which to record memory snapshots.
        max_entries: Max entries for torch.cuda.memory._record_memory_history.
    """

    profile_steps: list[int] = field(
        default_factory=lambda: [0, 1],
        metadata={"help": "List of global steps to capture memory snapshots."},
    )
    max_entries: int = field(
        default=100000,
        metadata={"help": "Max entries for memory history ring buffer."},
    )


@dataclass
class PerfTracerConfig:
    """Configuration for perf tracer emission."""

    experiment_name: str = MISSING
    trial_name: str = MISSING
    fileroot: str = MISSING
    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Explicitly enable or disable perf tracing. Set to true to capture perf traces."
            )
        },
    )
    save_interval: int = field(
        default=1,
        metadata={
            "help": (
                "Flush trace events to disk every N calls to save(step=...). "
                "A value of 1 writes on every step; values <= 0 fall back to 1."
            )
        },
    )
    profile_steps: list[int] | None = field(
        default=None,
        metadata={
            "help": (
                "List of step numbers at which to capture detailed profiling traces. "
                "If None, no detailed profiling traces are captured."
            )
        },
    )
    session_tracer: SessionTracerConfig | None = field(
        default=None,
        metadata={"help": "Session tracing configuration."},
    )


@dataclass
class NameResolveConfig:
    """Configuration for distributed name resolution and service discovery."""

    type: str = field(
        default="nfs",
        metadata={
            "help": "Type of the distributed KV store for name resolving.",
            "choices": ["nfs", "etcd3", "ray"],
        },
    )
    nfs_record_root: str = field(
        default="/tmp/areal/name_resolve",
        metadata={
            "help": "Record root for NFS name resolving. Should be available on all nodes."
        },
    )
    etcd3_addr: str = field(
        default="localhost:2379", metadata={"help": "Address of the ETCD3 server."}
    )
    ray_actor_name: str = field(
        default="ray_kv_store",
        metadata={"help": "Name of the distributed Ray KV store."},
    )


@dataclass
class ClusterSpecConfig:
    """Configuration for cluster specification and distributed computing setup."""

    name_resolve: NameResolveConfig = field(
        default_factory=NameResolveConfig,
        metadata={"help": "Name resolving configuration."},
    )
    cluster_name: str = field(
        default="local",
        metadata={"help": "Name of the cluster. Used to set specific environs."},
    )
    fileroot: str = field(
        default="/tmp/areal/",
        metadata={
            "help": "Root for logs and checkpoints. Should be available on all nodes."
        },
    )
    n_nodes: int = field(
        default=32,
        metadata={
            "help": "The size of the cluster. Used to decide slurm hostname suffix."
        },
    )
    n_gpus_per_node: int = field(
        default=8,
        metadata={"help": "Number of GPUs per node (physical)."},
    )


@dataclass
class SchedulerConfig:
    """Configuration for worker scheduling. Used in the single-controller mode. Experimental."""

    type: str | None = field(default=None)
    endpoint: str = field(default="http://localhost:8081")
    deploy_mode: str = field(default="separation")
    functioncall_service_domain: str = field(default="http://localhost:8080")
    reward_functioncall_config: dict = field(default_factory=dict)
    reward_model_path: str = field(default="")
    reward_model_service_url: str = field(default="http://localhost:30000/classify")


@dataclass
class _DatasetConfig:
    """Configuration for dataset loading and preprocessing."""

    split: str = field(
        default="train",
        metadata={"help": "Dataset split to use, e.g., 'train', 'test'."},
    )
    path: str = field(
        default=MISSING,
        metadata={
            "help": "Path to the dataset. Can be a local path or a HuggingFace dataset name."
        },
    )
    type: str = field(
        default=MISSING,
        metadata={"help": "Type of training method, e.g., 'sft', 'rl', etc."},
    )
    batch_size: int = field(
        default=1, metadata={"help": "Batch size for the dataloader"}
    )
    shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the dataset"}
    )
    pin_memory: bool = field(
        default=False,
        metadata={
            "help": "Pin memory for faster data loading (set True for GPU training)"
        },
    )
    num_workers: int = field(
        default=0, metadata={"help": "Number of worker processes for data loading"}
    )
    num_dataset_workers: int = field(
        default=1,
        metadata={
            "help": "Number of remote data-service worker processes to launch when using scheduling_spec."
        },
    )
    drop_last: bool = field(
        default=True, metadata={"help": "Drop the last incomplete batch"}
    )
    max_length: int | None = field(
        default=None,
        metadata={
            "help": "Maximum token length of sequences in dataset. Longer sequences are filtered out."
        },
    )
    dataset_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Additional keyword arguments for dataset loading. "
            "These are passed to the dataset loading function `get_custom_dataset`."
        },
    )
    scheduling_spec: SchedulingSpec | None = field(
        default_factory=lambda: SchedulingSpec(
            cpu=1, gpu=0, mem=10, cmd="python3 -m areal.infra.rpc.guard"
        ),
        metadata={
            "help": "Scheduling spec for remote data loading workers. "
            "If set, dataset loading will be offloaded to a data service with remote workers."
        },
    )
    setup_timeout: float = field(
        default=120.0,
        metadata={
            "help": "Timeout in seconds for the data service to load and register a dataset. "
            "Increase this value when loading large datasets for the first time "
            "(e.g. HuggingFace datasets that require downloading and preprocessing)."
        },
    )


@dataclass
class TrainDatasetConfig(_DatasetConfig):
    """Configuration for training dataset loading and preprocessing."""


@dataclass
class ValidDatasetConfig(_DatasetConfig):
    """Configuration for validation dataset loading and preprocessing.

    It has different default values with `TrainDatasetConfig`.
    `shuffle` and `drop_last` default to False.
    """

    split: str = field(
        default="test",
        metadata={"help": "Dataset split to use, e.g., 'train', 'test'."},
    )
    shuffle: bool = field(
        default=False, metadata={"help": "Whether to shuffle the dataset"}
    )
    drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch"}
    )


@dataclass
class BaseExperimentConfig:
    """Base configuration class for all experiment types with common settings."""

    # NOTE: we need this unified config class because different experiments
    # have different config structures, e.g., GRPO has two engine configs,
    # but SFT only has a single one. We use subclasses to represent these structures.
    experiment_name: str = field(
        default=MISSING,
        metadata={"help": "Name of the experiment (no '_' or '/'). Required."},
    )
    trial_name: str = field(
        default=MISSING,
        metadata={"help": "Name of the trial (no '-' or '/'). Required."},
    )
    cluster: ClusterSpecConfig = field(
        default_factory=ClusterSpecConfig,
        metadata={"help": "Cluster specification. Mainly used by slurm."},
    )
    allocation_mode: str = field(
        default="",
        metadata={
            "help": "DEPRECATED: Use per-engine 'backend' fields instead (e.g., actor.backend, rollout.backend). "
            "Legacy pattern-based GPU parallel strategy allocation mode. "
            "Only used by SPMD launchers (local/ray/slurm). Manual migration to per-engine 'backend' fields is required.",
        },
    )
    seed: int = field(default=1, metadata={"help": "Random seed for reproducibility."})
    enable_offload: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable training offload using torch_memory_saver. "
            "This requires setting up the environment for TMS (e.g., via LD_PRELOAD)."
        },
    )
    total_train_epochs: int = field(
        default=1, metadata={"help": "Total number of epochs to train the model."}
    )
    total_train_steps: int | None = field(
        default=None,
        metadata={
            "help": "Terminate training after this number of steps. "
            "For benchmarking purposes only. None indicates normal training."
        },
    )
    total_train_n_seqs: int | None = field(
        default=None,
        metadata={
            "help": "Terminate training after consuming this number of samples. "
            "For benchmarking purposes only. None indicates normal training."
        },
    )
    tokenizer_path: str = field(
        default="",
        metadata={"help": "Path to the tokenizer."},
    )

    train_dataset: TrainDatasetConfig = field(default_factory=TrainDatasetConfig)
    valid_dataset: ValidDatasetConfig | None = field(default=None)

    saver: SaverConfig = field(default_factory=SaverConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    stats_logger: StatsLoggerConfig = field(default_factory=StatsLoggerConfig)
    perf_tracer: PerfTracerConfig | None = field(
        default=None,
        metadata={"help": "Performance tracer configuration. None means disabled."},
    )
    memory_profiler: MemoryProfilerConfig | None = field(
        default=None,
        metadata={
            "help": "Memory snapshot profiler configuration. None means disabled."
        },
    )
    recover: RecoverConfig = field(default_factory=RecoverConfig)

    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    vllm: vLLMConfig = field(default_factory=vLLMConfig)

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self):
        """Validate training configuration."""
        if self.total_train_epochs <= 0:
            raise ValueError(
                f"total_train_epochs must be positive, got {self.total_train_epochs}"
            )


@dataclass
class SFTConfig(BaseExperimentConfig):
    """Configuration for Supervised Fine-Tuning (SFT) experiments."""

    actor: TrainEngineConfig = field(default_factory=TrainEngineConfig)


@dataclass
class RWConfig(BaseExperimentConfig):
    """Configuration for Reward Model (RW) training experiments."""

    actor: TrainEngineConfig = field(default_factory=TrainEngineConfig)

    def __post_init__(self):
        super().__post_init__()
        if not getattr(self.actor, "is_critic", False):
            raise ValueError(
                "RWConfig requires actor.is_critic=True for reward modeling. "
                "Set 'actor.is_critic: true' in your YAML config."
            )


@dataclass
class DPOEngineConfig(TrainEngineConfig):
    """Engine configuration for DPO training, extending TrainEngineConfig with DPO-specific fields."""

    beta: float = field(
        default=0.1,
        metadata={"help": "KL penalty coefficient for DPO loss."},
    )

    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": "DPO loss variant. "
            "'sigmoid': original DPO loss (Rafailov et al. 2023). "
            "'ipo': Identity Preference Optimization with per-token length normalization (Azar et al. 2023).",
            "choices": ["sigmoid", "ipo"],
        },
    )

    def __post_init__(self):
        super().__post_init__()
        _valid = {"sigmoid", "ipo"}
        if self.loss_type not in _valid:
            raise ValueError(
                f"Unsupported DPO loss_type '{self.loss_type}'. "
                f"Must be one of {sorted(_valid)}."
            )


@dataclass
class DPOConfig(BaseExperimentConfig):
    """Configuration for Direct Preference Optimization (DPO) experiments."""

    actor: DPOEngineConfig = field(default_factory=DPOEngineConfig)

    ref: DPOEngineConfig = field(default_factory=DPOEngineConfig)

    def __post_init__(self):
        super().__post_init__()
        if getattr(self.actor, "is_critic", False):
            raise ValueError(
                "DPOConfig requires a language model (is_critic=False). "
                "Remove 'actor.is_critic: true' from your YAML config."
            )


@dataclass
class TeacherConfig:
    engine_type: str = field(
        default="rollout",
        metadata={
            "help": "Teacher engine type. 'rollout' uses inference engine scoring; "
            "'train' uses the legacy train-engine teacher path.",
            "choices": ["rollout", "train"],
        },
    )
    rollout: InferenceEngineConfig | None = field(default=None)
    train: PPOActorConfig | None = field(
        default=None,
        metadata={
            "help": "Legacy train-engine teacher config. Required when engine_type='train'."
        },
    )
    path: str = field(
        default="",
        metadata={
            "help": "Teacher model path. If set, overrides shared rollout backend model path."
        },
    )
    offload: bool = field(
        default=False,
        metadata={"help": "Whether to offload teacher rollout model between steps"},
    )
    rl_loss_weight: float = field(
        default=1.0,
        metadata={"help": "RL loss weight"},
    )

    distill_loss_weight: float = field(
        default=0.005,
        metadata={"help": "Distillation loss weight"},
    )

    def __post_init__(self):
        if self.rollout is not None and self.train is not None:
            warnings.warn(
                "Both teacher.rollout and teacher.train are configured; "
                f"teacher.engine_type={self.engine_type!r} selects which one is used.",
                stacklevel=2,
            )
        if self.engine_type == "rollout" and self.rollout is None:
            raise ValueError(
                "teacher.rollout must be provided when teacher.engine_type='rollout'."
            )
        if self.engine_type == "train" and self.train is None:
            raise ValueError(
                "teacher.train must be provided when teacher.engine_type='train'."
            )


@dataclass
class PPOConfig(BaseExperimentConfig):
    """Configuration for Proximal Policy Optimization (PPO) reinforcement learning experiments."""

    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    eval_gconfig: GenerationHyperparameters | None = field(
        default=None,
        metadata={
            "help": "Generation hyperparameters for evaluation. If None, use gconfig."
        },
    )
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    actor: PPOActorConfig = field(default_factory=PPOActorConfig)
    ref: PPOActorConfig | None = field(default=None)
    critic: PPOCriticConfig | None = field(default=None)
    teacher: TeacherConfig | None = field(
        default=None,
        metadata={
            "help": (
                "Optional teacher model configuration used for on-policy "
                "distillation during PPO training. If provided, the actor "
                "may be trained to match the teacher in addition to the "
                "standard PPO objective."
            )
        },
    )
    dynamic_bs: bool = field(
        default=False,
        metadata={
            "help": "Enable dynamic batch sizing in prepare_batch. When True, batch collection "
            "stops when (accepted + rejected) >= batch_size, returning only accepted results. "
            "This results in variable-sized batches of valid data."
        },
    )

    def __post_init__(self):
        """Validate the eval generation config."""
        if self.eval_gconfig is None:
            self.eval_gconfig = self.gconfig.new()
        # Propagate the LoRA adapter name to the rollout engine so the OpenAI-proxy
        # generation path requests the same adapter the trainer loads. The request
        # side (ArealOpenAI) cannot read gconfig.lora_name, so it must come from
        # the engine config. Single source of truth: gconfig.lora_name.
        if self.rollout.use_lora and not self.rollout.lora_name:
            self.rollout.lora_name = self.gconfig.lora_name
        super().__post_init__()


@dataclass
class GRPOConfig(PPOConfig):
    """A dummy place holder of GRPO config for backward compatibility."""

    pass


def parse_cli_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Path to the main configuration file", required=True
    )
    # The first argument might be the path to a training script,
    # which should be ignored by the argument parser.
    if argv and argv[0].endswith(".py"):
        argv = argv[1:]
    args, overrides = parser.parse_known_args(argv)
    # Initialize hydra config
    config_file = Path(args.config).absolute()
    assert config_file.exists(), f"Config file {config_file} does not exist."
    # hydra only recognize relative paths
    relpath = Path(os.path.relpath(str(config_file), Path(__file__).parent.absolute()))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra_init(config_path=str(relpath.parent), job_name="app", version_base=None)
    cfg = hydra_compose(
        config_name=str(relpath.name).split(".yaml")[0],
        overrides=overrides,
    )
    return cfg, config_file


_LEGACY_REJECTION_SAMPLING_KEYS = {
    "behave_imp_weight_cap",
    "behave_imp_weight_mode",
}

_LEGACY_MIGRATION_MESSAGE = (
    "Config keys 'behave_imp_weight_cap' and 'behave_imp_weight_mode' have been "
    "removed. Use 'rejection_sampling' sub-config instead.\n"
    "Migration mapping:\n"
    "  behave_imp_weight_mode='disabled'  -> rejection_sampling: null\n"
    "  behave_imp_weight_mode='token_mask', behave_imp_weight_cap=X\n"
    "    -> rejection_sampling: {level: token, action: mask, metric: ratio, upper: X}\n"
    "  behave_imp_weight_mode='token_truncate', behave_imp_weight_cap=X\n"
    "    -> rejection_sampling: {level: token, action: clamp, metric: ratio, upper: X}\n"
)


def _migrate_legacy_rejection_sampling(cfg: DictConfig) -> DictConfig:
    """Intercept removed behave_imp_weight_* keys and raise actionable error."""
    # Walk top-level and known nested actor/teacher configs for legacy keys.
    sections_to_check = ["actor", "teacher"]
    for section in sections_to_check:
        if not OmegaConf.is_missing(cfg, section) and section in cfg:
            sub = cfg[section]
            if sub is None or not isinstance(sub, DictConfig):
                continue
            found = _LEGACY_REJECTION_SAMPLING_KEYS.intersection(sub.keys())
            if found:
                raise ValueError(
                    f"Found removed config key(s) {found} under '{section}'. "
                    + _LEGACY_MIGRATION_MESSAGE
                )
    return cfg


def to_structured_cfg(cfg, config_cls):
    # Intercept legacy config keys before merge to give actionable error.
    _migrate_legacy_rejection_sampling(cfg)
    # Merge with the default configuration.
    # The yaml and commandline can omit some default values defined in python dataclasses.
    default_cfg = OmegaConf.structured(config_cls)
    cfg = OmegaConf.merge(default_cfg, cfg)
    return cfg


def load_expr_config(argv: list[str], config_cls: type[ConfigT]) -> tuple[ConfigT, str]:
    cfg, config_file = parse_cli_args(argv)
    cfg = to_structured_cfg(cfg, config_cls=config_cls)
    cfg = OmegaConf.to_object(cfg)
    assert isinstance(cfg, config_cls)

    # Setup environment
    name_resolve.reconfigure(cfg.cluster.name_resolve)

    from areal.utils.stats_logger import StatsLogger

    # Save configuration as yaml
    if os.getenv("RANK", "0") == "0":
        save_config(cfg, StatsLogger.get_log_path(cfg.stats_logger))

    return cfg, str(config_file)


def conf_as_dict(cfg):
    if isinstance(cfg, (OmegaConf, DictConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return asdict(cfg)


def save_config(cfg, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    config_save_path = os.path.join(log_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        config_dict: dict = asdict(cfg)
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
        )
