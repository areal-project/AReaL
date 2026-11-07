# SPDX-License-Identifier: Apache-2.0
# Initial analytical primitives frozen from cfdbb716; owned independently here.

"""CPU-only model inputs, separate from the training/launcher config schema."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

from areal.elastic.contracts import ExecutionConfig as CandidateConfig


def positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ParallelLayout:
    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    etp: int = 1

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field.name} must be a positive integer")
        if self.world_size % (self.pp * self.ep * self.etp):
            raise ValueError("Expert layout does not divide attention world size")

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp

    @property
    def expert_dp(self) -> int:
        return self.world_size // (self.pp * self.ep * self.etp)

    @classmethod
    def from_allocation(cls, allocation: str) -> tuple[str, ParallelLayout]:
        """Read AReaL's canonical hybrid MoE notation without importing engines."""
        candidate = CandidateConfig(allocation, "sglang:d1")
        return "megatron", cls(*candidate.actor_parallel)


@dataclass(frozen=True)
class Hardware:
    """Per-GPU rates. Bandwidths are GB/s, one direction, not bidirectional sums.

    Inter-node bandwidth is an explicit scenario input, not inferred from GPU type.
    Named bundled presets are reference systems, not an assertion about a trial.
    """

    name: str
    bf16_tflops: float
    hbm_gbps: float
    intra_node_gbps: float
    inter_node_gbps: float
    gpus_per_node: int
    memory_gb: float
    intra_latency_us: float = 3
    inter_latency_us: float = 12

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Hardware name cannot be empty")
        for field in fields(self):
            if field.name != "name":
                positive(field.name, getattr(self, field.name))
        if type(self.gpus_per_node) is not int:
            raise ValueError("gpus_per_node must be an integer")


@dataclass(frozen=True)
class ModelShape:
    """Explicit dense-attention/MoE dimensions, supplied from the target model config."""

    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    experts: int
    top_k: int
    vocab_size: int
    tied_embeddings: bool = False
    mlp_type: str = "moe"

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name not in {"tied_embeddings", "mlp_type"}:
                value = getattr(self, field.name)
                if type(value) is not int or value < 1:
                    raise ValueError(f"{field.name} must be a positive integer")
        if self.top_k > self.experts or self.attention_heads % self.kv_heads:
            raise ValueError("Invalid top_k or grouped-query attention dimensions")
        if self.mlp_type not in {"dense", "moe"} or (
            self.mlp_type == "dense" and (self.experts, self.top_k) != (1, 1)
        ):
            raise ValueError("Dense MLP requires experts=top_k=1 and no router")

    @property
    def attention_parameters_per_layer(self) -> int:
        return (
            2
            * self.hidden_size
            * self.head_dim
            * (self.attention_heads + self.kv_heads)
        )

    @property
    def expert_parameters_per_layer(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size * self.experts

    @property
    def router_parameters_per_layer(self) -> int:
        return self.hidden_size * self.experts if self.mlp_type == "moe" else 0

    @property
    def head_parameters(self) -> int:
        return self.hidden_size * self.vocab_size

    @property
    def total_parameters(self) -> int:
        norms = 2 * self.hidden_size + 2 * self.head_dim
        return (
            self.layers
            * (
                self.attention_parameters_per_layer
                + self.expert_parameters_per_layer
                + self.router_parameters_per_layer
                + norms
            )
            + (1 if self.tied_embeddings else 2) * self.head_parameters
            + self.hidden_size
        )


@dataclass(frozen=True)
class ActorConfig:
    backend: str
    parallel: ParallelLayout
    max_tokens_per_mb: int = 32768
    n_mbs: int = 1
    n_mbs_divisor: int = 1
    granularity: int = 1
    packing_algorithm: str = "ffd"
    pad_to_maximum: bool = False
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = False
    checkpoint_recompute_fraction: float = 1.0
    checkpoint_note: str = "Full layer recomputation when enabled"
    expert_dispatcher: str = "alltoall"
    reshard_after_forward: bool = True
    reshard_head_after_forward: bool = False
    ppo_n_minibatches: int = 1
    recompute_logprob: bool = False
    use_decoupled_loss: bool = False
    prox_logp_method: str = "recompute"
    distributed_optimizer: bool = True
    grad_bytes: int = 4
    overlap_grad_reduce: bool = False
    compile: bool = False
    enable_tree_training: bool = False
    pipeline_layers: tuple[int, ...] | None = None
    optimizer_type: str = "adam"
    optimizer_dtype: str = "float32"
    optimizer_weight_decay: bool = True
    precision_aware_optimizer: bool = False
    main_grad_bytes: int = 4
    # None excludes master/remainder storage from the static lower bound.
    optimizer_master_storage_bytes: int | None = None
    enable_chunked_logits: bool = False
    lm_head_loss_chunk_size: int = 0
    entropy_requires_grad: bool = True

    def __post_init__(self) -> None:
        if self.pipeline_layers is not None:
            object.__setattr__(self, "pipeline_layers", tuple(self.pipeline_layers))
        if self.backend != "megatron":
            raise ValueError("Basic Actor cost model supports Megatron")
        if self.precision_aware_optimizer and self.backend != "megatron":
            raise ValueError("Precision-aware Adam component requires Megatron")
        if (
            type(self.lm_head_loss_chunk_size) is not int
            or self.lm_head_loss_chunk_size < 0
        ):
            raise ValueError("LM-head loss chunk size must be a nonnegative integer")
        if self.lm_head_loss_chunk_size > 0 and (
            not self.enable_chunked_logits or self.entropy_requires_grad
        ):
            raise ValueError(
                "LM-head chunking requires chunked logits without entropy gradients"
            )
        if self.main_grad_bytes not in {
            2,
            4,
        } or self.optimizer_master_storage_bytes not in {None, 2, 4}:
            raise ValueError("Unknown main-gradient or master/remainder storage width")
        if self.optimizer_type != "adam" or self.optimizer_dtype != "float32":
            raise ValueError("Megatron component requires FP32 Adam states")
        for name in (
            "pad_to_maximum",
            "gradient_checkpointing",
            "reshard_after_forward",
            "reshard_head_after_forward",
            "recompute_logprob",
            "use_decoupled_loss",
            "distributed_optimizer",
            "overlap_grad_reduce",
            "compile",
            "enable_tree_training",
            "optimizer_weight_decay",
            "precision_aware_optimizer",
            "enable_chunked_logits",
            "entropy_requires_grad",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.parallel.etp != 1:
            raise ValueError(
                "Expert tensor parallelism requires another compute component"
            )
        if self.expert_dispatcher not in {"alltoall", "hierarchical"}:
            raise ValueError("Supported expert_dispatcher: alltoall or hierarchical")
        if (
            not math.isfinite(self.checkpoint_recompute_fraction)
            or not 0 <= self.checkpoint_recompute_fraction <= 1
        ):
            raise ValueError(
                "checkpoint_recompute_fraction must be between zero and one"
            )
        for name in (
            "max_tokens_per_mb",
            "n_mbs",
            "n_mbs_divisor",
            "granularity",
            "ppo_n_minibatches",
            "grad_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dtype not in {"bfloat16", "bf16", "float16", "fp16"}:
            raise ValueError(
                "This calibration supports BF16/FP16 full-parameter training"
            )
        if self.packing_algorithm != "ffd" or self.enable_tree_training:
            raise ValueError(
                "FFD packed sequences are supported; tree/KK require another packing component"
            )
        if self.prox_logp_method not in {
            "recompute",
            "metrics",
            "loglinear",
            "reuse_train_logp",
        }:
            raise ValueError("Unknown AReaL proximal logprob method")
        if self.prox_logp_method == "reuse_train_logp" and self.ppo_n_minibatches != 1:
            raise ValueError("reuse_train_logp requires ppo_n_minibatches=1")
        if self.pipeline_layers is not None and (
            len(self.pipeline_layers) != self.parallel.pp
            or any(type(n) is not int or n < 1 for n in self.pipeline_layers)
        ):
            raise ValueError(
                "pipeline_layers must contain one positive layer count per PP stage"
            )

    @property
    def compute_prox_logp(self) -> bool:
        return (
            self.prox_logp_method in {"recompute", "metrics"}
            if self.use_decoupled_loss
            else self.recompute_logprob
        )

    @property
    def minimum_microbatches(self) -> int:
        pipeline_min = (
            2 * self.parallel.pp if self.backend == "megatron" else self.parallel.pp
        )
        return max(self.n_mbs, pipeline_min if self.parallel.pp > 1 else 1)

    @classmethod
    def from_areal(cls, data: dict[str, Any]) -> ActorConfig:
        """Accept the resolved experiment config or just its actor section.

        Interpolations must be resolved by AReaL/OmegaConf before passing a config.
        train_dataset.batch_size is deliberately NOT interpreted as actor n_seqs.
        """
        actor = data.get("actor", data)
        backend, parallel = ParallelLayout.from_allocation(actor["backend"])
        mb = actor.get("mb_spec", {})
        megatron = actor.get("megatron", {})
        if parallel.pp > 1 and megatron.get("virtual_pipeline_parallel_size", 1) != 1:
            raise ValueError("Evaluator supports non-interleaved 1F1B, without VPP")
        ddp = megatron.get("ddp", {})
        precision_aware = megatron.get("use_precision_aware_optimizer", False)
        if precision_aware and (
            megatron.get("main_grads_dtype", "float32") not in {"bfloat16", "float32"}
            or any(
                megatron.get(k, "float32") != "float32"
                for k in ("main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype")
            )
        ):
            raise ValueError("Precision-aware Adam requires FP32 master/moments")
        optimizer = actor.get("optimizer", {})
        if optimizer is None or optimizer.get("type", "adam") != "adam":
            raise ValueError("Megatron component requires Adam")
        if actor.get("use_lora", False) or megatron.get("fp8_config"):
            raise ValueError("LoRA/FP8 require another operator profile")
        if actor.get("enable_offload", data.get("enable_offload", False)):
            raise ValueError("CPU offload is outside this Megatron execution model")
        if megatron.get("enable_fp32_lm_head", False):
            raise ValueError("FP32 LM head needs another operator profile")
        if mb.get("max_tokens_per_mb") is None:
            raise ValueError("Supply mb_spec.max_tokens_per_mb")
        selective = megatron.get("recompute_granularity", "full") == "selective"
        return cls(
            backend,
            parallel,
            **{
                k: mb[k]
                for k in (
                    "max_tokens_per_mb",
                    "n_mbs",
                    "n_mbs_divisor",
                    "granularity",
                    "packing_algorithm",
                )
                if mb.get(k) is not None
            },
            **{
                k: actor[k]
                for k in (
                    "pad_to_maximum",
                    "dtype",
                    "gradient_checkpointing",
                    "ppo_n_minibatches",
                    "recompute_logprob",
                    "use_decoupled_loss",
                    "prox_logp_method",
                    "enable_tree_training",
                )
                if k in actor
            },
            precision_aware_optimizer=precision_aware,
            main_grad_bytes=2 if megatron.get("main_grads_dtype") == "bfloat16" else 4,
            enable_chunked_logits=megatron.get("enable_chunked_logits", False),
            lm_head_loss_chunk_size=megatron.get("lm_head_loss_chunk_size", 0),
            entropy_requires_grad=megatron.get("entropy_requires_grad", True),
            optimizer_weight_decay=optimizer.get("weight_decay", 0.0) != 0,
            distributed_optimizer=ddp.get("use_distributed_optimizer", True),
            overlap_grad_reduce=ddp.get("overlap_grad_reduce", False),
            grad_bytes=4 if ddp.get("grad_reduce_in_fp32", True) else 2,
            expert_dispatcher="hierarchical"
            if megatron.get("moe_enable_deepep", False)
            else "alltoall",
            checkpoint_recompute_fraction=0.25 if selective else 1.0,
            checkpoint_note="Selective recompute uses a 0.25 analytical prior"
            if selective
            else "Full layer recomputation when enabled",
            compile=bool(megatron.get("cuda_graph_impl")),
        )
