# SPDX-License-Identifier: Apache-2.0

"""Evaluation configuration, hardware/model shapes and calibration contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

from .workload import parse_actor_parallel


def nonnegative(name: str, value: float, *, positive: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )


def positive(name: str, value: float) -> None:
    nonnegative(name, value, positive=True)


def integer(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ParameterCalibration:
    """Fitted operator parameters for one backend revision, across parallel layouts."""

    runtime_revision: str
    source_ids: tuple[str, ...]
    parameters: dict[str, float]

    def apply(self, defaults, runtime_revision: str):
        if self.runtime_revision != runtime_revision:
            raise ValueError(
                f"Calibration requires backend revision {self.runtime_revision!r}"
            )
        if not self.source_ids or not all(
            isinstance(s, str) and s.strip() for s in self.source_ids
        ):
            raise ValueError("Calibration requires measurement source IDs")
        allowed = {item.name for item in fields(defaults)} - {"collectives_per_layer"}
        if not self.parameters or not self.parameters.keys() <= allowed:
            raise ValueError(f"Expected fitted coefficients from {sorted(allowed)}")
        return replace(defaults, **self.parameters)


@dataclass(frozen=True)
class ParallelLayout:
    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    etp: int = 1

    def __post_init__(self) -> None:
        for item in fields(self):
            integer(item.name, getattr(self, item.name), minimum=1)
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
        return "megatron", cls(*parse_actor_parallel(allocation))


@dataclass(frozen=True)
class Hardware:
    """Per-GPU rates. Bandwidths are GB/s, one direction."""

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
        for item in fields(self):
            if item.name != "name":
                positive(item.name, getattr(self, item.name))
        integer("gpus_per_node", self.gpus_per_node, minimum=1)


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
    optimizer_master_storage_bytes: int | None = None
    enable_chunked_logits: bool = False
    lm_head_loss_chunk_size: int = 0
    entropy_requires_grad: bool = True
    logprobs_chunk_size: int = 1024

    def __post_init__(self) -> None:
        if self.pipeline_layers is not None:
            object.__setattr__(self, "pipeline_layers", tuple(self.pipeline_layers))
        if self.backend != "megatron":
            raise ValueError("Actor cost model supports Megatron")
        if self.packing_algorithm != "ffd" or self.enable_tree_training:
            raise ValueError(
                "Actor cost model supports FFD packing without tree training"
            )
        if self.optimizer_type != "adam" or self.optimizer_dtype != "float32":
            raise ValueError("Megatron component requires FP32 Adam states")
        if self.dtype not in {"bfloat16", "bf16", "float16", "fp16"}:
            raise ValueError("Actor cost model supports BF16/FP16 training")
        if self.expert_dispatcher not in {"alltoall", "hierarchical"}:
            raise ValueError("Supported expert_dispatcher: alltoall or hierarchical")
        if self.prox_logp_method not in {
            "recompute",
            "metrics",
            "loglinear",
            "reuse_train_logp",
        }:
            raise ValueError("Unknown AReaL proximal logprob method")
        for name in (
            "max_tokens_per_mb",
            "n_mbs",
            "n_mbs_divisor",
            "granularity",
            "ppo_n_minibatches",
            "grad_bytes",
            "logprobs_chunk_size",
        ):
            integer(name, getattr(self, name), minimum=1)
        integer("lm_head_loss_chunk_size", self.lm_head_loss_chunk_size)

    @property
    def compute_prox_logp(self) -> bool:
        return (
            self.prox_logp_method in {"recompute", "metrics"}
            if self.use_decoupled_loss
            else self.recompute_logprob
        )

    @property
    def minimum_microbatches(self) -> int:
        return max(self.n_mbs, 2 * self.parallel.pp if self.parallel.pp > 1 else 1)

    @classmethod
    def from_areal(cls, data: dict[str, Any]) -> ActorConfig:
        """Accept a resolved experiment config or its actor section."""
        actor = data.get("actor", data)
        backend, parallel = ParallelLayout.from_allocation(actor["backend"])
        mb = actor.get("mb_spec", {})
        megatron = actor.get("megatron", {})
        ddp = megatron.get("ddp", {})
        if parallel.pp > 1 and megatron.get("virtual_pipeline_parallel_size", 1) != 1:
            raise ValueError("Evaluator supports non-interleaved 1F1B, without VPP")
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
                    "logprobs_chunk_size",
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
            compile=bool(megatron.get("cuda_graph_impl")),
        )


@dataclass(frozen=True)
class ServiceSlice:
    request_id: str
    context_tokens: int
    query_tokens: int
    kind: Literal["prefill", "decode"]
    emit_token: bool

    @property
    def attention_pairs(self) -> float:
        queries = self.query_tokens
        return queries * self.context_tokens + queries * (queries + 1) / 2


@dataclass(frozen=True)
class RolloutConfig:
    replicas: int
    tp: int
    max_batch_sequences: int
    max_batch_tokens: int
    kv_capacity_tokens: int
    backend: str = "sglang"
    weight_bytes: int = 2
    kv_bytes: int = 2
    cache_mode: Literal["none", "prefix"] = "none"
    pp: int = 1
    dp_size: int = 1
    ep_size: int = 1
    mem_fraction_static: float = 1.0
    prefill_chunk_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "replicas",
            "tp",
            "max_batch_sequences",
            "max_batch_tokens",
            "kv_capacity_tokens",
            "weight_bytes",
            "kv_bytes",
            "pp",
            "dp_size",
            "ep_size",
        ):
            integer(name, getattr(self, name), minimum=1)
        if self.backend != "sglang" or self.cache_mode not in {"none", "prefix"}:
            raise ValueError("Rollout model covers SGLang with explicit cache mode")
        if self.weight_bytes != 2 or self.kv_bytes != 2:
            raise ValueError("Quantized weights/KV require a separate operator profile")
        if self.tp % self.dp_size or self.tp % self.ep_size:
            raise ValueError("SGLang attention DP and EP must divide TP")
        if self.max_batch_sequences < self.dp_size:
            raise ValueError("SGLang needs at least one slot per attention DP group")
        if self.prefill_chunk_tokens is not None:
            integer(
                "prefill_chunk_tokens", self.prefill_chunk_tokens, minimum=self.dp_size
            )
        if (
            not math.isfinite(self.mem_fraction_static)
            or not 0 < self.mem_fraction_static <= 1
        ):
            raise ValueError("Static memory fraction must be in (0, 1]")

    @property
    def world_size(self) -> int:
        return self.replicas * self.tp * self.pp

    @property
    def attention_tp(self) -> int:
        return self.tp // self.dp_size


@dataclass(frozen=True)
class InferenceRates:
    dense: float
    attention: float
    hbm: float
    network: float
    launch_seconds: float
    collectives_per_layer: int = 2

    def __post_init__(self) -> None:
        for name in ("dense", "attention", "hbm", "network"):
            value = getattr(self, name)
            positive(name, value)
            if value > 1:
                raise ValueError("Effective rate cannot exceed specified peak")
        nonnegative("launch_seconds", self.launch_seconds)
        integer("collectives_per_layer", self.collectives_per_layer)


@dataclass(frozen=True)
class ModelShape:
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
        for item in fields(self):
            if item.name not in {"tied_embeddings", "mlp_type"}:
                integer(item.name, getattr(self, item.name), minimum=1)
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

    def attention_parameters_per_rank(self, tp: int) -> float:
        return (
            2
            * self.hidden_size
            * self.head_dim
            * (self.attention_heads / tp + max(1, self.kv_heads / tp))
        )

    def attention_flops_per_pair(self, layers: int) -> int:
        return 4 * layers * self.attention_heads * self.head_dim

    @property
    def expert_parameters_per_layer(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size * self.experts

    @property
    def router_parameters_per_layer(self) -> int:
        return self.hidden_size * self.experts if self.mlp_type == "moe" else 0

    @property
    def active_expert_parameters_per_layer(self) -> float:
        return self.expert_parameters_per_layer * self.top_k / self.experts

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
