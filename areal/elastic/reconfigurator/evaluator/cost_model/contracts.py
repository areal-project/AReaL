# SPDX-License-Identifier: Apache-2.0

"""Component configuration, workload shapes, estimates and calibration records."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Literal

from areal.elastic.contracts import ExecutionConfig as CandidateConfig

Evidence = Literal["measured", "calibrated", "assumed", "unsupported"]
Feasibility = Literal["feasible", "infeasible", "unknown"]
Features = tuple[tuple[str, float], ...]


def nonnegative(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if positive and value == 0:
        raise ValueError(f"{name} must be positive")


def integer(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def evidence_of(values: tuple[Evidence, ...]) -> Evidence:
    order = ("measured", "calibrated", "assumed", "unsupported")
    return max(values, key=order.index, default="measured")


@dataclass(frozen=True)
class Estimate:
    seconds: float | None
    evidence: Evidence
    feasibility: Feasibility = "unknown"
    notes: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.evidence not in {"measured", "calibrated", "assumed", "unsupported"}:
            raise ValueError("Unknown estimate evidence")
        if self.feasibility not in {"feasible", "infeasible", "unknown"}:
            raise ValueError("Unknown feasibility")
        if (self.seconds is None) != (self.evidence == "unsupported"):
            raise ValueError("Unsupported estimates must have unknown duration")
        if self.seconds is not None:
            nonnegative("seconds", self.seconds)
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "source_ids", tuple(self.source_ids))

    def require_seconds(self) -> float:
        if self.seconds is None or self.feasibility == "infeasible":
            raise UnsupportedEstimate("; ".join(self.notes) or "Unavailable estimate")
        return self.seconds


class UnsupportedEstimate(ValueError):
    """A missing model or known infeasibility, never an implicit zero duration."""


@dataclass(frozen=True)
class TimedTask:
    name: str
    estimate: Estimate
    predecessors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task name is required")
        object.__setattr__(self, "predecessors", tuple(self.predecessors))


def critical_path(tasks: tuple[TimedTask, ...]) -> tuple[tuple[str, float, float], ...]:
    """Schedule a dependency DAG; resource serialization must be explicit edges."""
    pending = {task.name: task for task in tasks}
    if len(pending) != len(tasks):
        raise ValueError("Duplicate task name")
    if any(parent not in pending for t in tasks for parent in t.predecessors):
        raise ValueError("Unknown task dependency")
    end: dict[str, float] = {}
    spans = []
    while pending:
        ready = [
            task
            for task in pending.values()
            if all(p in end for p in task.predecessors)
        ]
        if not ready:
            raise ValueError("Cyclic task dependencies")
        for task in ready:
            start = max((end[p] for p in task.predecessors), default=0.0)
            end[task.name] = start + task.estimate.require_seconds()
            spans.append((task.name, start, end[task.name]))
            del pending[task.name]
    return tuple(spans)


@dataclass(frozen=True)
class CalibrationPoint:
    sample_id: str
    profile: str
    phase: str
    predicted_seconds: float
    measured_seconds: float
    features: Features
    timing_boundary: str = ""

    def __post_init__(self) -> None:
        if not all((self.sample_id, self.profile, self.phase)):
            raise ValueError(
                "Calibration requires sample, profile and phase identities"
            )
        nonnegative("predicted_seconds", self.predicted_seconds, positive=True)
        nonnegative("measured_seconds", self.measured_seconds, positive=True)
        object.__setattr__(self, "features", tuple(tuple(x) for x in self.features))
        if not self.features or len(dict(self.features)) != len(self.features):
            raise ValueError("Calibration features must have unique names")
        for name, value in self.features:
            nonnegative(name, value)


@dataclass(frozen=True)
class PhaseCalibration:
    version: str
    profile: str
    phase: str
    scale: float
    bounds: tuple[tuple[str, float, float], ...]
    sample_ids: tuple[str, ...]
    timing_boundary: str = ""

    def __post_init__(self) -> None:
        nonnegative("scale", self.scale, positive=True)
        if not all(
            (self.version, self.profile, self.phase, self.sample_ids, self.bounds)
        ):
            raise ValueError("Calibration needs version, scope, bounds and provenance")
        object.__setattr__(self, "bounds", tuple(tuple(b) for b in self.bounds))
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Duplicate calibration sample IDs")
        if len({b[0] for b in self.bounds}) != len(self.bounds):
            raise ValueError("Duplicate calibration feature bounds")
        for name, low, high in self.bounds:
            nonnegative(name, low)
            nonnegative(name, high)
            if high < low:
                raise ValueError("Reversed calibration bounds")

    def apply(
        self,
        seconds: float,
        profile: str,
        phase: str,
        features: Features,
        *,
        timing_boundary: str = "",
    ) -> Estimate:
        nonnegative("seconds", seconds)
        values = dict(features)
        violations = []
        if (profile, phase) != (self.profile, self.phase):
            violations.append("Calibration profile or phase mismatch")
        if timing_boundary != self.timing_boundary:
            violations.append("Calibration timing boundary mismatch")
        for name, low, high in self.bounds:
            if name not in values:
                violations.append(f"Missing calibration feature: {name}")
            elif not low <= values[name] <= high:
                violations.append(
                    f"Outside calibration bound: {name}={values[name]} not in [{low}, {high}]"
                )
        if violations:
            return Estimate(
                None, "unsupported", notes=tuple(violations), source_ids=self.sample_ids
            )
        return Estimate(seconds * self.scale, "calibrated", source_ids=self.sample_ids)


def apply_calibration(
    seconds: float,
    profile: str,
    phase: str,
    features: Features,
    calibration: tuple[PhaseCalibration, ...],
    *,
    timing_boundary: str = "",
) -> Estimate:
    matching = [c for c in calibration if (c.profile, c.phase) == (profile, phase)]
    if len(matching) > 1:
        raise ValueError("Ambiguous calibration for profile and phase")
    if not matching:
        return Estimate(
            seconds, "assumed", notes=("Analytical prior; no matching calibration",)
        )
    return matching[0].apply(
        seconds, profile, phase, features, timing_boundary=timing_boundary
    )


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

    def attention_parameters_per_rank(self, tp: int) -> float:
        # GQA repeats a KV head when TP is larger than the number of KV heads.
        return (
            2
            * self.hidden_size
            * self.head_dim
            * (self.attention_heads / tp + max(1, self.kv_heads / tp))
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
    optimizer_master_storage_bytes: int | None = None
    enable_chunked_logits: bool = False
    lm_head_loss_chunk_size: int = 0
    entropy_requires_grad: bool = True

    def __post_init__(self) -> None:
        if self.pipeline_layers is not None:
            object.__setattr__(self, "pipeline_layers", tuple(self.pipeline_layers))
        if self.backend != "megatron":
            raise ValueError("Actor cost model supports Megatron")
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
            or any(type(n) is not int or n < 0 for n in self.pipeline_layers)
            or any(n == 0 for n in self.pipeline_layers[1:-1])
        ):
            raise ValueError("Only first/last PP stages may have no decoder layers")

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


@dataclass(frozen=True)
class ServiceSlice:
    request_id: str
    context_tokens: int
    query_tokens: int
    kind: Literal["prefill", "decode"]
    emit_token: bool

    def __post_init__(self) -> None:
        if not self.request_id or self.kind not in {"prefill", "decode"}:
            raise ValueError("Request identity and service kind are required")
        integer("context_tokens", self.context_tokens)
        integer("query_tokens", self.query_tokens, minimum=1)
        if self.kind == "decode" and (self.query_tokens != 1 or not self.emit_token):
            raise ValueError("A decode slice emits exactly one token")
        if type(self.emit_token) is not bool:
            raise ValueError("emit_token must be boolean")


@dataclass(frozen=True)
class ServiceState:
    """Occupied token slots, including explicitly retained prefix-cache entries."""

    kv_tokens: int

    def __post_init__(self) -> None:
        integer("kv_tokens", self.kv_tokens)


@dataclass(frozen=True)
class RolloutConfig:
    replicas: int
    tp: int
    max_batch_sequences: int
    # Prefill cap per DP group; legacy mode caps total iteration queries instead.
    max_batch_tokens: int
    kv_capacity_tokens: int
    backend: str = "sglang"
    weight_bytes: int = 2
    kv_bytes: int = 2
    cache_mode: Literal["none", "prefix"] = "none"
    pp: int = 1
    # Attention DP groups inside one TP×PP instance, not external replicas.
    dp_size: int = 1
    ep_size: int = 1
    mem_fraction_static: float = 1.0
    # Fixed backend chunk size before internal DP division; None keeps legacy batching.
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
            raise ValueError(
                "This service model covers SGLang with explicit cache mode"
            )
        if self.weight_bytes != 2 or self.kv_bytes != 2:
            raise ValueError("Quantized weights/KV require a separate operator profile")
        if self.tp % self.dp_size or self.tp % self.ep_size:
            raise ValueError("SGLang attention DP and EP must divide TP")
        if self.max_batch_sequences < self.dp_size:
            raise ValueError(
                "SGLang needs at least one running slot per attention DP group"
            )
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
    def attention_tp(self):
        return self.tp // self.dp_size


@dataclass(frozen=True)
class InferenceRates:
    """Explicit effective/peak fractions; no company hardware calibration implied."""

    dense: float
    attention: float
    hbm: float
    network: float
    launch_seconds: float
    collectives_per_layer: int = 2

    def __post_init__(self) -> None:
        for name in ("dense", "attention", "hbm", "network"):
            value = getattr(self, name)
            nonnegative(name, value, positive=True)
            if value > 1:
                raise ValueError("Effective rate cannot exceed specified peak")
        nonnegative("launch_seconds", self.launch_seconds)
        integer("collectives_per_layer", self.collectives_per_layer)


@dataclass(frozen=True)
class ServiceEstimate:
    profile: str
    duration: Estimate
    features: Features
    kv_tokens_after: int


@dataclass(frozen=True)
class TransferTask:
    name: str
    predecessors: tuple[str, ...] = ()
    measured: Estimate | None = None
    transferred_bytes: float | None = None
    bandwidth_bytes_s: float | None = None
    setup_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "predecessors", tuple(self.predecessors))
        if not self.name:
            raise ValueError("Transfer task name is required")
        for name in ("transferred_bytes", "bandwidth_bytes_s", "setup_seconds"):
            value = getattr(self, name)
            if value is not None:
                nonnegative(name, value, positive=name == "bandwidth_bytes_s")
        if self.measured is not None and any(
            x is not None
            for x in (
                self.transferred_bytes,
                self.bandwidth_bytes_s,
                self.setup_seconds,
            )
        ):
            raise ValueError("Choose explicit duration or a transfer model, not both")

    def estimate(self) -> Estimate:
        if self.measured is not None:
            return self.measured
        if any(
            x is None
            for x in (
                self.transferred_bytes,
                self.bandwidth_bytes_s,
                self.setup_seconds,
            )
        ):
            return Estimate(
                None,
                "unsupported",
                notes=(f"Missing transfer measurements/parameters: {self.name}",),
            )
        return Estimate(
            self.setup_seconds + self.transferred_bytes / self.bandwidth_bytes_s,
            "assumed",
            notes=("Explicit latency plus bytes/bandwidth prior",),
        )


@dataclass(frozen=True)
class AllocationChange:
    task: str
    when: Literal["start", "end"]
    action: Literal["acquire", "release"]
    role: Literal["actor", "rollout"]
    devices: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", tuple(self.devices))
        if (
            self.when not in {"start", "end"}
            or self.action not in {"acquire", "release"}
            or self.role not in {"actor", "rollout"}
        ):
            raise ValueError("Invalid allocation change")
        if (
            not self.devices
            or len(set(self.devices)) != len(self.devices)
            or any(not d for d in self.devices)
        ):
            raise ValueError("Device identities must be nonempty and unique")


@dataclass(frozen=True)
class TransitionProfile:
    action: str
    source: str | None
    target: str | None
    tasks: tuple[TransferTask, ...]
    allocation_changes: tuple[AllocationChange, ...] = ()
    blocks: tuple[str, ...] = ("actor",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "allocation_changes", tuple(self.allocation_changes))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if not self.action or not self.tasks:
            raise ValueError("Even a zero-cost transition needs an explicit task")
        names = {t.name for t in self.tasks}
        if any(c.task not in names for c in self.allocation_changes):
            raise ValueError("Allocation event refers to an unknown transition task")
        if any(b not in {"actor", "generation", "admission"} for b in self.blocks):
            raise ValueError("Unknown transition blocking scope")
