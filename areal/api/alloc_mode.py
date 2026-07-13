# SPDX-License-Identifier: Apache-2.0

import enum
import math
from dataclasses import dataclass, field
from typing import Literal

from lark import Lark, Transformer

from areal.api.cli_args import SchedulingStrategy, SchedulingStrategyType
from areal.utils import logging

logger = logging.getLogger("AllocMode")


class AllocationType(enum.Enum):
    """Backward Compatible: Type of resource allocation strategy."""

    COLOCATE = 0  # Shared resources between training and inference (including SFT/training-only)
    DECOUPLED_TRAIN = 1  # Separate resources for training and inference
    LLM_SERVER_ONLY = 2  # Inference-only allocation


class AllocationValidationError(Exception):
    """Raised when allocation mode validation fails."""


class InvalidAllocationModeError(Exception):
    """Legacy exception for backward compatibility with existing code."""


@dataclass
class ParallelStrategy:
    """5D parallel strategy supporting tensor, pipeline, data, context, and expert parallelism.

    This class represents a comprehensive parallelization strategy for distributed ML workloads,
    particularly designed for large language models and mixture-of-experts architectures.

    The five dimensions of parallelism are:
    - Tensor parallelism: Splits individual operations (like matrix multiplications) across devices
    - Pipeline parallelism: Splits model layers across devices in a pipeline fashion
    - Data parallelism: Replicates the model and splits data across devices
    - Context parallelism: Splits sequence length across devices (attention-specific)
    - Expert parallelism: Splits experts in MoE models across devices

    For implementation details, refer to:
    https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding

    Args:
        tensor_parallel_size: Number of devices for tensor model parallelism (default: 1)
        pipeline_parallel_size: Number of pipeline parallel stages (default: 1)
        data_parallel_size: Number of data parallel replicas for ZeRO optimization (default: 1)
        context_parallel_size: Number of devices for context parallelism in attention modules (default: 1)
        expert_parallel_size: Number of devices for expert parallelism in MoE models (default: 1)
        expert_tensor_parallel_size: Tensor parallelism size specifically for expert modules (default: 1)

    Note:
        - Context parallelism is only effective for attention modules
        - Expert parallelism is only effective for MoE (Mixture of Experts) modules
    """

    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Size of tensor-model parallelism"}
    )
    pipeline_parallel_size: int = field(
        default=1, metadata={"help": "Number of pipeline parallel stages"}
    )
    data_parallel_size: int = field(
        default=1, metadata={"help": "Data parallelism size for ZeRO optimization"}
    )
    context_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Context parallelism size for attention modules. "
            "Note that context parallelism is only effective for attention modules."
        },
    )
    expert_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Expert parallelism size for MoE models. "
            "Note that expert parallelism is only effective for expert modules."
        },
    )
    expert_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Tensor parallelism size for expert modules. "
            "By default, it is 1 which disables expert tensor parallelism."
        },
    )

    def __post_init__(self):
        """Initialize computed properties and validate configuration."""
        if self.expert_parallel_size > 1:
            # Calculate expert model parallel size for validation
            self.expert_model_parallel_size = (
                self.pipeline_parallel_size
                * self.expert_tensor_parallel_size
                * self.expert_parallel_size
            )

            # Validate that world size is divisible by expert model parallel size
            assert self.world_size % self.expert_model_parallel_size == 0, (
                f"Expert model parallel size {self.expert_model_parallel_size} "
                f"cannot divide world size {self.world_size}."
            )

    @property
    def expert_data_parallel_size(self) -> int:
        """Data parallelism size for expert modules in MoE models."""
        if not hasattr(self, "expert_model_parallel_size"):
            return self.data_parallel_size
        return self.world_size // self.expert_model_parallel_size

    # Abbreviated properties for convenience
    @property
    def tp_size(self) -> int:
        """Tensor parallelism size (abbreviated)."""
        return self.tensor_parallel_size

    @property
    def pp_size(self) -> int:
        """Pipeline parallelism size (abbreviated)."""
        return self.pipeline_parallel_size

    @property
    def dp_size(self) -> int:
        """Data parallelism size (abbreviated)."""
        return self.data_parallel_size

    @property
    def cp_size(self) -> int:
        """Context parallelism size (abbreviated)."""
        return self.context_parallel_size

    @property
    def ep_size(self) -> int:
        """Expert parallelism size (abbreviated)."""
        return self.expert_parallel_size

    @property
    def etp_size(self) -> int:
        """Expert tensor parallelism size (abbreviated)."""
        return self.expert_tensor_parallel_size

    @property
    def edp_size(self) -> int:
        """Expert data parallelism size (abbreviated)."""
        return self.expert_data_parallel_size

    @property
    def world_size(self) -> int:
        """Total number of devices required for this parallelization strategy."""
        return (
            self.data_parallel_size
            * self.context_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
        )

    def __str__(self):
        """String representation showing all non-default parallelism dimensions."""
        parts = [
            f"tp={self.tensor_parallel_size}",
            f"pp={self.pipeline_parallel_size}",
            f"dp={self.data_parallel_size}",
        ]

        if self.context_parallel_size > 1:
            parts.append(f"cp={self.context_parallel_size}")
        if self.expert_parallel_size > 1:
            parts.append(f"ep={self.expert_parallel_size}")
            if self.expert_tensor_parallel_size != 1:
                parts.append(f"ep_tp={self.expert_tensor_parallel_size}")

        return f"Parallel({','.join(parts)})"

    @staticmethod
    def parallelism_eq(this, other):
        """Compare two parallelism configurations for equality.

        Args:
            this: First ParallelStrategy to compare
            other: Second ParallelStrategy to compare

        Returns:
            bool: True if all parallelism dimensions match

        Note:
            Implemented as static method to avoid OmegaConf compatibility issues.
        """
        return (
            (this.tensor_parallel_size == other.tensor_parallel_size)
            and (this.pipeline_parallel_size == other.pipeline_parallel_size)
            and (this.data_parallel_size == other.data_parallel_size)
            and (this.context_parallel_size == other.context_parallel_size)
            and (this.expert_parallel_size == other.expert_parallel_size)
            and (this.expert_tensor_parallel_size == other.expert_tensor_parallel_size)
        )


@dataclass
class FSDPParallelStrategy(ParallelStrategy):
    """FSDP parallel strategy."""

    @staticmethod
    def parallelism_eq(this, other):
        """Compare FSDP parallelism configurations."""
        return ParallelStrategy.parallelism_eq(this, other)


@dataclass
class MegatronParallelStrategy(ParallelStrategy):
    """Megatron parallel strategy with additional sequence parallelism and virtual pipeline parallelism."""

    virtual_pipeline_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Virtual pipeline parallelism size for megatron modules "
            "for interleaved pipeline schedule. Default value is 1 (disabled)."
        },
    )
    use_sequence_parallel: bool = field(
        default=False,
        metadata={
            "help": "Enable sequence parallelism. Only used with tensor-model parallelism in Megatron",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        vpp = self.virtual_pipeline_parallel_size
        if vpp <= 1:
            self.virtual_pipeline_parallel_size = 1
        elif self.pipeline_parallel_size <= 1:
            raise AllocationValidationError(
                "Virtual pipeline parallelism requires pipeline_parallel_size > 1."
            )

    @staticmethod
    def parallelism_eq(this, other):
        """Compare Megatron parallelism configurations (excluding sequence parallelism)."""
        return ParallelStrategy.parallelism_eq(this, other) and (
            this.virtual_pipeline_parallel_size == other.virtual_pipeline_parallel_size
        )


@dataclass
class ModelAllocation:
    """Single model allocation with backend, name, parallel strategy, and scheduling.

    Parameters
    ----------
    backend : str
        Backend type ("sglang", "vllm", "fsdp", "megatron", "archon")
    name : str, optional
        Component name for referencing via allocation_mode[name]
    parallel : ParallelStrategy
        Parallelization strategy (tp, pp, dp, cp, ep sizes)
    scheduling_strategy : SchedulingStrategy
        Resource scheduling (separation or colocation)

    Examples
    --------
    >>> ModelAllocation("sglang", "rollout", ParallelStrategy(dp=2), SchedulingStrategy("separation"))
    """

    backend: Literal["fsdp", "megatron", "archon", "vllm", "sglang"]
    name: str | None
    parallel: ParallelStrategy
    scheduling_strategy: SchedulingStrategy

    def __post_init__(self):
        if self.backend == "fsdp":
            if (
                self.parallel.pipeline_parallel_size > 1
                or self.parallel.expert_parallel_size > 1
            ):
                raise AllocationValidationError(
                    f"FSDP backend only supports data/tensor/context parallelism. "
                    f"Got strategy: {self.parallel}"
                )

    @classmethod
    def from_str(
        cls,
        spec: str,
        name: str | None = None,
        scheduling_strategy: SchedulingStrategy | None = None,
    ) -> "ModelAllocation":
        """Parse a single backend:parallelism string into a ModelAllocation.

        Parameters
        ----------
        spec : str
            Single component spec like ``"fsdp:d4"``, ``"sglang:d4t2"``,
            or ``"megatron:(attn:d1p12t4|ffn:d1p12e4)"``.
            An explicit backend prefix is always required.
        name : str, optional
            Role name (e.g., ``"actor"``, ``"rollout"``).
        scheduling_strategy : SchedulingStrategy, optional
            Scheduling strategy. Defaults to separation.

        Returns
        -------
        ModelAllocation

        Raises
        ------
        ValueError
            If *spec* contains ``'+'`` (multi-component strings are not allowed).
        """
        if "+" in spec:
            raise ValueError(
                "ModelAllocation.from_str() accepts a single component spec. "
                "Multi-component strings containing '+' are not allowed. "
                "Use separate per-engine 'backend' fields instead "
                "(e.g., actor.backend='fsdp:d4', rollout.backend='sglang:d4')."
            )

        parser = _LLMParallelParser()
        result = parser.parse(spec)

        # Extract the single ModelAllocation from the parse result
        if isinstance(result, list):
            if len(result) != 1:
                raise ValueError(
                    f"Expected a single allocation from spec '{spec}', "
                    f"got {len(result)} allocations."
                )
            alloc = result[0]
        elif isinstance(result, ModelAllocation):
            alloc = result
        else:
            raise ValueError(f"Unexpected parse result type: {type(result)}")

        # Override name if provided
        if name is not None:
            alloc.name = name

        # Override scheduling_strategy if provided, otherwise default to separation
        if scheduling_strategy is not None:
            alloc.scheduling_strategy = scheduling_strategy
        else:
            alloc.scheduling_strategy = SchedulingStrategy(
                type=SchedulingStrategyType.separation, target=None
            )

        return alloc

    @property
    def world_size(self):
        if self.scheduling_strategy.type == SchedulingStrategyType.colocation.value:
            return 0
        return self.parallel.world_size

    def __str__(self):
        dims = []
        if self.parallel.data_parallel_size != 1:
            dims.append(f"d{self.parallel.data_parallel_size}")
        if self.parallel.pipeline_parallel_size != 1:
            dims.append(f"p{self.parallel.pipeline_parallel_size}")
        if self.parallel.tensor_parallel_size != 1:
            dims.append(f"t{self.parallel.tensor_parallel_size}")
        if self.parallel.context_parallel_size != 1:
            dims.append(f"c{self.parallel.context_parallel_size}")
        if self.parallel.expert_parallel_size != 1:
            dims.append(f"e{self.parallel.expert_parallel_size}")

        if not dims:  # Show at least data parallel if all dimensions are 1
            dims.append(f"d{self.parallel.data_parallel_size}")

        result = "".join(dims)
        if self.name:
            result = f"{self.backend}({self.name}):{result}"
        else:
            result = f"{self.backend}:{result}"
        return result


@dataclass
class _AllocationMode:
    """DEPRECATED — Legacy resource allocation configuration for SPMD launchers only.

    Use :class:`ModelAllocation` with per-engine ``backend`` fields instead.

    This class is retained only for backward compatibility with SPMD launchers
    (local, ray, slurm) and will be removed in a future version.

    Parameters
    ----------
    allocations : list[ModelAllocation]
        List of ModelAllocation objects, each representing a component

    Notes
    -----
    Access patterns:
        - allocation_mode[name]: Get allocation by name
        - allocation_mode.allocations: Get all allocations
        - allocation_mode.gen: Backward-compatible (single inference only)
        - allocation_mode.train: Backward-compatible (single training only)

    Examples
    --------
    Two named components:

    >>> mode = _AllocationMode.from_str("sglang[rollout]:d2+fsdp[actor]:d4")
    >>> rollout = mode["rollout"]

    Three components (names required):

    >>> mode = _AllocationMode.from_str("sglang[r]:d2+fsdp[a]:d4+fsdp[c]:d4")

    Colocation (actor and critic share 4 GPUs):

    >>> mode = _AllocationMode.from_str("sglang[r]:d2+fsdp[a]:d4|fsdp[c]:d4")
    """

    allocations: list[ModelAllocation] = field(default_factory=list)

    @classmethod
    def from_str(cls, allocation_mode: str):
        """Parse allocation mode string into _AllocationMode object.

        Parameters
        ----------
        allocation_mode : str
            String representation of allocation mode

        Returns
        -------
        _AllocationMode
            Parsed allocation configuration

        Raises
        ------
        AllocationValidationError
            When validation fails (duplicate names, missing names, etc.)
        ValueError
            When parsing fails

        Notes
        -----
        Syntax:
            - backend(name):dims - Named component
            - component+component - Disaggregation (separate GPUs)
            - component|component - Colocation (shared GPUs, names required)
            - Operator precedence: | binds tighter than +

        Examples
        --------
        Two components, no names:

        >>> _AllocationMode.from_str("sglang:d4t2+fsdp:d8")

        Two named components:

        >>> _AllocationMode.from_str("sglang[rollout]:d2+fsdp[actor]:d4")

        Three+ components (names required):

        >>> _AllocationMode.from_str("sglang[r]:d2+fsdp[a]:d4+fsdp[c]:d4")

        Colocation (r separated, a|c share GPUs):

        >>> _AllocationMode.from_str("sglang[r]:d2+fsdp[a]:d4|fsdp[c]:d4")
        """
        parser = _LLMParallelParser()
        result = parser.parse(allocation_mode)
        return parser._convert_to_allocation_mode(result)

    def __getitem__(self, name: str) -> ModelAllocation:
        """Get allocation by name."""
        for alloc in self.allocations:
            if alloc.name == name:
                return alloc
        raise KeyError(f"No allocation found with name: {name}")

    @property
    def world_size(self):
        return sum(alloc.world_size for alloc in self.allocations)

    def _get_inference_allocations(self) -> list[ModelAllocation]:
        """Get all inference allocations (sglang, vllm backends)."""
        return [a for a in self.allocations if a.backend in ("sglang", "vllm")]

    def _get_training_allocations(self) -> list[ModelAllocation]:
        """Get all training allocations (fsdp, megatron, archon backends)."""
        return [
            a for a in self.allocations if a.backend in ("fsdp", "megatron", "archon")
        ]

    ########### Legacy Attributes for Backward Compatibility ###########
    # DEPRECATED: AllocationMode and these properties are legacy. Use ModelAllocation directly.
    # These properties are only retained for SPMD launcher compatibility and will be removed.
    @property
    def type_(self) -> AllocationType:
        """DEPRECATED: Infer allocation type from allocations. Use ModelAllocation directly."""
        if len(self.allocations) not in [1, 2]:
            raise AttributeError(
                "Can only infer allocation type from 1 or 2 allocations."
            )

        if len(self.allocations) == 1:
            if self.allocations[0].backend in ("sglang", "vllm"):
                return AllocationType.LLM_SERVER_ONLY
            return AllocationType.COLOCATE

        inf_alloc = self._get_inference_allocations()
        train_alloc = self._get_training_allocations()
        if not (len(inf_alloc) == 1 and len(train_alloc) == 1):
            raise AttributeError(
                "Ambiguous allocation type: expected one inference and one training allocation."
            )
        if (
            inf_alloc[0].scheduling_strategy.type
            == SchedulingStrategyType.separation.value
            and train_alloc[0].scheduling_strategy.type
            == SchedulingStrategyType.separation.value
        ):
            return AllocationType.DECOUPLED_TRAIN
        return AllocationType.COLOCATE

    @property
    def gen(self) -> ParallelStrategy:
        """Backward compatible: returns parallel strategy for single inference allocation."""
        inf_allocs = self._get_inference_allocations()
        if len(inf_allocs) == 0:
            return None
        if len(inf_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'gen' property: found {len(inf_allocs)} inference allocations. "
                f"Use allocation_mode[name] or allocation_mode.allocations instead."
            )
        return inf_allocs[0].parallel

    @property
    def train(self) -> ParallelStrategy | None:
        """Backward compatible: returns parallel strategy for single training allocation."""
        train_allocs = self._get_training_allocations()
        if len(train_allocs) == 0:
            return None
        if len(train_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'train' property: found {len(train_allocs)} training allocations. "
                f"Use allocation_mode[name] or allocation_mode.allocations instead."
            )
        return train_allocs[0].parallel

    @property
    def gen_backend(self) -> str | None:
        """Backward compatible: returns backend for single inference allocation."""
        inf_allocs = self._get_inference_allocations()
        if len(inf_allocs) == 0:
            return None
        if len(inf_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'gen_backend' property: found {len(inf_allocs)} inference allocations. "
                f"Use allocation_mode[name].backend or allocation_mode.allocations instead."
            )
        return inf_allocs[0].backend

    @property
    def train_backend(self) -> str | None:
        """Backward compatible: returns backend for single training allocation."""
        train_allocs = self._get_training_allocations()
        if len(train_allocs) == 0:
            return None
        if len(train_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'train_backend' property: found {len(train_allocs)} training allocations. "
                f"Use allocation_mode[name].backend or allocation_mode.allocations instead."
            )
        return train_allocs[0].backend

    @property
    def gen_instance_size(self) -> int:
        """Backward compatible: returns instance size for single inference allocation."""
        inf_allocs = self._get_inference_allocations()
        if len(inf_allocs) == 0:
            raise AttributeError("No inference allocations found")
        if len(inf_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'gen_instance_size' property: found {len(inf_allocs)} inference allocations. "
                f"Use allocation_mode[name].parallel.tp_size * pp_size instead."
            )
        return inf_allocs[0].parallel.tp_size * inf_allocs[0].parallel.pp_size


# Grammar-based parser using Lark
# Operator precedence: | (colocation) binds tighter than + (disaggregation)
# Example: "a+b|c" parses as "a+(b|c)", not "(a+b)|c"
ALLOCATION_GRAMMAR = """
    start: expression

    expression: disaggregate_chain | component

    disaggregate_chain: component ("+" component)+
    component: colocate_expr | single_allocation
    single_allocation: inf_para | train_para
    colocate_expr: single_allocation ("|" single_allocation)+

    inf_para: modern_inf_para
    modern_inf_para: INFER_BACKEND ("[" NAME "]")? ":" inf_dim+
    train_para: train_backend_with_name | train_backend_hybrid | train_backend_only | train_name_only | train_dims_only | hybrid_moe_syntax
    train_backend_with_name: TRAIN_BACKEND "[" NAME "]" ":" common_dim+
    train_backend_hybrid: TRAIN_BACKEND ":" hybrid_moe_syntax
    train_backend_only: TRAIN_BACKEND ":" common_dim+
    train_name_only: "[" NAME "]" ":" common_dim+
    train_dims_only: common_dim+

    hybrid_moe_syntax: "("? attn_section "|" ffn_section ")"?
    attn_section: "attn" ":" attn_dim+
    ffn_section: "ffn" ":" ffn_dim+

    // Training parallelism strategy
    common_dim: DIM_TYPE NUMBER
    attn_dim: ATTN_DIM_TYPE NUMBER
    ffn_dim: FFN_DIM_TYPE NUMBER

    // Inference parallelism strategy
    inf_dim: INF_DIM_TYPE NUMBER

    DIM_TYPE: "p" | "d" | "t" | "c" | "e"
    ATTN_DIM_TYPE: "c" | "d" | "t" | "p"
    FFN_DIM_TYPE: "d" | "e" | "t" | "p"
    INF_DIM_TYPE: "d" | "t" | "p"

    INFER_BACKEND: "sglang" | "vllm"
    TRAIN_BACKEND: "fsdp" | "megatron" | "archon"

    NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: /[1-9][0-9]*/

    %import common.WS
    %ignore WS
"""


@dataclass
class ParallelDimension:
    """Single parallelism dimension with type and size.

    Used internally by the grammar parser to represent individual
    parallelism specifications before combining them into strategies.
    """

    type_: str  # Dimension type ("d", "t", "p", "c", "e")
    size: int  # Parallelism degree

    def __str__(self):
        return f"{self.type_}{self.size}"


@dataclass
class InferenceParallelism:
    """Backward Compatible: Inference parallelism configuration with backend and validation.

    Represents the parallelization strategy for inference workloads,
    including the specific backend (SGLang, vLLM) and associated
    validation rules.
    """

    backend: str
    strategy: ParallelStrategy

    def __str__(self):
        dims = []
        if self.strategy.data_parallel_size != 1:
            dims.append(f"d{self.strategy.data_parallel_size}")
        if self.strategy.tensor_parallel_size != 1:
            dims.append(f"t{self.strategy.tensor_parallel_size}")
        if self.strategy.pipeline_parallel_size != 1:
            dims.append(f"p{self.strategy.pipeline_parallel_size}")
        if not dims:  # Show at least data parallel if all dimensions are 1
            dims.append(f"d{self.strategy.data_parallel_size}")
        return f"{self.backend}:{''.join(dims)}"


class _ParallelStrategyTransformer(Transformer):
    """Lark transformer to convert parse tree to lists of ModelAllocation objects."""

    def __init__(self):
        super().__init__()
        self.seen_names = set()

    def _validate_name(self, name: str | None):
        """Validate and track component names for uniqueness."""
        if name is not None:
            if name in self.seen_names:
                raise AllocationValidationError(f"Duplicate component name: {name}")
            self.seen_names.add(name)

    def _build_model_allocation(
        self,
        backend: str,
        name: str | None,
        strategy: ParallelStrategy,
        scheduling: SchedulingStrategy,
    ) -> ModelAllocation:
        """Build ModelAllocation with validation."""
        self._validate_name(name)
        return ModelAllocation(
            backend=backend,
            name=name,
            parallel=strategy,
            scheduling_strategy=scheduling,
        )

    def start(self, items):
        return items[0]

    def expression(self, items):
        return items[0]

    def disaggregate_chain(self, items):
        """Handle multi-component disaggregation: comp1 + comp2 + comp3..."""
        all_allocations = []
        for item in items:
            if isinstance(item, list):
                all_allocations.extend(item)
            else:
                all_allocations.append(item)

        # Validate: 3+ components must all have names
        if len(all_allocations) >= 3:
            unnamed = [a for a in all_allocations if a.name is None]
            if unnamed:
                raise AllocationValidationError(
                    f"When using 3+ components, all must have names. "
                    f"Found {len(unnamed)} unnamed components."
                )

        return all_allocations

    def component(self, items):
        return items[0]

    def single_allocation(self, items):
        return items[0]

    def colocate_expr(self, items):
        """Handle colocation: comp1 | comp2 | comp3..."""
        allocations = []
        anchor_name = None

        for i, item in enumerate(items):
            if isinstance(item, list):
                # This shouldn't happen in practice
                allocations.extend(item)
            else:
                alloc = item
                if i == 0:
                    # First component is the anchor
                    anchor_name = alloc.name
                    alloc.scheduling_strategy = SchedulingStrategy(
                        type=SchedulingStrategyType.separation, target=None
                    )
                else:
                    # Rest colocate with anchor
                    if alloc.name is None:
                        raise AllocationValidationError(
                            "Components in colocation group must have names"
                        )
                    alloc.scheduling_strategy = SchedulingStrategy(
                        type=SchedulingStrategyType.colocation, target=anchor_name
                    )

                    # Validate world sizes match
                    if alloc.parallel.world_size != allocations[0].parallel.world_size:
                        raise AllocationValidationError(
                            f"Colocated components must have matching world sizes. "
                            f"'{anchor_name}' has {allocations[0].parallel.world_size}, "
                            f"'{alloc.name}' has {alloc.parallel.world_size}."
                        )
                allocations.append(alloc)

        return allocations

    def inf_para(self, items):
        return items[0]

    def modern_inf_para(self, items):
        backend = str(items[0])
        name = None
        dim_start_idx = 1

        # Check if name is provided
        if len(items) > 1 and isinstance(items[1], str):
            name = str(items[1])
            dim_start_idx = 2

        dimensions = items[dim_start_idx:]

        # Build ParallelStrategy
        strategy_kwargs = {}
        for dim in dimensions:
            if dim.type_ == "d":
                strategy_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                strategy_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                strategy_kwargs["pipeline_parallel_size"] = dim.size

        strategy = ParallelStrategy(**strategy_kwargs)
        return self._build_model_allocation(
            backend,
            name,
            strategy,
            SchedulingStrategy(type=SchedulingStrategyType.separation, target=None),
        )

    def train_para(self, items):
        """Pass through result from one of the train_* alternatives."""
        result = items[0]

        # If result is a ParallelStrategy (from hybrid_moe_syntax), it means
        # no backend prefix was given — reject it.
        if isinstance(result, ParallelStrategy):
            raise AllocationValidationError(
                "Backend must be explicitly specified for hybrid MoE parallelism. "
                "Use e.g. 'megatron:(attn:d1p12t4|ffn:d1p12e4)'. "
                "Auto-backend selection is no longer supported."
            )

        return result

    def train_backend_with_name(self, items):
        """Handle: TRAIN_BACKEND ( NAME ) : common_dim+"""
        backend = str(items[0])
        name = str(items[1])
        dims = items[2:]

        strategy_kwargs = {}
        for dim in dims:
            if dim.type_ == "d":
                strategy_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                strategy_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                strategy_kwargs["pipeline_parallel_size"] = dim.size
            elif dim.type_ == "c":
                strategy_kwargs["context_parallel_size"] = dim.size
            elif dim.type_ == "e":
                strategy_kwargs["expert_parallel_size"] = dim.size

        strategy = ParallelStrategy(**strategy_kwargs)
        return self._build_model_allocation(
            backend,
            name,
            strategy,
            SchedulingStrategy(type=SchedulingStrategyType.separation, target=None),
        )

    def train_backend_hybrid(self, items):
        """Handle: TRAIN_BACKEND : hybrid_moe_syntax"""
        backend = str(items[0])
        strategy = items[1]  # ParallelStrategy from hybrid_moe_syntax

        return self._build_model_allocation(
            backend,
            None,
            strategy,
            SchedulingStrategy(type=SchedulingStrategyType.separation, target=None),
        )

    def train_backend_only(self, items):
        """Handle: TRAIN_BACKEND : common_dim+"""
        backend = str(items[0])
        dims = items[1:]

        strategy_kwargs = {}
        for dim in dims:
            if dim.type_ == "d":
                strategy_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                strategy_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                strategy_kwargs["pipeline_parallel_size"] = dim.size
            elif dim.type_ == "c":
                strategy_kwargs["context_parallel_size"] = dim.size
            elif dim.type_ == "e":
                strategy_kwargs["expert_parallel_size"] = dim.size

        strategy = ParallelStrategy(**strategy_kwargs)
        return self._build_model_allocation(
            backend,
            None,
            strategy,
            SchedulingStrategy(type=SchedulingStrategyType.separation, target=None),
        )

    def train_name_only(self, items):
        """Handle: ( NAME ) : common_dim+"""
        raise AllocationValidationError(
            "Backend must be explicitly specified. "
            "Got a named component without a backend prefix. "
            "Use e.g. 'fsdp[actor]:d4', 'megatron[actor]:d4t2p2'. "
            "Auto-backend selection is no longer supported."
        )

    def train_dims_only(self, items):
        """Handle: common_dim+"""
        raise AllocationValidationError(
            "Backend must be explicitly specified. "
            "Got bare parallelism dimensions without a backend prefix. "
            "Use e.g. 'fsdp:d4', 'megatron:d4t2p2', 'sglang:d4'. "
            "Auto-backend selection is no longer supported."
        )

    def common_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def attn_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def ffn_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def inf_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def expert_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def attn_para(self, items):
        return items  # Return list of dimensions

    def expert_para(self, items):
        return items  # Return list of dimensions

    def hybrid_train_para(self, items):
        attn_dims = items[0]  # List of dimensions for attention modules
        expert_dims = items[1]  # List of dimensions for expert modules

        # Build attention strategy
        attn_kwargs = {
            "data_parallel_size": 1,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "context_parallel_size": 1,
        }
        for dim in attn_dims:
            if dim.type_ == "d":
                attn_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                attn_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                attn_kwargs["pipeline_parallel_size"] = dim.size
            elif dim.type_ == "c":
                attn_kwargs["context_parallel_size"] = dim.size

        # Build expert strategy parameters
        # - pp: inherits from attn if omitted, must match if specified
        # - dp: derived from world_size if omitted
        # - tp/ep: default to 1 if omitted
        expert_data_parallel_size = None
        expert_pipeline_parallel_size = None
        expert_tensor_parallel_size = 1
        expert_parallel_size = 1

        for dim in expert_dims:
            if dim.type_ == "d":
                expert_data_parallel_size = dim.size
            elif dim.type_ == "p":
                expert_pipeline_parallel_size = dim.size
            elif dim.type_ == "t":
                expert_tensor_parallel_size = dim.size
            elif dim.type_ == "e":
                expert_parallel_size = dim.size

        # expert PP: inherit from attn if omitted, validate match if specified
        if expert_pipeline_parallel_size is None:
            expert_pipeline_parallel_size = attn_kwargs["pipeline_parallel_size"]
        elif expert_pipeline_parallel_size != attn_kwargs["pipeline_parallel_size"]:
            raise AllocationValidationError(
                f"Pipeline parallel size for attention and FFN modules must be identical. "
                f"Got attention: {attn_kwargs['pipeline_parallel_size']}, FFN: {expert_pipeline_parallel_size}."
            )

        # Calculate attn world size
        attn_world_size = math.prod(
            [
                attn_kwargs["data_parallel_size"],
                attn_kwargs["tensor_parallel_size"],
                attn_kwargs["pipeline_parallel_size"],
                attn_kwargs["context_parallel_size"],
            ]
        )

        # expert DP: derive from world_size if omitted
        if expert_data_parallel_size is None:
            ffn_non_dp_size = (
                expert_parallel_size
                * expert_tensor_parallel_size
                * expert_pipeline_parallel_size
            )
            if attn_world_size % ffn_non_dp_size != 0:
                raise AllocationValidationError(
                    f"Cannot derive expert dp: attn world_size ({attn_world_size}) "
                    f"is not divisible by ffn ep*tp*pp ({ffn_non_dp_size})."
                )
            expert_data_parallel_size = attn_world_size // ffn_non_dp_size

        # Validate world sizes match
        expert_world_size = math.prod(
            [
                expert_data_parallel_size,
                expert_pipeline_parallel_size,
                expert_tensor_parallel_size,
                expert_parallel_size,
            ]
        )

        if attn_world_size != expert_world_size:
            raise InvalidAllocationModeError(
                f"World size for expert modules and attention modules must be identical. "
                f"Got attention: {attn_world_size}, expert: {expert_world_size}."
            )

        # Create final strategy combining both
        final_strategy_kwargs = attn_kwargs.copy()
        final_strategy_kwargs["expert_parallel_size"] = expert_parallel_size
        final_strategy_kwargs["expert_tensor_parallel_size"] = (
            expert_tensor_parallel_size
        )

        strategy = ParallelStrategy(**final_strategy_kwargs)
        return strategy  # Return ParallelStrategy, will be wrapped by train_para

    def hybrid_moe_syntax(self, items):
        # items should be [attn_section_result, ffn_section_result]
        attn_dims = items[0]
        ffn_dims = items[1]
        return self.hybrid_train_para([attn_dims, ffn_dims])

    def attn_section(self, items):
        # items will be the attn_dim+ results (ignoring "attn" and ":" literals)
        return items

    def ffn_section(self, items):
        # items will be the ffn_dim+ results (ignoring "ffn" and ":" literals)
        return items

    def DIM_TYPE(self, token):
        return str(token)

    def ATTN_DIM_TYPE(self, token):
        return str(token)

    def FFN_DIM_TYPE(self, token):
        return str(token)

    def EXPERT_DIM_TYPE(self, token):
        return str(token)

    def INF_DIM_TYPE(self, token):
        return str(token)

    def INFER_BACKEND(self, token):
        return str(token)

    def TRAIN_BACKEND(self, token):
        return str(token)

    def NUMBER(self, token):
        return int(token)

    def NAME(self, token):
        return str(token)


class _LLMParallelParser:
    """Internal LLM parallel strategy parser using Lark grammar.

    This parser handles the modern allocation mode syntax with explicit
    backend specifications, comprehensive validation, and support for
    complex allocation patterns including disaggregated and colocated
    configurations.
    """

    def __init__(self):
        self.parser = Lark(ALLOCATION_GRAMMAR, parser="earley", ambiguity="explicit")

    def parse(self, expression: str):
        try:
            tree = self.parser.parse(expression)
            transformer = _ParallelStrategyTransformer()
            result = transformer.transform(tree)
            return result
        except (AllocationValidationError, InvalidAllocationModeError):
            # Re-raise validation errors without modification
            raise
        except Exception as e:
            # Check for wrapped validation errors in lark VisitError
            import traceback

            tb = traceback.format_exception(type(e), e, e.__traceback__)
            tb_str = "".join(tb)

            if "AllocationValidationError" in tb_str:
                # Extract the validation error message
                lines = tb_str.split("\n")
                for line in lines:
                    if "AllocationValidationError:" in line:
                        msg = line.split("AllocationValidationError:")[-1].strip()
                        raise AllocationValidationError(msg)
                raise AllocationValidationError(str(e))
            elif "InvalidAllocationModeError" in tb_str:
                # Extract the invalid allocation error message
                lines = tb_str.split("\n")
                for line in lines:
                    if "InvalidAllocationModeError:" in line:
                        msg = line.split("InvalidAllocationModeError:")[-1].strip()
                        raise InvalidAllocationModeError(msg)
                raise InvalidAllocationModeError(str(e))

            err_hint = """
Hints:
1. The parsing logic requires colons instead of dots to separate backends from dimensions, e.g., use "sglang:d4+fsdp:d4" instead of "sglang.d4+fsdp.d4".
2. Check https://areal-project.github.io/AReaL/en/tutorial/megatron.html for allowed syntax and examples with complex MoE models.
"""
            raise ValueError(f"Parsing error: {e}\n{err_hint}")

    def _convert_to_allocation_mode(self, result):
        """Convert parsed result to _AllocationMode object.

        Args:
            result: Parsed result (list of ModelAllocation)

        Returns:
            _AllocationMode: Converted allocation mode configuration

        Raises:
            ValueError: When expression type is not recognized
        """
        if isinstance(result, list):
            # Main case: list of ModelAllocation objects
            return _AllocationMode(allocations=result)
        elif isinstance(result, ModelAllocation):
            # Single allocation
            return _AllocationMode(allocations=[result])
        else:
            raise ValueError(f"Unknown result type: {type(result)}")


# ---------------------------------------------------------------------------
# ``AllocationMode`` is removed.  Use :class:`ModelAllocation` with per-engine
# ``backend`` fields instead.  The internal ``_AllocationMode`` class is
# retained only for backward-compatible SPMD launcher parsing.
# ---------------------------------------------------------------------------

# Provide a clear error if someone still tries to import the old name.
# This avoids silent breakage.


def __getattr__(name):
    if name == "AllocationMode":
        raise AttributeError(
            "AllocationMode has been removed. Use ModelAllocation with per-engine "
            "'backend' fields instead (e.g., actor.backend='fsdp:d4', "
            "rollout.backend='sglang:d4'). "
            "See docs/en/reference/alloc_mode.md for migration details."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
