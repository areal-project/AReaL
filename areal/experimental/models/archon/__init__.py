# SPDX-License-Identifier: Apache-2.0

# Import model spec modules to trigger registration
# Use direct module path to avoid triggering qwen2/qwen3 __init__.py first
from areal.experimental.models.archon.base import BaseStateDictAdapter
from areal.experimental.models.archon.expert_parallel import (
    ExpertParallel,
    ExpertTensorParallel,
    TensorParallel,
    apply_expert_parallel,
)
from areal.experimental.models.archon.model_spec import (
    ModelSpec,
    get_model_spec,
    get_supported_model_types,
    is_supported_model,
)
from areal.experimental.models.archon.moe_weight_converter import (
    MoEConversionState,
    MoEWeightConverter,
)
from areal.experimental.models.archon.parallel_dims import (
    ArchonParallelDims,
)
from areal.experimental.models.archon.pipeline_parallel import (
    generate_llm_fqn_per_model_part,
    pipeline_llm,
    pipeline_module_split,
)
from areal.experimental.models.archon.qwen2 import spec as qwen2_spec  # noqa: F401
from areal.experimental.models.archon.qwen3 import spec as qwen3_spec  # noqa: F401
from areal.experimental.models.archon.qwen3_5 import spec as qwen3_5_spec  # noqa: F401

__all__ = [
    "ArchonParallelDims",
    "BaseStateDictAdapter",
    "ExpertParallel",
    "ExpertTensorParallel",
    "MoEConversionState",
    "MoEWeightConverter",
    "ModelSpec",
    "TensorParallel",
    "apply_expert_parallel",
    "generate_llm_fqn_per_model_part",
    "get_model_spec",
    "get_supported_model_types",
    "is_supported_model",
    "pipeline_llm",
    "pipeline_module_split",
]
