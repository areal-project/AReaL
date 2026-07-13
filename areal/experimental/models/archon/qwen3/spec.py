# SPDX-License-Identifier: Apache-2.0

from areal.experimental.models.archon.model_spec import ModelSpec, register_model_spec
from areal.experimental.models.archon.pipeline_parallel import pipeline_llm
from areal.experimental.models.archon.qwen3.infra.parallelize import parallelize_qwen3
from areal.experimental.models.archon.qwen3.model.args import Qwen3ModelArgs
from areal.experimental.models.archon.qwen3.model.model import Qwen3Model
from areal.experimental.models.archon.qwen3.model.state_dict_adapter import (
    Qwen3StateDictAdapter,
)

QWEN3_SPEC = ModelSpec(
    name="Qwen3",
    model_class=Qwen3Model,
    model_args_class=Qwen3ModelArgs,
    state_dict_adapter_class=Qwen3StateDictAdapter,
    parallelize_fn=parallelize_qwen3,
    supported_model_types=frozenset({"qwen3", "qwen3_moe"}),
    pipelining_fn=pipeline_llm,
)

# Auto-register when module is imported
register_model_spec(QWEN3_SPEC)

__all__ = ["QWEN3_SPEC"]
