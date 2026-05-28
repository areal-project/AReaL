# SPDX-License-Identifier: Apache-2.0

from areal.experimental.models.archon.model_spec import ModelSpec, register_model_spec
from areal.experimental.models.archon.pipeline_parallel import pipeline_llm
from areal.experimental.models.archon.qwen2.infra.parallelize import parallelize_qwen2
from areal.experimental.models.archon.qwen2.model.args import Qwen2ModelArgs
from areal.experimental.models.archon.qwen2.model.model import Qwen2Model
from areal.experimental.models.archon.qwen2.model.state_dict_adapter import (
    Qwen2StateDictAdapter,
)

QWEN2_SPEC = ModelSpec(
    name="Qwen2",
    model_class=Qwen2Model,
    model_args_class=Qwen2ModelArgs,
    state_dict_adapter_class=Qwen2StateDictAdapter,
    parallelize_fn=parallelize_qwen2,
    supported_model_types=frozenset({"qwen2"}),
    pipelining_fn=pipeline_llm,
)

# Auto-register when module is imported
register_model_spec(QWEN2_SPEC)

__all__ = ["QWEN2_SPEC"]
