# SPDX-License-Identifier: Apache-2.0

from areal.experimental.models.archon.model_spec import ModelSpec, register_model_spec
from areal.experimental.models.archon.pipeline_parallel import pipeline_llm
from areal.experimental.models.archon.qwen3_5.infra.parallelize import (
    parallelize_qwen3_5,
)
from areal.experimental.models.archon.qwen3_5.model.args import Qwen3_5ModelArgs
from areal.experimental.models.archon.qwen3_5.model.model import Qwen3_5Model
from areal.experimental.models.archon.qwen3_5.model.state_dict_adapter import (
    Qwen3_5StateDictAdapter,
)

QWEN3_5_SPEC = ModelSpec(
    name="Qwen3_5",
    model_class=Qwen3_5Model,
    model_args_class=Qwen3_5ModelArgs,
    state_dict_adapter_class=Qwen3_5StateDictAdapter,
    parallelize_fn=parallelize_qwen3_5,
    supported_model_types=frozenset(
        {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
    ),
    pipelining_fn=pipeline_llm,
)

# Auto-register when module is imported
register_model_spec(QWEN3_5_SPEC)

__all__ = ["QWEN3_5_SPEC"]
