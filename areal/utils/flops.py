# SPDX-License-Identifier: Apache-2.0

"""Model FLOPs registry. Estimators return forward + backward FLOPs per sequence.

One multiply-add is two FLOPs; backward is estimated as twice forward.
Counts describe useful dense-equivalent math, excluding recomputation, optimizer,
normalization, softmax, activation functions, vision encoders and auxiliary MTP.
"""

from collections.abc import Callable
from typing import Any

FlopsEstimator = Callable[[int], float]
FlopsFactory = Callable[[Any], FlopsEstimator]
_FACTORIES: dict[str, FlopsFactory] = {}


def register_flops_estimator(model_type: str, factory: FlopsFactory) -> None:
    """Register a config -> (sequence length -> training FLOPs) factory.

    Call in each training worker before engine initialization. A factory can close
    over the actual model config, so local checkpoints and model variants work.
    Explicit registration replaces the existing estimator for this architecture.
    """
    if not model_type or not callable(factory):
        raise ValueError("A model_type and callable factory are required")
    _FACTORIES[model_type] = factory


def get_flops_estimator(config: Any) -> FlopsEstimator | None:
    """Resolve a registered architecture using the actual text model config."""
    config = getattr(config, "text_config", config)
    factory = _FACTORIES.get(getattr(config, "model_type", ""))
    return factory(config) if factory is not None else None


def _qwen_moe_factory(config: Any, *, hybrid: bool) -> FlopsEstimator:
    h = config.hidden_size
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    d = config.head_dim
    # Gate/up/down projections for selected experts and the shared expert.
    shared = getattr(config, "shared_expert_intermediate_size", 0) if hybrid else 0
    mlp = 6 * h * (config.num_experts_per_tok * config.moe_intermediate_size + shared)
    mlp += 2 * h * config.num_experts  # router projection
    if shared:
        mlp += 2 * h  # shared expert sigmoid gate projection
    # Qwen3.5's Q projection also produces the attention output gate.
    full_projection = 2 * h * d * ((3 if hybrid else 2) * heads + 2 * kv_heads)
    full_layers = layers
    linear = 0
    if hybrid:
        layer_types = getattr(config, "layer_types", None)
        if layer_types:
            if len(layer_types) != layers or any(
                t not in ("full_attention", "linear_attention") for t in layer_types
            ):
                raise ValueError("Invalid Qwen3.5 layer_types for FLOPs estimation")
            full_layers = layer_types.count("full_attention")
        else:
            full_layers = layers // getattr(config, "full_attention_interval", 4)
        k = config.linear_num_key_heads * config.linear_key_head_dim
        v = config.linear_num_value_heads * config.linear_value_head_dim
        # Q/K/V, z, a/b and output projections, plus depthwise convolution.
        linear = 2 * h * (2 * k + 3 * v + 2 * config.linear_num_value_heads)
        linear += 2 * (2 * k + v) * config.linear_conv_kernel_dim
        # Recurrent-equivalent delta rule: state prediction, rank-one update,
        # query readout (three key_dim x value_dim multiply-adds per value head).
        linear += (
            6
            * config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim
        )
    per_token = layers * mlp + full_layers * full_projection
    per_token += (layers - full_layers) * linear
    per_token += 2 * h * config.vocab_size  # LM head, including tied weights

    def estimate(sequence_length: int) -> float:
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length < 0
        ):
            raise ValueError("sequence_length must be a nonnegative integer")
        # Causal QK^T and AV: L(L+1)/2 attended pairs, two matmuls.
        attention = (
            full_layers * 2 * heads * d * sequence_length * (sequence_length + 1)
        )
        return float(3 * (per_token * sequence_length + attention))

    return estimate


def qwen3_moe_flops(config: Any) -> FlopsEstimator:
    """Qwen3 MoE, including Qwen/Qwen3-30B-A3B."""
    return _qwen_moe_factory(config, hybrid=False)


def qwen3_5_moe_flops(config: Any) -> FlopsEstimator:
    """Qwen3.5 hybrid MoE text backbone, including Qwen3.5-35B-A3B."""
    return _qwen_moe_factory(config, hybrid=True)


register_flops_estimator("qwen3_moe", qwen3_moe_flops)
register_flops_estimator("qwen3_5_moe", qwen3_5_moe_flops)
register_flops_estimator("qwen3_5_moe_text", qwen3_5_moe_flops)
