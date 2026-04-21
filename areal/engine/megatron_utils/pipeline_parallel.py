# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are adapted from NVIDIA Megatron-LM, licensed under the
# following terms:
#
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
import enum
import math

from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig

from areal.api import MegatronParallelStrategy
from areal.utils import logging

logger = logging.getLogger("MCore PipelineParallel")


class LayerType(enum.Enum):
    """Layer type
    embedding: embedding layer
    loss: loss layer
    encoder: encoder layer, not implemented yet, expect to be used in MLLM models
    decoder: decoder layer
    mtp: multi-token prediction layer, not implemented yet
    """

    embedding = 1
    loss = 2
    encoder = 3
    decoder = 4
    mtp = 5


class PipelineParallelLayerLayout:
    """Configuration of custom pipeline parallel layer partitioning."""

    def __repr__(self):
        return self.input_data

    def __init__(self, layout: str | list, pipeline_model_parallel_size: int):
        """Initialize PipelineParallelLayerLayout from a list or a str.
        Format validation will be done here.
        """

        self.input_data = layout
        if isinstance(layout, str):
            layout = PipelineParallelLayerLayout.parse_str_to_list(layout)
        else:
            layout = copy.deepcopy(layout)
        assert all(isinstance(row, list) for row in layout), (
            f"pipeline_model_parallel_layout must be a list of lists, but got"
            f" {[type(row) for row in layout]=}"
        )

        # Check PP size and get VPP size
        assert len(layout) % pipeline_model_parallel_size == 0, (
            f"pipeline_model_parallel_layout must be divisible"
            f" by pipeline_model_parallel_size ({len(layout)=},"
            f" {pipeline_model_parallel_size=})"
        )
        virtual_pipeline_model_parallel_size = (
            len(layout) // pipeline_model_parallel_size
        )

        # Convert 1D layout to 2D layout
        layout = [
            [
                layout[vpp_rank * pipeline_model_parallel_size + pp_rank]
                for vpp_rank in range(virtual_pipeline_model_parallel_size)
            ]
            for pp_rank in range(pipeline_model_parallel_size)
        ]

        # Convert all strings in pipeline_model_parallel_layout to LayerType
        for pp_rank in range(pipeline_model_parallel_size):
            for vpp_rank in range(virtual_pipeline_model_parallel_size):
                transferred_layout = []
                for layer_type in layout[pp_rank][vpp_rank]:
                    assert isinstance(layer_type, LayerType) or isinstance(
                        layer_type, str
                    ), (
                        f"elements in pipeline_model_parallel_layout must be LayerType or str,"
                        f" but got {type(layer_type)}."
                    )
                    if isinstance(layer_type, str):
                        layer_type = layer_type.strip().lower()
                        assert layer_type in LayerType.__members__, (
                            f"{layer_type} is not a valid LayerType"
                        )
                        layer_type = LayerType[layer_type]
                    transferred_layout.append(layer_type)
                layout[pp_rank][vpp_rank] = transferred_layout

        # Flatten the pipeline layout in layer id order.
        flatten_layout = []
        for vpp_rank in range(virtual_pipeline_model_parallel_size):
            for row in layout:
                flatten_layout.extend(row[vpp_rank])

        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self.layout = layout
        self.flatten_layout = flatten_layout


def configure_pipeline_layer_splits(
    parallel_strategy: MegatronParallelStrategy,
    hf_config: PretrainedConfig,
    tf_config: TransformerConfig,
) -> TransformerConfig:
    pp_size = parallel_strategy.pipeline_parallel_size
    vpp_size = parallel_strategy.virtual_pipeline_parallel_size

    if pp_size <= 1:
        return tf_config

    total_stages = pp_size * vpp_size
    total_layers = getattr(tf_config, "num_layers", None)
    if not isinstance(total_layers, int) or total_layers <= 0:
        return tf_config

    layer_param_weights, embedding_params, output_params = (
        estimate_stage_parameter_buckets(hf_config, tf_config)
    )
    if not layer_param_weights or all(weight <= 0 for weight in layer_param_weights):
        return tf_config

    if len(layer_param_weights) != total_layers:
        total_layers = min(total_layers, len(layer_param_weights))
        layer_param_weights = layer_param_weights[:total_layers]

    stage_lengths = _compute_stage_layer_lengths(
        layer_param_weights,
        embedding_params,
        output_params,
        total_stages,
    )
    if not stage_lengths:
        logger.warning(
            "Falling back to default pipeline layout; unable to find valid stage partition."
        )
        return tf_config

    layout: list[list[str]] = []
    for idx, length in enumerate(stage_lengths):
        stage_layers: list[str] = []
        if idx == 0:
            stage_layers.append("embedding")
        stage_layers.extend(["decoder"] * length)
        if idx == total_stages - 1:
            stage_layers.append("loss")
        layout.append(stage_layers)
    layout = PipelineParallelLayerLayout(
        layout=layout,
        pipeline_model_parallel_size=pp_size,
    )

    setattr(tf_config, "pipeline_model_parallel_layout", layout)
    if hasattr(tf_config, "num_layers_in_first_pipeline_stage"):
        setattr(tf_config, "num_layers_in_first_pipeline_stage", None)
    if hasattr(tf_config, "num_layers_in_last_pipeline_stage"):
        setattr(tf_config, "num_layers_in_last_pipeline_stage", None)
    if hasattr(tf_config, "account_for_embedding_in_pipeline_split"):
        setattr(tf_config, "account_for_embedding_in_pipeline_split", False)
    if hasattr(tf_config, "account_for_loss_in_pipeline_split"):
        setattr(tf_config, "account_for_loss_in_pipeline_split", False)

    stage_loads: list[float] = []
    cursor = 0
    for idx, length in enumerate(stage_lengths):
        load = 0.0
        if idx == 0:
            load += embedding_params
        stage_end = min(cursor + length, len(layer_param_weights))
        if stage_end > cursor:
            load += sum(layer_param_weights[cursor:stage_end])
        cursor = stage_end
        if idx == total_stages - 1:
            load += output_params
        stage_loads.append(load)

    logger.info(
        "Configured pipeline layout (per-stage decoder counts / params): %s / %s (pp=%s, vpp=%s)",
        stage_lengths,
        [f"{value / 1e6:.2f}M" for value in stage_loads],
        pp_size,
        vpp_size,
    )
    return tf_config


def _compute_stage_layer_lengths(
    layer_param_weights: list[float],
    embedding_params: float,
    output_params: float,
    pp_size: int,
) -> list[int]:
    total_layers = len(layer_param_weights)
    if total_layers == 0 or pp_size <= 0:
        return []

    prefix_sums = [0.0]
    for value in layer_param_weights:
        prefix_sums.append(prefix_sums[-1] + float(value))

    def segment_sum(start: int, end: int) -> float:
        return prefix_sums[end] - prefix_sums[start]

    stages = pp_size
    dp = [[math.inf] * (total_layers + 1) for _ in range(stages + 1)]
    choice = [[-1] * (total_layers + 1) for _ in range(stages + 1)]
    dp[0][0] = 0.0

    for stage in range(1, stages + 1):
        base = 0.0
        if stage == 1:
            base = embedding_params
        elif stage == stages:
            base = output_params

        for end in range(0, total_layers + 1):
            for start in range(0, end + 1):
                prev = dp[stage - 1][start]
                if not math.isfinite(prev):
                    continue
                if stage not in (1, stages) and start == end:
                    continue
                load = base + segment_sum(start, end)
                candidate = max(prev, load)
                if candidate < dp[stage][end]:
                    dp[stage][end] = candidate
                    choice[stage][end] = start

    if not math.isfinite(dp[stages][total_layers]):
        return []

    lengths = [0] * stages
    idx = total_layers
    for stage in range(stages, 0, -1):
        split = choice[stage][idx]
        if split < 0:
            return []
        lengths[stage - 1] = idx - split
        idx = split

    return lengths


def estimate_stage_parameter_buckets(
    hf_conf: PretrainedConfig, tf_conf: TransformerConfig
) -> tuple[list[float], float, float]:
    total_layers = getattr(tf_conf, "num_layers", None)
    if not isinstance(total_layers, int) or total_layers <= 0:
        total_layers = getattr(hf_conf, "num_hidden_layers", 0)
    if not isinstance(total_layers, int) or total_layers <= 0:
        return ([], 0.0, 0.0)

    hidden_size = getattr(hf_conf, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(tf_conf, "hidden_size", None)
    if hidden_size is None or hidden_size <= 0:
        return ([], 0.0, 0.0)

    intermediate_size = getattr(
        hf_conf,
        "intermediate_size",
        getattr(tf_conf, "ffn_hidden_size", hidden_size * 4),
    )
    num_heads = getattr(
        hf_conf, "num_attention_heads", getattr(tf_conf, "num_attention_heads", 1)
    )
    num_kv_heads = getattr(
        hf_conf,
        "num_key_value_heads",
        getattr(tf_conf, "num_query_groups", num_heads),
    )
    head_dim = getattr(hf_conf, "head_dim", None)
    if head_dim is None:
        kv_channels = getattr(tf_conf, "kv_channels", None)
        if kv_channels:
            head_dim = kv_channels
        else:
            head_dim = hidden_size // max(num_heads, 1)

    attention_bias = bool(
        getattr(hf_conf, "attention_bias", False)
        or getattr(tf_conf, "add_bias_linear", False)
    )
    add_bias_linear = bool(getattr(tf_conf, "add_bias_linear", False))
    gated_linear_unit = bool(getattr(tf_conf, "gated_linear_unit", False))

    def mlp_params(intermediate: int | None) -> float:
        if not intermediate or intermediate <= 0:
            return 0.0
        proj = hidden_size * intermediate
        gate = hidden_size * intermediate if gated_linear_unit else 0.0
        down = intermediate * hidden_size
        bias = 0.0
        if add_bias_linear:
            bias = (2 if gated_linear_unit else 1) * intermediate + hidden_size
        return float(proj + gate + down + bias)

    q_params = hidden_size * (num_heads * head_dim)
    kv_proj = num_kv_heads * head_dim
    k_params = hidden_size * kv_proj
    v_params = hidden_size * kv_proj
    o_params = hidden_size * hidden_size
    attn_bias_params = 0.0
    if attention_bias:
        attn_bias_params = (num_heads * head_dim) + (2 * kv_proj) + hidden_size

    attn_and_norm = (
        q_params
        + k_params
        + v_params
        + o_params
        + attn_bias_params
        + (2.0 * hidden_size)
    )
    dense_mlp = mlp_params(int(intermediate_size))
    dense_layer_params = float(attn_and_norm + dense_mlp)

    vocab_size = getattr(hf_conf, "vocab_size", 0)
    embedding_params = float(vocab_size * hidden_size)
    if getattr(hf_conf, "embedding_bias", False):
        embedding_params += float(vocab_size)

    output_params = float(vocab_size * hidden_size)
    if getattr(hf_conf, "tie_word_embeddings", False):
        output_params = embedding_params

    layer_weights: list[float] = [dense_layer_params] * total_layers

    num_moe_experts = getattr(tf_conf, "num_moe_experts", None)
    if num_moe_experts:
        moe_hidden_size = getattr(tf_conf, "moe_ffn_hidden_size", None)
        if moe_hidden_size is None:
            moe_hidden_size = getattr(hf_conf, "moe_intermediate_size", None)
        if moe_hidden_size is None:
            moe_hidden_size = intermediate_size

        expert_parallel_size = getattr(tf_conf, "expert_model_parallel_size", 1) or 1
        num_local_experts = math.ceil(num_moe_experts / max(expert_parallel_size, 1))

        router_params = hidden_size * num_moe_experts
        if getattr(tf_conf, "moe_router_enable_expert_bias", False):
            router_params += num_moe_experts

        expert_params = mlp_params(int(moe_hidden_size)) * num_local_experts

        shared_size = getattr(tf_conf, "moe_shared_expert_intermediate_size", None)
        shared_params = mlp_params(int(shared_size)) if shared_size else 0.0

        moe_layer_params = float(
            attn_and_norm + router_params + expert_params + shared_params
        )

        moe_layer_indices: set[int] = set()
        freq = getattr(tf_conf, "moe_layer_freq", 1)
        if isinstance(freq, int):
            step = abs(freq)
            assert step >= 1
            for idx in range(step - 1, total_layers, step):
                moe_layer_indices.add(idx)
        elif isinstance(freq, list) or isinstance(freq, tuple):
            assert len(freq) == total_layers
            for idx, is_moe in enumerate(freq):
                if is_moe:
                    moe_layer_indices.add(idx)

        for idx in moe_layer_indices:
            layer_weights[idx] = moe_layer_params

    return (layer_weights, embedding_params, output_params)
