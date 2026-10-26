# SPDX-License-Identifier: Apache-2.0
"""Build a small native Qwen4Exp graph and validate actual parameter mappings."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoConfig

from areal.models.mcore.qwen4_exp_megatron_bridge import (
    register_qwen4_exp_megatron_bridge,
)


def _shrink_provider(provider) -> None:
    """Use all Qwen4Exp layer types while keeping the allocation below 1 GB."""

    values = {
        "num_layers": 2,
        "hidden_size": 256,
        "ffn_hidden_size": 128,
        "num_attention_heads": 4,
        "num_query_groups": 2,
        "kv_channels": 64,
        "seq_length": 32,
        "max_position_embeddings": 32,
        "vocab_size": 256,
        "padded_vocab_size": 256,
        "make_vocab_size_divisible_by": 128,
        "should_pad_vocab": False,
        "num_moe_experts": 8,
        "moe_ffn_hidden_size": 128,
        "moe_router_topk": 2,
        "moe_shared_expert_intermediate_size": 128,
        "moe_layer_freq": [True, True],
        "linear_attention_freq": [True, False],
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "hc_count": 4,
        "hc_lowrank": 32,
        "ple_layer_ids": [],
        "ple_embed_dim": 256,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 64,
        "indexer_budget": 16,
        "indexer_compress_ratio": 2,
        "mrope_section": [3, 3, 2],
        "mtp_num_layers": 1,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "use_cpu_initialization": False,
        "gradient_accumulation_fusion": False,
        "apply_rope_fusion": False,
        "apply_rotary_pos_emb_fusion": False,
    }
    for name, value in values.items():
        setattr(provider, name, value)


def main() -> None:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(17)
    try:
        register_qwen4_exp_megatron_bridge()
        config = AutoConfig.from_pretrained(os.environ["MODEL_PATH"])
        auto_bridge = AutoBridge.from_hf_config(config)
        provider = auto_bridge.to_megatron_provider(load_weights=False)
        _shrink_provider(provider)
        provider.finalize()
        with torch.device("cuda"):
            model = provider.provide(pre_process=True, post_process=True)
        registry = auto_bridge._model_bridge.mapping_registry()

        parameter_names = sorted(name for name, _ in model.named_parameters())
        buffer_names = sorted(name for name, _ in model.named_buffers())
        state_names = set(model.state_dict())
        missing_parameters = [
            name
            for name in parameter_names
            if registry.megatron_to_hf_lookup(name) is None
        ]
        # TE state is not an HF model tensor. Rotary inv_freq is regenerated
        # from config and is non-persistent in the source model.
        ignored_buffers = [
            name
            for name in buffer_names
            if name.endswith("._extra_state")
            or name.endswith("rotary_pos_emb.inv_freq")
        ]
        missing_buffers = [
            name
            for name in buffer_names
            if name in state_names
            and name not in ignored_buffers
            and registry.megatron_to_hf_lookup(name) is None
        ]

        print("parameter_count", len(parameter_names))
        print("buffer_count", len(buffer_names))
        print(
            "mtp_parameter_count",
            sum("mtp" in name.split(".") for name in parameter_names),
        )
        print("missing_parameter_count", len(missing_parameters))
        print("missing_buffer_count", len(missing_buffers))
        if missing_parameters:
            print("missing_parameters", json.dumps(missing_parameters, indent=2))
        if missing_buffers:
            print("missing_buffers", json.dumps(missing_buffers, indent=2))
        print("all_parameter_names", json.dumps(parameter_names, indent=2))
        assert not missing_parameters
        assert not missing_buffers
        assert hasattr(model, "mtp") and len(model.mtp.layers) == 1
        assert any("in_proj_qkvz.weight" in name for name in parameter_names)
        assert any("in_proj_ba.weight" in name for name in parameter_names)
        assert any("indexer.index_qk_proj.weight" in name for name in parameter_names)
        assert any("attn_hyper_connection" in name for name in parameter_names)
        assert any("mlp_hyper_connection" in name for name in parameter_names)
        print("peak_allocated_bytes", torch.cuda.max_memory_allocated())
        print("model_build_smoke PASS")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
