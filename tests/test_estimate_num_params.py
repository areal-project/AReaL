import os

import mbridge
import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from transformers import AutoModelForCausalLM

from areal.engine.megatron_utils.pipeline_parallel import (
    estimate_stage_parameter_buckets,
)
from areal.models.mcore.registry import make_hf_and_mcore_config
from areal.utils.network import find_free_ports


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen3-1.7B",
        # "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B",
    ],
)
def test_estimate_num_params(model_name_or_path):
    try:
        # Dummy process group for mbridge initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_ports(1)[0])
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        mpu.initialize_model_parallel()
        tensor_parallel.model_parallel_cuda_manual_seed(0)

        # use local model if possible
        local_path = os.path.join(
            "/storage/openpsi/models", model_name_or_path.replace("/", "__")
        )
        if os.path.exists(local_path):
            model_name_or_path = local_path

        bridge = mbridge.AutoBridge.from_pretrained(model_name_or_path)
        hf_config, tf_config = make_hf_and_mcore_config(
            model_name_or_path, dtype=torch.bfloat16, bridge=bridge
        )
        layer_weights, embedding_params, output_params = (
            estimate_stage_parameter_buckets(hf_config, tf_config)
        )
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(hf_config)
        total_params = sum(p.numel() for p in model.parameters())
        estimated_params = sum(layer_weights) + embedding_params
        if not hf_config.tie_word_embeddings:
            estimated_params += output_params
        # Allow a small tolerance due to potential differences in counting methods
        rdiff = abs(total_params - estimated_params) / total_params
        assert rdiff < 0.05
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()
        assert not dist.is_initialized()
