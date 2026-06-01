# SPDX-License-Identifier: Apache-2.0

"""V1 xccl path: FSDP BF16 training -> SGLang FP8 weight sync integration test.

Verifies that FSDPEngine can broadcast FP8 block-wise quantized weights to a
RemoteSGLangEngine over the legacy xccl NCCL path without error.
"""

import os

import pytest
import torch.distributed as dist

from tests.utils import get_model_path

from areal.api import FinetuneSpec, ModelAllocation, WeightUpdateMeta
from areal.api.cli_args import (
    InferenceEngineConfig,
    OptimizerConfig,
    SGLangConfig,
    TrainEngineConfig,
)
from areal.engine import FSDPEngine, RemoteSGLangEngine
from areal.utils import network

pytestmark = pytest.mark.sglang
EXPR_NAME = "test_fsdp_fp8_weight_sync"
TRIAL_NAME = "trial_fp8"
MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)
GROUP_NAME = "test_fp8_weight_sync_group"


@pytest.fixture(scope="module")
def sglang_server():
    host = network.gethostip()
    dist_port = network.find_free_ports(1)[0]
    sglang_args = SGLangConfig.build_args(
        sglang_config=SGLangConfig(
            mem_fraction_static=0.2,
            model_path=MODEL_PATH,
            skip_tokenizer_init=False,
            log_level="info",
            quantization="fp8",
            quantization_config={"weight_block_size": [128, 128]},
        ),
        tp_size=1,
        base_gpu_id=1,
        dist_init_addr=network.format_hostport(host, dist_port),
    )

    temp_config = InferenceEngineConfig(
        backend="sglang:d1",
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
    )
    server_manager = RemoteSGLangEngine(temp_config)

    try:
        yield server_manager.launch_server(sglang_args)
    finally:
        server_manager.destroy()


@pytest.mark.slow
def test_fsdpengine_fp8_weight_update_to_remote(tmp_path_factory, sglang_server):
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = network.gethostip()
    os.environ["MASTER_PORT"] = str(network.find_free_ports(1)[0])
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    engine_config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
    )
    engine = FSDPEngine(engine_config)
    remote_engine = None
    try:
        engine.create_process_group()
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        engine.initialize(None, ft_spec)

        config = InferenceEngineConfig(
            backend="sglang:d1", experiment_name=EXPR_NAME, trial_name=TRIAL_NAME
        )
        remote_engine = RemoteSGLangEngine(config)
        remote_engine.initialize(
            addr=network.format_hostport(sglang_server.host, sglang_server.port)
        )

        meta = WeightUpdateMeta.from_fsdp_xccl(
            gen_allocation=ModelAllocation.from_str("sglang:d1"),
            quantization="fp8",
            quantization_config={"weight_block_size": [128, 128]},
        )
        meta.nccl_group_name = GROUP_NAME

        engine.connect_engine(remote_engine, meta)
        engine.update_weights(meta)
    finally:
        if remote_engine is not None:
            remote_engine.destroy()
        engine.destroy()
        assert not dist.is_initialized()
