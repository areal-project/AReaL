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
EXPR_NAME = "test_fsdp_engine_nccl"
TRIAL_NAME = "trial_nccl"
MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)
GROUP_NAME = "test_nccl_group"


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
        ),
        tp_size=1,
        base_gpu_id=1,
        dist_init_addr=network.format_hostport(host, dist_port),
    )

    # Create engine instance for server management
    temp_config = InferenceEngineConfig(
        backend="sglang:d1",
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
    )
    server_manager = RemoteSGLangEngine(temp_config)

    try:
        # Launch server via engine API
        yield server_manager.launch_server(sglang_args)
    finally:
        # Cleanup using engine API
        server_manager.destroy()


# We have integration tests for this now. Skipping in CI.
@pytest.mark.slow
def test_fsdpengine_nccl_weight_update_to_remote(tmp_path_factory, sglang_server):
    # Set environment variables for torch distributed
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = network.gethostip()
    os.environ["MASTER_PORT"] = str(network.find_free_ports(1)[0])
    # required by sglang
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    # Initialize FSDPEngine
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

        # Initialize RemoteSGLangEngine
        config = InferenceEngineConfig(
            backend="sglang:d1", experiment_name=EXPR_NAME, trial_name=TRIAL_NAME
        )
        remote_engine = RemoteSGLangEngine(config)
        remote_engine.initialize(
            addr=network.format_hostport(sglang_server.host, sglang_server.port)
        )

        # Get WeightUpdateMeta
        meta = WeightUpdateMeta.from_fsdp_xccl(
            gen_allocation=ModelAllocation.from_str("sglang:d1"),
        )
        meta.nccl_group_name = GROUP_NAME

        engine.connect_engine(remote_engine, meta)

        # Broadcast weights
        engine.update_weights(meta)
        print("uploaded weights to remote engine", flush=True)
    finally:
        # Cleanup in reverse order
        if remote_engine is not None:
            remote_engine.destroy()
        engine.destroy()
        assert not dist.is_initialized()
