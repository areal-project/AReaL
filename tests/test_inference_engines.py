"""Test suite for remote inference engines (vLLM and SGLang)."""

import os

import pytest
import torch.distributed as dist

from tests.utils import get_model_path

from areal.api import RolloutWorkflow, WeightUpdateMeta
from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
    vLLMConfig,
)
from areal.utils import network
from areal.utils.data import concat_padded_tensors, get_batch_size
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.pkg_version import is_available

MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)

IS_VLLM_INSTALLED = is_available("vllm")
IS_SGLANG_INSTALLED = is_available("sglang")


def _dummy_reward_fn(*args, **kwargs):
    """Dummy reward function for testing."""
    return 1.0


@pytest.fixture(
    params=[
        pytest.param({"backend": "vllm", "method": "init"}, marks=pytest.mark.vllm),
        pytest.param({"backend": "sglang", "method": "init"}, marks=pytest.mark.sglang),
        pytest.param(
            {"backend": "vllm", "method": "from_pretrained"},
            marks=pytest.mark.vllm_from_pretrained,
        ),
        pytest.param(
            {"backend": "sglang", "method": "from_pretrained"},
            marks=pytest.mark.sglang_from_pretrained,
        ),
    ],
    scope="module",
)
def inference_engine(request):
    """Fixture for remote inference engines only (vLLM and SGLang)."""
    backend = request.param["backend"]
    method = request.param["method"]

    # Skip if vLLM is not installed
    if backend == "vllm" and not IS_VLLM_INSTALLED:
        pytest.skip("vLLM is not installed")

    from areal.utils import seeding

    expr_name = f"test_remote_{backend}_engine"
    trial_name = "trial_0"

    seeding.set_random_seed(1, expr_name)

    dist_port = network.find_free_ports(1)[0]
    host = network.gethostip()

    # Configure SGLang (only when sglang is installed)
    sglang_args = None
    if IS_SGLANG_INSTALLED:
        sglang_config = SGLangConfig(
            skip_tokenizer_init=True,
            model_path=MODEL_PATH,
            mem_fraction_static=0.2,
            context_length=128,
        )
        sglang_args = SGLangConfig.build_args(
            sglang_config=sglang_config,
            tp_size=1,
            base_gpu_id=0,
            dist_init_addr=f"{host}:{dist_port}",
        )

    # Configure vLLM
    vllm_config = vLLMConfig(
        skip_tokenizer_init=False,
        model=MODEL_PATH,
        gpu_memory_utilization=0.2,
        max_model_len=128,
        enforce_eager=True,  # reduce launch overhead
    )
    vllm_args = vLLMConfig.build_args(
        vllm_config=vllm_config,
        tp_size=1,
        pp_size=1,
    )

    # Launch remote server and initialize engine
    if backend == "vllm":
        from areal.engine import RemotevLLMEngine

        engine_class = RemotevLLMEngine
        server_args = vllm_args
    else:  # sglang
        if not IS_SGLANG_INSTALLED:
            pytest.skip("SGLang is not installed")
        from areal.engine.sglang_remote import RemoteSGLangEngine

        engine_class = RemoteSGLangEngine
        server_args = sglang_args
    if method == "init":
        # Create engine instance for server management
        temp_config = InferenceEngineConfig(
            backend="sglang:d1",
            experiment_name=expr_name,
            trial_name=trial_name,
            setup_timeout=360,
        )
        server_manager = engine_class(temp_config)
    else:
        server_manager = engine_class.from_pretrained(
            tokenizer_path=MODEL_PATH,
            dp_size=1,
            experiment_name=expr_name,
            trial_name=trial_name,
            setup_timeout=360,
        )

    try:
        # Launch server via engine API
        server_info = server_manager.launch_server(server_args)

        # Set environment for remote engine
        os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{server_info.host}:{server_info.port}"

        yield {
            "engine_class": engine_class,
            "expr_name": expr_name,
            "trial_name": trial_name,
            "host": host,
            "port": server_info.port,
        }
    finally:
        # Cleanup using engine API
        server_manager.destroy()


# ============================================================================
# Unified Tests
# ============================================================================


@pytest.mark.parametrize("n_samples", [1, 2, 4])
@pytest.mark.slow
@pytest.mark.ci
def test_rollout(inference_engine, n_samples):
    """Test engine rollout with different sample sizes."""
    from areal.workflow import RLVRWorkflow

    config = InferenceEngineConfig(
        backend="sglang:d1",
        experiment_name=inference_engine["expr_name"],
        trial_name=inference_engine["trial_name"],
        max_concurrent_rollouts=2,
        consumer_batch_size=2,
        enable_rollout_tracing=True,
        setup_timeout=360,
        max_head_offpolicyness=int(1e10),
    )

    engine = inference_engine["engine_class"](config)
    engine.initialize()

    gconfig = GenerationHyperparameters(max_new_tokens=16, greedy=False)
    tokenizer = load_hf_tokenizer(MODEL_PATH)

    workflow = RLVRWorkflow(
        reward_fn=_dummy_reward_fn,
        gconfig=gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
    )

    data = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    result = engine.rollout_batch([data] * 2, workflow=workflow, group_size=n_samples)
    assert isinstance(result, list)
    concatenated = concat_padded_tensors(result)
    bs = get_batch_size(concatenated)
    assert bs == 2 * n_samples

    class NullWorkflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            return None

    # Test workflow returning None
    result = engine.rollout_batch(
        [data] * 2,
        workflow=NullWorkflow(),
    )
    assert result == []

    engine.destroy()
    assert not dist.is_initialized()


@pytest.mark.parametrize("ofp", [0, 1, 4, 16])
@pytest.mark.parametrize("bs", [2, 4])
@pytest.mark.parametrize("n_samples", [2, 1])
@pytest.mark.slow
@pytest.mark.ci
def test_staleness_control(inference_engine, bs, ofp, n_samples):
    """Test engine staleness control mechanism."""
    from areal.workflow import RLVRWorkflow

    config = InferenceEngineConfig(
        backend="sglang:d1",
        experiment_name=inference_engine["expr_name"],
        trial_name=inference_engine["trial_name"],
        consumer_batch_size=bs,
        max_head_offpolicyness=ofp,
        enable_rollout_tracing=True,
        setup_timeout=360,
    )

    engine = inference_engine["engine_class"](config)
    engine.initialize()

    gconfig = GenerationHyperparameters(max_new_tokens=2, greedy=False)
    tokenizer = load_hf_tokenizer(MODEL_PATH)

    workflow = RLVRWorkflow(
        reward_fn=_dummy_reward_fn,
        gconfig=gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
    )
    data = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    for _ in range(bs * 2):
        engine.submit(data, workflow=workflow, group_size=n_samples)

    if ofp < 1:
        # Due to controlled offpolicyness, not all requests are committed
        with pytest.raises(TimeoutError):
            engine.wait(count=bs * 2, timeout=10)
    else:
        results = engine.wait(count=bs * 2, timeout=10)
        result = concat_padded_tensors([r for r in results if r is not None])
        assert result["attention_mask"].shape[0] == bs * 2 * n_samples

    # Update model version
    engine.set_version(1)
    print("Updated model version", flush=True)

    # submit again
    for _ in range(bs * 2):
        engine.submit(data, workflow=workflow, group_size=n_samples)

    if ofp < 2:
        # Due to controlled offpolicyness, not all requests are committed
        with pytest.raises(TimeoutError):
            engine.wait(count=bs * 4, timeout=5)
    else:
        # 2 * bs samples haved been retrived above
        results_list = engine.wait(count=bs * 2, timeout=5)
        results = concat_padded_tensors([r for r in results_list if r is not None])
        assert results["attention_mask"].shape[0] == bs * 2 * n_samples

    engine.destroy()
    assert not dist.is_initialized()


@pytest.mark.slow
@pytest.mark.ci
def test_wait_for_task(inference_engine):
    """Test wait_for_task functionality with real inference engines."""
    from areal.workflow import RLVRWorkflow

    config = InferenceEngineConfig(
        backend="sglang:d1",
        experiment_name=inference_engine["expr_name"],
        trial_name=inference_engine["trial_name"],
        max_concurrent_rollouts=8,
        consumer_batch_size=4,
        setup_timeout=360,
        max_head_offpolicyness=int(1e10),
    )

    engine = inference_engine["engine_class"](config)
    engine.initialize()

    gconfig = GenerationHyperparameters(max_new_tokens=8, greedy=False, n_samples=1)
    tokenizer = load_hf_tokenizer(MODEL_PATH)

    workflow = RLVRWorkflow(
        reward_fn=_dummy_reward_fn,
        gconfig=gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
    )

    data = {"messages": [{"role": "user", "content": "Hello"}]}

    # Test 1: Wait for specific task
    task_id1 = engine.submit(data, workflow=workflow)
    task_id2 = engine.submit(data, workflow=workflow)
    task_id3 = engine.submit(data, workflow=workflow)

    result2 = engine.wait_for_task(task_id2, timeout=30.0)
    assert result2 is not None
    assert isinstance(result2, dict)

    result1 = engine.wait_for_task(task_id1, timeout=30.0)
    assert result1 is not None

    result3 = engine.wait_for_task(task_id3, timeout=30.0)
    assert result3 is not None

    # Test 2: Invalid task_id raises ValueError
    with pytest.raises(ValueError, match="never submitted"):
        engine.wait_for_task(999999, timeout=1.0)

    # Test 3: Timeout with raise_timeout=False returns None
    slow_task_id = engine.submit(data, workflow=workflow)
    # Immediately try to get it with very short timeout
    result = engine.wait_for_task(slow_task_id, timeout=0.001, raise_timeout=False)
    # Should timeout and return None (task hasn't completed yet)
    # But if it completed fast, that's ok too
    if result is None:
        # Now get it for real
        result = engine.wait_for_task(slow_task_id, timeout=30.0)
        assert result is not None

    # Test 4: Mix wait_for_task with regular wait()
    task_ids = [engine.submit(data, workflow=workflow) for _ in range(4)]

    # Get specific task
    specific_result = engine.wait_for_task(task_ids[1], timeout=30.0)
    assert specific_result is not None

    # Get remaining 3 with regular wait()
    remaining = engine.wait(count=3, timeout=30.0)
    assert len(remaining) == 3
    assert all(r is not None for r in remaining)

    engine.destroy()
    assert not dist.is_initialized()


@pytest.mark.slow
@pytest.mark.ci
def test_disk_update_weights_from_fsdp_engine(tmp_path_factory, inference_engine):
    """Test disk-based weight updates from FSDP engine to inference engine."""

    # setup FSDP engine
    from areal.api import FinetuneSpec
    from areal.api.cli_args import OptimizerConfig, TrainEngineConfig
    from areal.engine import FSDPEngine

    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7777"

    engine_config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name=inference_engine["expr_name"],
        trial_name=inference_engine["trial_name"],
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
    )
    train_engine = FSDPEngine(engine_config)
    train_engine.create_process_group()
    inf_engine = None
    try:
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        train_engine.initialize(None, ft_spec)
        train_engine.model_version = 100

        # setup name resolve
        import areal.utils.name_resolve as name_resolve
        from areal.api.cli_args import NameResolveConfig

        nfs_record_root = tmp_path_factory.mktemp("nfs_record_path")
        name_resolve_config = NameResolveConfig(
            type="nfs", nfs_record_root=nfs_record_root
        )
        name_resolve.reconfigure(name_resolve_config)

        config = InferenceEngineConfig(
            backend="sglang:d1",
            experiment_name=inference_engine["expr_name"],
            trial_name=inference_engine["trial_name"],
        )
        # initialize inference engine
        inf_engine = inference_engine["engine_class"](config)
        inf_engine.initialize()
        inf_engine.set_version(100)

        # test update weights
        path = tmp_path_factory.mktemp("update_weights_from_disk")
        update_weight_meta = WeightUpdateMeta(type="disk", path=str(path))
        train_engine.connect_engine(inf_engine, update_weight_meta)
        train_engine.set_version(100)
        train_engine.update_weights(update_weight_meta)
    finally:
        train_engine.destroy()
        if inf_engine is not None:
            inf_engine.destroy()
        assert not dist.is_initialized()
