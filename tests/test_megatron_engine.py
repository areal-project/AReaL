import json
import os
import time
from importlib.metadata import version as get_version
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from tests.utils import get_model_path

from areal.api import FinetuneSpec, SaveLoadMeta
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MegatronEngineConfig,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import MegatronEngine
from areal.infra.platforms import current_platform
from areal.models.mcore.bailing_v3_bridge import BailingV3Bridge
from areal.utils import logging

logger = logging.getLogger("TestMegatronEngine")

VOCAB_SIZE = 100
MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


@pytest.fixture(scope="module")
def mock_input(
    batch_size=5,
    min_seqlen=10,
    max_seqlen=20,
    device=current_platform.device_type,
) -> dict[str, Any]:
    """Create mock padded input data (same format for huggingface) for testing.
    Returns a dict with input_ids, attention_mask, and position_ids.
    """
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(
        0, VOCAB_SIZE, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device=device)

    attn_mask[
        torch.arange(0, max_seqlen, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


def mock_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    **kwargs,
) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logprobs)


def test_mark_duplicated_params_clears_tp_metadata_for_replicated_params():
    model = torch.nn.Module()

    duplicated_linear = torch.nn.Linear(2, 2, bias=False)
    duplicated_linear.tp_size = 1
    duplicated_linear.weight.tensor_model_parallel = True

    expert_linear = torch.nn.Linear(2, 2, bias=False)
    expert_linear.tp_size = 1
    expert_linear.weight.tensor_model_parallel = True

    sharded_linear = torch.nn.Linear(2, 2, bias=False)
    sharded_linear.tp_size = 4
    sharded_linear.weight.tensor_model_parallel = True

    model.add_module("duplicated_linear", duplicated_linear)
    model.add_module("moe_expert_linear", expert_linear)
    model.add_module("sharded_linear", sharded_linear)

    engine = type("DummyEngine", (), {"model": [model]})()

    MegatronEngine._mark_duplicated_params(engine)

    assert duplicated_linear.weight._is_duplicated
    assert not duplicated_linear.weight.tensor_model_parallel

    assert not hasattr(expert_linear.weight, "_is_duplicated")
    assert expert_linear.weight.tensor_model_parallel

    assert not hasattr(sharded_linear.weight, "_is_duplicated")
    assert sharded_linear.weight.tensor_model_parallel


def test_native_mbridge_save_finalizes_hf_config(monkeypatch, tmp_path):
    """The native mbridge saver runs the shared rank-zero HF finalizer."""
    engine = MegatronEngine.__new__(MegatronEngine)
    engine.model = [object()]
    engine.bridge_cls = "mbridge"
    engine.mcore_config = SimpleNamespace(use_mbridge_save=True)
    engine.config = SimpleNamespace(is_critic=False, path="/models/source")
    engine.bridge = Mock(safetensor_io=object())
    engine.process_group_initialized = True
    engine._cpu_group = object()

    finalize = Mock()
    source_config = {"model_type": "runtime_only", "torch_dtype": "bfloat16"}
    snapshot = Mock(return_value=source_config)
    monkeypatch.setattr(
        "areal.engine.megatron_engine.finalize_hf_export",
        finalize,
    )
    monkeypatch.setattr(
        "areal.engine.megatron_engine.load_hf_config_snapshot",
        snapshot,
    )
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(current_platform, "synchronize", lambda: None)

    engine._save_model_to_hf(str(tmp_path), base_model_path=str(tmp_path))

    snapshot.assert_called_once_with(str(tmp_path))
    engine.bridge.save_weights.assert_called_once_with(
        models=engine.model,
        weights_path=str(tmp_path),
    )
    finalize.assert_called_once_with(
        engine.bridge.hf_config,
        str(tmp_path),
        source_model_path=str(tmp_path),
        source_config=source_config,
    )


def test_bailing_v3_mbridge_declared_mtp_is_scrubbed(tmp_path):
    """Bailing V3 declares MTP in HF config but intentionally drops its head."""
    engine = MegatronEngine.__new__(MegatronEngine)
    engine.bridge_cls = "mbridge"
    engine.mcore_config = SimpleNamespace(enable_mtp=False)
    engine.logger = Mock()
    engine.bridge = Mock(spec=BailingV3Bridge)
    engine.hf_config = SimpleNamespace(num_nextn_predict_layers=1)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"num_nextn_predict_layers": 1}, f)

    assert engine._mtp_head_dropped
    engine._scrub_mtp_from_saved_config(str(tmp_path))

    with open(tmp_path / "config.json") as f:
        assert json.load(f)["num_nextn_predict_layers"] == 0


def test_bailing_v3_bridge_rejects_virtual_pipeline_parallelism(monkeypatch):
    """Bailing V3 fails before model construction when VPP is requested."""
    engine = MegatronEngine.__new__(MegatronEngine)
    engine.bridge_cls = "mbridge"
    engine.config = SimpleNamespace(path="organization/ling-v3")
    engine.mcore_config = SimpleNamespace(
        enable_mtp=False, virtual_pipeline_parallel_size=2
    )
    monkeypatch.setattr(
        "areal.engine.megatron_engine.PretrainedConfig.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            architectures=["BailingMoeV3ForCausalLM"]
        ),
    )

    with pytest.raises(ValueError, match="does not support virtual pipeline"):
        engine._build_hf_mcore_bridge()


def test_bailing_v3_bridge_rejects_mtp_construction(monkeypatch):
    """Bailing V3 cannot claim an MTP head that its mbridge never constructs."""
    engine = MegatronEngine.__new__(MegatronEngine)
    engine.bridge_cls = "mbridge"
    engine.config = SimpleNamespace(path="organization/ling-v3")
    engine.mcore_config = SimpleNamespace(
        enable_mtp=True, virtual_pipeline_parallel_size=1
    )
    monkeypatch.setattr(
        "areal.engine.megatron_engine.PretrainedConfig.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            architectures=["BailingMoeV3ForCausalLM"]
        ),
    )

    with pytest.raises(ValueError, match="does not support enable_mtp"):
        engine._build_hf_mcore_bridge()


# Cannot use a "module" scope since process groups can only be initialized once.
@pytest.fixture
def engine():
    logger.info(f"megatron.core version={get_version('megatron.core')}")
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7777",
        }
    )
    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(),
    )
    alloc_mode = ModelAllocation.from_str("fsdp:d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = MegatronEngine(config)
    engine.create_process_group(alloc_mode.parallel)
    engine.initialize(addr=None, ft_spec=ft_spec)
    logger.info(f"mcore GPTModel initialized: {engine.model}")
    engine.get_device_stats().log("initialize")
    try:
        yield engine
    finally:
        engine.destroy()
        assert not dist.is_initialized()


def test_simple_forward(engine, mock_input):
    engine.eval()
    result = engine.forward(mock_input)
    logger.info(f"Forward done, result: {result}")


def test_simple_train(engine, mock_input):
    engine.train()
    train_result = engine.train_batch(
        mock_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: torch.tensor(1.0, device=engine.device),
    )
    engine.step_lr_scheduler()
    logger.info(f"Train done, result={train_result}")


@torch.no_grad()
def test_hf_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("hf_engine_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="hf",
        tokenizer=tokenizer,
        with_optim=False,
        base_model_path=None,
    )

    old = engine.forward(input_=mock_input)
    start = time.perf_counter()
    engine.save(save_load_meta)
    logger.info(f"Save done, time cost: {time.perf_counter() - start:.4f} seconds.")
    with open(path / "config.json") as f:
        saved_config = json.load(f)
    assert saved_config["model_type"] == engine.hf_config.model_type
    for name, param in engine.model.named_parameters():
        param.zero_()

    start = time.perf_counter()
    engine.load(save_load_meta)
    logger.info(f"Load done, time cost: {time.perf_counter() - start:.4f} seconds.")
    new = engine.forward(input_=mock_input)
    assert torch.allclose(old, new)


@torch.no_grad()
@pytest.mark.slow
def test_dcp_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("megatron_engine_dcp_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
        tokenizer=tokenizer,
        with_optim=True,
        base_model_path=None,
    )

    old = engine.forward(input_=mock_input)
    start = time.perf_counter()
    engine.save(save_load_meta)
    logger.info(f"Save done, time cost: {time.perf_counter() - start:.4f} seconds.")
    for name, param in engine.model.named_parameters():
        param.zero_()

    start = time.perf_counter()
    engine.load(save_load_meta)
    logger.info(f"Load done, time cost: {time.perf_counter() - start:.4f} seconds.")
    new = engine.forward(input_=mock_input)
    assert torch.allclose(old, new)
