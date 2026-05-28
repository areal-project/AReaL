"""Test script for Engine implementation."""

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from tests.utils import get_model_path

from areal.api import FinetuneSpec, SaveLoadMeta
from areal.api.cli_args import (
    FSDPEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine.fsdp_utils.attn_impl import BUILTIN_ATTN_IMPLS
from areal.infra.platforms import current_platform

VOCAB_SIZE = 100
MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


class DummyDeviceStats:
    def log(self, *_args, **_kwargs) -> None:
        pass


class DummyModel:
    def __init__(self):
        self.use_kernels = False
        self.gradient_checkpointing_calls = []

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing_calls.append(kwargs)


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


def get_engine(engine_type: str, model_path: str):
    from areal.engine import FSDPEngine

    engine_cls = {"fsdp": FSDPEngine}[engine_type]

    engine_config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name=f"test-{engine_type}-engine",
        trial_name="test0",
        path=model_path,
        optimizer=OptimizerConfig(),
    )
    engine = engine_cls(engine_config)
    engine.create_process_group()
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=100, train_batch_size=2)
    engine.initialize(None, ft_spec)
    return engine


def mock_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    **kwargs,
) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logprobs)


@pytest.fixture(
    params=[
        {"type": "fsdp", "construction": "config"},
        {"type": "fsdp", "construction": "from_pretrained"},
    ]
)
def engine(request):
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7777",
        }
    )

    construction = request.param["construction"]
    engine_type = request.param["type"]

    if construction == "config":
        engine = get_engine(engine_type, MODEL_PATH)
    else:
        # Create the engine using from_pretrained method, without TrainEngineConfig
        from areal.engine import FSDPEngine

        engine = FSDPEngine.from_pretrained(
            model=MODEL_PATH,
            experiment_name="test_exp",
            trial_name="test_trial",
            dp_size=1,
            tp_size=1,
            learning_rate=1e-5,
        )

        engine.create_process_group()
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        engine.initialize(None, ft_spec)

    print(f"✓ {engine_type.upper()} Engine created successfully")
    try:
        yield engine
    finally:
        engine.destroy()
        assert not dist.is_initialized()


@torch.no_grad()
def test_forward_microbatch(engine, mock_input):
    engine.eval()
    engine.config.mb_spec = MicroBatchSpec(n_mbs=2, max_tokens_per_mb=100)
    x2 = engine.forward(input_=mock_input)
    engine.config.mb_spec = MicroBatchSpec(n_mbs=1, max_tokens_per_mb=100)
    x1 = engine.forward(input_=mock_input)

    attn_mask = mock_input["attention_mask"]
    loss_mask = attn_mask.clone()
    loss_mask[:, :-1] = attn_mask[:, :-1] & attn_mask[:, 1:]
    loss_mask[:, -1] = False

    x1_valid = x1[loss_mask]
    x2_valid = x2[loss_mask]
    assert torch.allclose(x1_valid, x2_valid, atol=1e-4, rtol=1e-3), (
        (x1_valid - x2_valid).abs().max().item()
    )


@torch.no_grad()
def test_eval_batch(engine, mock_input):
    engine.eval()
    engine.config.mb_spec = MicroBatchSpec(n_mbs=2, max_tokens_per_mb=100)
    eval_result = engine.eval_batch(
        input_=mock_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )
    assert isinstance(eval_result, torch.Tensor), "Evaluation should return a tensor"
    assert eval_result.is_cuda, "Evaluation tensor should be on CUDA device"
    assert eval_result is not None, "Evaluation should return a loss value"
    print(f"✓ Evaluation successful, loss: {eval_result.item()}")


def test_train_batch(engine, mock_input):
    engine.train()
    engine.config.mb_spec = MicroBatchSpec(n_mbs=2, max_tokens_per_mb=100)
    train_result = engine.train_batch(
        input_=mock_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )
    assert isinstance(train_result, dict), "Training should return a dictionary"
    assert train_result["grad_norm"] is not None
    assert train_result["lr"] is not None
    print("✓ Training successful")


@torch.no_grad()
def test_hf_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("hf_engine_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="hf",
        tokenizer=tokenizer,
        with_optim=True,
        base_model_path=None,
    )

    engine.config.mb_spec = MicroBatchSpec(n_mbs=1, max_tokens_per_mb=100)
    old = engine.forward(input_=mock_input)
    engine.save(save_load_meta)

    for name, param in engine.model.named_parameters():
        param.zero_()

    engine.load(save_load_meta)
    new = engine.forward(input_=mock_input)
    assert torch.allclose(old, new)


@torch.no_grad()
@pytest.mark.slow
def test_dcp_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("dcp_engine_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
        tokenizer=tokenizer,
        with_optim=True,
        base_model_path=None,
    )

    engine.config.mb_spec = MicroBatchSpec(n_mbs=1, max_tokens_per_mb=100)
    old = engine.forward(input_=mock_input)
    engine.save(save_load_meta)

    for name, param in engine.model.named_parameters():
        param.zero_()

    engine.load(save_load_meta)
    new = engine.forward(input_=mock_input)
    assert torch.allclose(old, new)


@pytest.mark.parametrize(
    "attn_impl",
    [
        *BUILTIN_ATTN_IMPLS,
        "kernels-community/flash-attn",
        "kernels-community/flash-attn@main:flash_attn_varlen_func",
        "flash_attention_2|kernels-community/flash-attn@main:flash_attn_varlen_func",
    ],
)
def test_train_engine_config_accepts_builtin_and_kernel_attn_impls(attn_impl):
    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        attn_impl=attn_impl,
    )

    assert config.attn_impl == attn_impl


@pytest.mark.parametrize(
    "attn_impl",
    [
        "kernels-community",
        "kernels-community/flash-attn/extra",
        "kernels-community/flash-attn:entry:extra",
    ],
)
def test_train_engine_config_rejects_invalid_kernel_attn_impl(attn_impl):
    with pytest.raises(ValueError, match="attn_impl must be one of"):
        TrainEngineConfig(
            backend="fsdp:d1",
            experiment_name="test-experiment",
            trial_name="trial0",
            path="test-model",
            attn_impl=attn_impl,
        )


@pytest.mark.parametrize(
    ("memory_efficient_load", "expected_loader"),
    [(False, "from_pretrained"), (True, "from_config")],
)
def test_create_llm_actor_or_critic_forwards_attn_impl(
    monkeypatch, memory_efficient_load, expected_loader
):
    import areal.engine.fsdp_engine as fsdp_module

    calls = []

    class FakeModelFactory:
        @staticmethod
        def from_config(config, **kwargs):
            calls.append(("from_config", config, kwargs))
            return DummyModel()

        @staticmethod
        def from_pretrained(pretrained_model_name_or_path, **kwargs):
            calls.append(("from_pretrained", pretrained_model_name_or_path, kwargs))
            return DummyModel()

    monkeypatch.setattr(
        fsdp_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="qwen2"),
    )
    monkeypatch.setattr(fsdp_module, "is_valid_vision_model", lambda *_args: False)
    monkeypatch.setattr(fsdp_module, "AutoModelForCausalLM", FakeModelFactory)

    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        attn_impl="kernels-community/flash-attn",
        fsdp=FSDPEngineConfig(memory_efficient_load=memory_efficient_load),
    )
    engine = fsdp_module.FSDPEngine(config)
    engine.model_config = SimpleNamespace()

    model = engine._create_llm_actor_or_critic()

    assert isinstance(model, DummyModel)
    assert calls[0][0] == expected_loader
    assert calls[0][2]["attn_implementation"] == "kernels-community/flash-attn"
    assert calls[0][2]["dtype"] == torch.bfloat16


@pytest.mark.parametrize("memory_efficient_load", [False, True])
def test_create_device_model_applies_use_kernels(monkeypatch, memory_efficient_load):
    import areal.engine.fsdp_engine as fsdp_module

    monkeypatch.setattr(
        fsdp_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="qwen2"),
    )
    monkeypatch.setattr(fsdp_module, "is_valid_vision_model", lambda *_args: False)
    monkeypatch.setattr(fsdp_module, "load_hf_tokenizer", lambda *_args: object())
    monkeypatch.setattr(
        fsdp_module.current_platform, "set_device", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(fsdp_module.current_platform, "device_type", "cpu")
    monkeypatch.setattr(
        fsdp_module.FSDPEngine,
        "get_device_stats",
        lambda self: DummyDeviceStats(),
    )

    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        use_kernels=True,
        gradient_checkpointing=True,
        fsdp=FSDPEngineConfig(memory_efficient_load=memory_efficient_load),
    )
    engine = fsdp_module.FSDPEngine(config)
    engine.logger = MagicMock()
    dummy_model = DummyModel()
    monkeypatch.setattr(
        engine,
        "_create_llm_actor_or_critic",
        lambda: dummy_model,
    )
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)

    engine._create_device_model()

    assert engine.model is dummy_model
    assert engine.model.use_kernels is True
    assert engine.model.gradient_checkpointing_calls == [
        {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    ]


def test_fsdp_engine_config_construction():
    """Test that FSDPEngine.from_pretrained builds a valid config."""
    import areal.engine.fsdp_engine as fsdp_module

    engine = fsdp_module.FSDPEngine.from_pretrained(
        model=MODEL_PATH,
        experiment_name="test_exp",
        trial_name="test_trial",
        dp_size=1,
        learning_rate=1e-5,
        use_lora=True,
    )
    config = TrainEngineConfig(
        path=MODEL_PATH,
        backend="fsdp:d1t1",
        optimizer=OptimizerConfig(lr=1e-5),
        use_lora=True,
    )
    engine2 = fsdp_module.FSDPEngine(config=config)
    assert engine.config.path == engine2.config.path
    assert engine.config.backend == engine2.config.backend
    assert engine.config.optimizer.lr == engine2.config.optimizer.lr
    assert engine.config.use_lora == engine2.config.use_lora


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 4
    or int(os.environ.get("WORLD_SIZE", "1")) < 4,
    reason="requires 4 GPUs and 4 processes (torchrun --nproc_per_node=4)",
)
def test_fsdp_engine_alloc_mode_construction():
    """
    Test that FSDPEngine.from_pretrained builds a valid config.

    Run with 4 processes (required for dp_size=2, tp_size=2):
        torchrun --nproc_per_node=4 -m pytest tests/test_train_engine.py -v -k test_fsdp_engine_alloc_mode_construction
    """
    import areal.engine.fsdp_engine as fsdp_module

    engine = fsdp_module.FSDPEngine.from_pretrained(
        model=MODEL_PATH,
        experiment_name="test_exp",
        trial_name="test_trial",
        dp_size=2,
        tp_size=2,
        learning_rate=1e-5,
        use_lora=True,
    )

    engine.create_process_group()

    dist.barrier()

    assert engine.parallel_helper.dp_size == 2
    assert engine.parallel_helper.tp_size == 2

    engine.destroy()
