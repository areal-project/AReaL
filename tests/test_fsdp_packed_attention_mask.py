# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import areal.engine.fsdp_engine as fsdp_module
from areal.api.cli_args import MicroBatchSpec, TrainEngineConfig


def _make_engine(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
) -> fsdp_module.FSDPEngine:
    monkeypatch.setattr(
        fsdp_module.AutoConfig,
        "from_pretrained",
        lambda **_: SimpleNamespace(model_type=model_type),
    )
    monkeypatch.setattr(fsdp_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(fsdp_module.dist, "get_rank", lambda group=None: 0)

    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-packed-attention-mask",
        trial_name="test0",
        path="unused",
        mb_spec=MicroBatchSpec(n_mbs=1, max_tokens_per_mb=16),
        pad_to_maximum=True,
    )
    engine = fsdp_module.FSDPEngine(config)
    engine.logger = MagicMock()
    return engine


@pytest.mark.parametrize(
    "model_type",
    [
        pytest.param("llama", id="llama-minicpm5"),
        pytest.param("qwen2", id="qwen2-layer-mapping"),
        pytest.param("qwen3", id="qwen3-layer-mapping"),
        pytest.param("qwen3_moe", id="qwen3-moe"),
        pytest.param("qwen3_5", id="qwen3-5"),
        pytest.param("gemma3", id="gemma3-layer-mapping"),
    ],
)
def test_prepare_mb_list_uses_model_compatible_attention_mask(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
) -> None:
    """Packed forwards should let each Transformers model build its own mask."""
    engine = _make_engine(monkeypatch, model_type)
    batch = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
        "attention_mask": torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        ),
    }

    mb_list = engine._prepare_mb_list(batch)

    assert mb_list.padded_mbs is not None
    for mb in [*mb_list.mbs, *mb_list.padded_mbs]:
        assert mb["attention_mask"] is None


@pytest.mark.parametrize("model_type", ["qwen2_5_vl", "qwen3_vl"])
def test_prepare_mb_list_uses_model_compatible_mask_for_qwen_vl(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
) -> None:
    """Qwen-VL mRoPE preparation should also pass a model-native mask."""
    engine = _make_engine(monkeypatch, model_type)
    position_ids = torch.arange(3).expand(3, 2, -1).clone()
    compute_3d_position_ids = MagicMock(return_value=position_ids)
    engine.model = SimpleNamespace(
        model=SimpleNamespace(compute_3d_position_ids=compute_3d_position_ids)
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
        "attention_mask": torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        ),
        "mm_token_type_ids": torch.zeros((2, 3), dtype=torch.long),
    }

    mb_list = engine._prepare_mb_list(batch)

    compute_3d_position_ids.assert_called_once()
    assert mb_list.padded_mbs is not None
    for mb in [*mb_list.mbs, *mb_list.padded_mbs]:
        assert mb["attention_mask"] is None


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_llama_builds_isolated_packed_mask_from_position_ids(
    attn_implementation: str,
) -> None:
    """A reset position sequence should prevent cross-sample attention."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            _attn_implementation=attn_implementation,
        )
    ).eval()

    first = torch.tensor([[1, 2, 3]])
    second = torch.tensor([[4, 5]])
    packed = torch.cat((first, second), dim=1)

    with torch.no_grad():
        first_logits = model(
            first,
            attention_mask=None,
            position_ids=torch.tensor([[0, 1, 2]]),
            use_cache=False,
        ).logits
        second_logits = model(
            second,
            attention_mask=None,
            position_ids=torch.tensor([[0, 1]]),
            use_cache=False,
        ).logits
        packed_logits = model(
            packed,
            attention_mask=None,
            position_ids=torch.tensor([[0, 1, 2, 0, 1]]),
            use_cache=False,
        ).logits

    torch.testing.assert_close(
        packed_logits[:, :3], first_logits, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        packed_logits[:, 3:], second_logits, rtol=1e-5, atol=1e-6
    )
