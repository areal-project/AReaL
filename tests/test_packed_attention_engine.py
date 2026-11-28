# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close
from transformers import (
    AttentionInterface,
    AutoModelForCausalLM,
    Gemma3Config,
    Qwen3Config,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import areal.engine.fsdp_engine as engine_module
from areal.engine.fsdp_engine import FSDPEngine


@pytest.fixture(autouse=True)
def preserve_sdpa_registration():
    """Do not leak the engine's global SDPA override into other tests."""
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        yield
    finally:
        AttentionInterface.register("sdpa", original)


class _PreparedMicroBatches(list):
    """Minimal adapter for the protocol used by forward_backward_batch."""

    def __init__(self, inputs):
        item = SimpleNamespace(
            padded_mb=inputs,
            orig_mb=inputs,
            padding_length=2,
        )
        super().__init__([item])
        self.mbs = [inputs]
        self.group_lens = [inputs["input_ids"].shape[-1] - 2]


def _make_engine(monkeypatch, model_config, attn_impl, model=None):
    # Exercise the real constructor, including SDPA registration, without
    # downloading config or initializing a distributed process group.
    monkeypatch.setattr(
        engine_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: model_config,
    )
    monkeypatch.setattr(
        engine_module.dist,
        "all_reduce",
        lambda *args, **kwargs: None,
    )
    config = SimpleNamespace(
        path="unused-local-test-model",
        optimizer=None,
        enable_tree_training=False,
        attn_impl=attn_impl,
        dtype="float32",
    )
    engine = FSDPEngine(config)
    engine.device = torch.device("cpu")
    engine.parallel_helper = SimpleNamespace(sp_size=1)
    engine._initialized = True
    engine._cpu_group = None
    if model is not None:
        engine.model = model
    return engine


def _sequence_loss(logits, sequences):
    """Next-token CE, excluding cross-sequence transitions and padding."""
    total = logits.new_zeros(())
    offset = 0
    token_count = 0
    for sequence in sequences:
        length = sequence.numel()
        total = total + F.cross_entropy(
            logits[offset : offset + length - 1],
            sequence[1:],
            reduction="sum",
        )
        offset += length
        token_count += length - 1
    return total / token_count


@pytest.mark.parametrize("attn_impl", ["sdpa", "eager"])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("sliding", [False, True])
def test_engine_packed_training_matches_separate(
    monkeypatch, attn_impl, checkpointing, sliding
):
    torch.manual_seed(17)
    layer_types = (
        ["full_attention", "sliding_attention"]
        if sliding
        else ["full_attention", "full_attention"]
    )
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
        use_cache=False,
        use_sliding_window=sliding,
        sliding_window=3 if sliding else None,
        layer_types=layer_types,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_impl)
    reference = deepcopy(model)

    model.train()
    reference.train()
    if checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    engine = _make_engine(monkeypatch, config, attn_impl, model)

    generator = torch.Generator().manual_seed(23)
    sequences = [
        torch.randint(1, config.vocab_size, (length,), generator=generator)
        for length in (5, 7, 4)
    ]
    bounds = [0]
    for sequence in sequences:
        bounds.append(bounds[-1] + sequence.numel())

    pad = torch.zeros(2, dtype=torch.long)
    inputs = {
        "input_ids": torch.cat([*sequences, pad]).unsqueeze(0),
        "position_ids": torch.cat(
            [*(torch.arange(s.numel()) for s in sequences), torch.arange(2)]
        ).unsqueeze(0),
        "cu_seqlens": torch.tensor(bounds, dtype=torch.int32),
        "attention_mask": {
            "full_attention": None,
            "sliding_attention": None,
        },
        "use_cache": False,
    }

    # Independent baseline: normal model forwards, no packed mask or bounds.
    reference_logits = torch.cat(
        [
            reference(sequence.unsqueeze(0), use_cache=False).logits[0]
            for sequence in sequences
        ],
        dim=0,
    )
    reference_loss = _sequence_loss(reference_logits, sequences)
    reference_loss.backward()

    captured = {}

    def process_output(logits, context):
        loss = _sequence_loss(logits, sequences)
        captured["logits"] = logits[: bounds[-1]].detach()
        captured["loss"] = loss.detach()
        return loss

    # The engine must supply the protection and execute backward itself.
    engine.forward_backward_batch(
        _PreparedMicroBatches(inputs),
        process_output,
        forward_only=False,
    )

    assert_close(
        captured["logits"],
        reference_logits.detach(),
        rtol=1e-4,
        atol=1e-5,
    )
    assert_close(
        captured["loss"],
        reference_loss.detach(),
        rtol=1e-4,
        atol=1e-5,
    )

    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        expected = reference_parameters[name]
        assert parameter.grad is not None, name
        assert expected.grad is not None, name
        assert_close(
            parameter.grad,
            expected.grad,
            rtol=1e-4,
            atol=1e-5,
            msg=lambda message, name=name: f"{name}: {message}",
        )

    # The prepared batch must remain reusable after forward/backward.
    assert "packed_seq_bounds" not in inputs
    assert inputs["attention_mask"] == {
        "full_attention": None,
        "sliding_attention": None,
    }


def test_eager_masks_read_gemma3_nested_text_config(monkeypatch):
    # Config-only test: no vision model or image tensors are created.
    config = Gemma3Config(
        text_config={
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "sliding_attention"],
            "sliding_window": 3,
        }
    )
    engine = _make_engine(monkeypatch, config, "eager")

    masks = engine._packed_attention_masks(
        {
            "input_ids": torch.ones((1, 8), dtype=torch.long),
            "cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32),
        }
    )
    full = masks["full_attention"][0, 0]
    sliding = masks["sliding_attention"][0, 0]
    blocked = torch.finfo(torch.float32).min

    # Same sequence: full attention allows distance 3; window=3 excludes it.
    assert full[3, 0].item() == 0.0
    assert sliding[3, 0].item() == blocked
    assert sliding[3, 1].item() == 0.0

    # Neither layer type may cross the packed-sequence boundary.
    assert full[4, 3].item() == blocked
    assert sliding[4, 3].item() == blocked

    # Both remain causal.
    assert full[0, 1].item() == blocked
    assert sliding[0, 1].item() == blocked
