# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for historical sampler heads across agent turns and tools."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from areal.api import ModelResponse
from areal.api.cli_args import GenerationHyperparameters
from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.proxy.server import (
    deserialize_interactions,
    serialize_interactions,
)
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils.score_centering import score_centering_logprobs


@pytest.fixture
def tokenizer():
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[EOS]": 1,
                "[PAD]": 2,
                "user": 3,
                "assistant": 4,
                "task": 5,
                "answer": 6,
                "tool": 7,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    result = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )
    result.chat_template = (
        "{% for m in messages %}{{ m.role }} {{ m.content or '' }} [EOS] {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    return result


def response(prompt, output, *, version=1, width=2, tokenizer=None, stop="stop"):
    return ModelResponse(
        input_tokens=list(prompt),
        output_tokens=list(output),
        output_logprobs=[-1.5 - version / 10] * len(output),
        output_versions=[version] * len(output),
        stop_reason=stop,
        sampler_head_ids=[list(range(3, 3 + width)) for _ in output] if width else None,
        sampler_head_logprobs=[[-2.0 - version / 10] * width for _ in output]
        if width
        else None,
        tokenizer=tokenizer,
    )


def interaction(prompt, output, *, parent=None, width=2, version=1, template="concat"):
    return InteractionWithTokenLogpReward(
        model_response=response(prompt, output, version=version, width=width),
        score_centering_top_k=width,
        parent=parent,
        chat_template_type=template,
        reward=float(version),
    )


@pytest.mark.parametrize("template", ["hf", "concat"])
@pytest.mark.parametrize("gap", [[], [7, 1, 4]])
@pytest.mark.parametrize("outputs", [([6, 1, 1], [6, 1]), ([], [6]), ([6], [])])
def test_historical_heads_align_with_masks_versions_and_empty_turns(
    template, gap, outputs
):
    """Synthetic EOS/tool prompts cannot acquire sampled-token metadata."""
    first, second = outputs
    parent = interaction([3, 5, 1, 4], first, template=template)
    prefix = parent.model_response.input_tokens + first
    child = interaction(
        prefix + gap, second, parent=parent, version=9, template=template
    )
    before = {k: v.clone() for k, v in parent.to_tensor_dict().items()}
    data = child.to_tensor_dict()
    assert data["sampler_head_ids"].shape == (*data["input_ids"].shape, 2)
    active = data["loss_mask"].bool()
    assert active.sum().item() == len(second) + (
        len(first) if template == "concat" else 0
    )
    assert not data["sampler_head_ids"][~active].any()
    assert torch.isneginf(data["sampler_head_logprobs"][~active]).all()
    if template == "concat":
        for key in (
            "sampler_head_ids",
            "sampler_head_logprobs",
            "versions",
            "logprobs",
            "loss_mask",
        ):
            torch.testing.assert_close(
                data[key][:, : len(prefix)], before[key], rtol=0, atol=0
            )
    if second:
        torch.testing.assert_close(
            data["sampler_head_logprobs"][0, -len(second) :],
            torch.full((len(second), 2), -2.9),
            rtol=0,
            atol=0,
        )
        assert data["versions"][0, -len(second) :].tolist() == [9] * len(second)
    for key, value in before.items():
        torch.testing.assert_close(parent.to_tensor_dict()[key], value, rtol=0, atol=0)
    # The HTTP representation transports complete tensor metadata, including -inf.
    restored = deserialize_interactions(serialize_interactions({"leaf": child}))[
        "leaf"
    ].to_tensor_dict()
    for key, value in data.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("widths", [(2, 0), (0, 2), (2, 3)])
def test_concat_rejects_mixed_head_widths(widths):
    """An ancestor without the same metadata cannot silently become trainable."""
    parent = interaction([3], [6], width=widths[0])
    child = interaction([3, 6, 7], [1], parent=parent, width=widths[1])
    with pytest.raises(ValueError, match="consistently|width changed"):
        child.to_tensor_dict()


@pytest.mark.parametrize("field", ["sampler_head_ids", "sampler_head_logprobs"])
@pytest.mark.parametrize("outputs", [[], [6]])
def test_requested_heads_must_be_present_even_for_empty_turn(field, outputs):
    """Remember requested K independently from a possibly missing response."""
    item = interaction([3], outputs)
    setattr(item.model_response, field, None)
    with pytest.raises(ValueError, match="Missing or misaligned"):
        item.to_tensor_dict()


@pytest.mark.parametrize("prompt", [[3], [3, 7], [7, 6, 1]])
def test_concat_rejects_retokenized_or_truncated_parent_prefix(prompt):
    parent = interaction([3], [6, 1])
    child = interaction(prompt, [6], parent=parent)
    with pytest.raises(ValueError, match="sampled token prefix"):
        child.to_tensor_dict()


@pytest.mark.parametrize("api_format", ["completions", "responses", "openai-agents"])
def test_generation_configuration_transports_heads_without_new_api_fields(api_format):
    config = GenerationHyperparameters(score_centering_top_k=128)
    payload = config.to_openai_args_dict(api_format=api_format)
    assert payload["extra_body"] == {"score_centering_top_k": 128}
    assert "score_centering_top_k" not in payload
    assert "extra_body" not in config.new(score_centering_top_k=0).to_openai_args_dict(
        api_format=api_format
    )
    with pytest.raises(ValueError, match="temperature=1"):
        config.new(temperature=0.6).to_openai_args_dict(api_format=api_format)


class RecordingEngine:
    def __init__(self, tokenizer, outputs=None):
        self.tokenizer = tokenizer
        self.requests = []
        self.outputs = outputs or [[6, 1, 1], [6, 1]]

    async def agenerate(self, request):
        index = len(self.requests)
        self.requests.append(request)
        return response(
            request.input_ids,
            self.outputs[index % len(self.outputs)],
            version=index + 1,
            width=request.gconfig.score_centering_top_k,
            tokenizer=self.tokenizer,
        )


def make_client(engine, tokenizer, template):
    return ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        chat_template_type=template,
        reasoning_parser="",
        api_key="test",
        base_url="http://test.invalid/v1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("template", ["hf", "concat"])
@pytest.mark.parametrize("enabled", [False, True])
async def test_openai_two_turn_export_preserves_raw_samples(
    api, template, enabled, tokenizer
):
    """Display strips repeated EOS; training must still retain every sample."""
    engine = RecordingEngine(tokenizer)
    client = make_client(engine, tokenizer, template)
    options = {"extra_body": {"score_centering_top_k": 2}} if enabled else {}
    message = {"role": "user", "content": "task"}
    try:
        if api == "chat":
            first = await client.chat.completions.create(
                messages=[message], max_completion_tokens=8, **options
            )
            second = await client.chat.completions.create(
                messages=[message, first.choices[0].message, message],
                max_completion_tokens=8,
                **options,
            )
        else:
            first = await client.responses.create(
                input=[message], max_output_tokens=8, **options
            )
            second = await client.responses.create(
                input=[message, *first.output, message], max_output_tokens=8, **options
            )
        client.set_reward(second.id, 1.0)
        exported = client.export_interactions(
            style="concat" if template == "concat" else "individual"
        )
        leaf = exported[second.id]
        assert leaf.parent is not None
        assert leaf.model_response.output_tokens == [6, 1]
        data = leaf.to_tensor_dict()
        assert ("sampler_head_ids" in data) is enabled
        assert [req.gconfig.score_centering_top_k for req in engine.requests] == [
            2 if enabled else 0
        ] * 2
        if enabled:
            assert torch.isfinite(
                data["sampler_head_logprobs"][data["loss_mask"].bool()]
            ).all()
            assert data["loss_mask"].sum().item() == (5 if template == "concat" else 2)
            assert data["versions"][data["loss_mask"].bool()].tolist() == (
                [1, 1, 1, 2, 2] if template == "concat" else [2, 2]
            )
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize(
    "options",
    [
        {"temperature": 0.6},
        {"top_p": 0.9},
        {"frequency_penalty": 0.1},
        {"presence_penalty": 0.1},
        {"logit_bias": {"1": 10}},
        {"response_format": {"type": "json_object"}},
        {"extra_body": {"score_centering_top_k": 2, "regex": "[0-9]+"}},
        {"extra_body": {"score_centering_top_k": True}},
        {"extra_body": {"score_centering_top_k": -1}},
        {"extra_body": {"score_centering_top_k": 1025}},
    ],
)
async def test_invalid_sampling_fails_before_generation_or_cache(
    api, options, tokenizer
):
    engine = RecordingEngine(tokenizer)
    client = make_client(engine, tokenizer, "concat")
    options = {"extra_body": {"score_centering_top_k": 2}, **options}
    try:
        with pytest.raises(ValueError, match="Score.centering"):
            if api == "chat":
                await client.chat.completions.create(
                    messages=[{"role": "user", "content": "task"}], **options
                )
            else:
                await client.responses.create(input="task", **options)
        assert not engine.requests
        assert not client.export_interactions(style="individual")
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("has_eos", [False, True])
@pytest.mark.parametrize("empty_gap", [False, True])
async def test_native_multiturn_appends_only_new_prompt_rows(
    tokenizer, monkeypatch, has_eos, empty_gap
):
    from areal.utils import stats_tracker
    from areal.workflow.multi_turn import MultiTurnWorkflow

    monkeypatch.setattr(
        stats_tracker,
        "get",
        lambda *args: SimpleNamespace(scalar=lambda **kwargs: None),
    )
    output = [6, 1] if has_eos else [6]
    engine = RecordingEngine(tokenizer, [output, [6, 1]])
    workflow = MultiTurnWorkflow(
        lambda **kwargs: 0,
        GenerationHyperparameters(score_centering_top_k=2),
        tokenizer,
        max_turns=2,
        turn_discount=0.9,
    )
    workflow.async_reward_fn = AsyncMock(side_effect=[0.0, 1.0])
    if empty_gap:
        workflow.multi_turn_prompt_ids = []
    result = await workflow.arun_episode(
        engine, {"messages": [{"role": "user", "content": "task"}]}
    )
    prefix = engine.requests[0].input_ids + output
    assert engine.requests[1].input_ids[: len(prefix)] == prefix
    active = result["loss_mask"].bool()
    assert active.sum().item() == len(output) + 2
    assert result["input_ids"].shape[1] == len(engine.requests[1].input_ids) + 2
    assert result["turn_ids"][active].tolist() == [0] * len(output) + [1, 1]
    assert result["versions"][active].tolist() == [1] * len(output) + [2, 2]
    assert torch.isneginf(result["sampler_head_logprobs"][~active]).all()
    assert result["rewards"].item() == pytest.approx(0.9)


@pytest.mark.parametrize("tis_cap", [None, 2.0])
def test_exported_agent_batch_has_dense_score_centering_gradient(tis_cap):
    """Compare exported historical metadata against an independent full-vocab oracle."""
    parent = interaction([3, 5], [6, 1])
    child = interaction([3, 5, 6, 1, 7, 4], [6, 1], parent=parent, version=5)
    data = child.to_tensor_dict()
    active = data["loss_mask"].bool().squeeze(0)
    labels = data["input_ids"][0, active].long()
    ids = data["sampler_head_ids"][0, active]
    qh = data["sampler_head_logprobs"][0, active].double()
    torch.manual_seed(13)
    logits = torch.randn(len(labels), 8, dtype=torch.float64, requires_grad=True)
    oracle = logits.detach().clone().requires_grad_()
    sampled, _, correction = score_centering_logprobs(
        logits, labels, ids, qh, tis_cap=tis_cap
    )
    p_log = oracle.log_softmax(-1)
    with torch.no_grad():
        p = p_log.exp()
        q = p.clone()
        head_mask = torch.zeros_like(p, dtype=torch.bool).scatter_(1, ids, True)
        rho = (1 - qh.exp().sum(-1)) / p.masked_fill(head_mask, 0).sum(-1)
        q *= rho[:, None]
        q.scatter_(1, ids, qh.exp())
        weight = torch.ones_like(p) if tis_cap is None else (p / q).clamp_max(tis_cap)
        coefficients = q * weight - (q * weight).sum(-1, keepdim=True) * p
    expected_correction = (coefficients * p_log).sum(-1)
    actual = (sampled - correction).sum()
    expected = (
        p_log.gather(-1, labels[:, None]).squeeze(-1) - expected_correction
    ).sum()
    actual.backward()
    expected.backward()
    # The head/tail surrogate omits a detached zero-gradient baseline, so its
    # scalar value need not equal the full-vocabulary expectation.
    assert torch.isfinite(correction).all()
    torch.testing.assert_close(logits.grad, oracle.grad, rtol=1e-10, atol=1e-10)


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("template", ["hf", "concat"])
async def test_real_tool_parser_reply_and_continuation_keep_sampled_heads(
    api, template, tokenizer
):
    """Parse a real text calculator call; its reply remains context-only."""
    raw_call = "<tool_call><function=add><parameter=a>2</parameter><parameter=b>3</parameter></function></tool_call>"
    tokenizer.add_tokens([raw_call])
    raw_id = tokenizer.convert_tokens_to_ids(raw_call)
    engine = RecordingEngine(tokenizer, [[raw_id, 1, 1], [6, 1]])
    client = ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        chat_template_type=template,
        tool_call_parser="qwen3_coder",
        tool_call_parser_allow_recovery=False,
        reasoning_parser="",
        api_key="test",
        base_url="http://test.invalid/v1",
    )
    definition = {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    }
    options = {"extra_body": {"score_centering_top_k": 2}}
    message = {"role": "user", "content": "task"}
    try:
        if api == "chat":
            tools = [{"type": "function", "function": definition}]
            first = await client.chat.completions.create(
                messages=[message], tools=tools, max_completion_tokens=8, **options
            )
            call = first.choices[0].message.tool_calls[0]
            import json

            args = json.loads(call.function.arguments)
            assert args == {"a": 2, "b": 3}
            reply = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(args["a"] + args["b"]),
            }
            second = await client.chat.completions.create(
                messages=[message, first.choices[0].message, reply],
                tools=tools,
                max_completion_tokens=8,
                **options,
            )
        else:
            tools = [{"type": "function", **definition}]
            first = await client.responses.create(
                input=[message], tools=tools, max_output_tokens=8, **options
            )
            call = next(item for item in first.output if item.type == "function_call")
            reply = {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": "5",
            }
            second = await client.responses.create(
                input=[message, *first.output, reply],
                tools=tools,
                max_output_tokens=8,
                **options,
            )
        parent = client.get_interaction(first.id)
        leaf = client.get_interaction(second.id)
        assert leaf.parent is parent
        assert parent.model_response.output_tokens == [raw_id, 1, 1]
        data = leaf.to_tensor_dict()
        active = data["loss_mask"].bool()
        assert data["input_ids"][active].tolist() == (
            [raw_id, 1, 1, 6, 1] if template == "concat" else [6, 1]
        )
        assert torch.isneginf(data["sampler_head_logprobs"][~active]).all()
        if template == "concat":
            end = parent.model_response.input_len + 3
            assert not data["loss_mask"][0, end : leaf.model_response.input_len].any()
            torch.testing.assert_close(
                data["sampler_head_logprobs"][0, :end],
                parent.to_tensor_dict()["sampler_head_logprobs"][0],
                rtol=0,
                atol=0,
            )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_streaming_and_retry_export_keep_only_consumed_heads(tokenizer):
    """A retried response and early-ended stream retain their own raw metadata."""
    engine = RecordingEngine(tokenizer)
    client = make_client(engine, tokenizer, "concat")
    message = {"role": "user", "content": "task"}
    options = {"max_completion_tokens": 8, "extra_body": {"score_centering_top_k": 2}}
    try:
        await client.chat.completions.create(messages=[message], **options)
        stream = await client.chat.completions.create(
            messages=[message], stream=True, **options
        )
        chunk = await anext(stream)
        await stream.aclose()
        exported = client.export_interactions(style="concat", drop_retry_orphans=True)
        assert list(exported) == [chunk.id]
        data = exported[chunk.id].to_tensor_dict()
        active = data["loss_mask"].bool()
        assert data["versions"][active].tolist() == [2, 2]
        torch.testing.assert_close(
            data["sampler_head_logprobs"][active],
            torch.full((2, 2), -2.2),
            rtol=0,
            atol=0,
        )
    finally:
        await client.close()


@pytest.mark.parametrize("widths", [(2, 3), (0, 2), (2, 0)])
def test_individual_batch_rejects_padding_the_candidate_dimension(widths):
    from areal.experimental.openai.types import concat_tensor_interactions

    first = interaction([3], [6], width=widths[0], template="hf")
    second = interaction([3, 5], [6, 1], width=widths[1], template="hf")
    with pytest.raises(ValueError, match="must share score_centering_top_k"):
        concat_tensor_interactions({"first": first, "second": second})


def test_separate_rollout_batches_cannot_pad_sampler_candidate_width():
    from areal.utils.data import concat_padded_tensors

    first = interaction([3], [6], width=2).to_tensor_dict()
    second = interaction([3, 5], [6, 1], width=3).to_tensor_dict()
    with pytest.raises(ValueError, match="only sequence padding"):
        concat_padded_tensors([first, second])
