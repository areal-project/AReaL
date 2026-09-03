# SPDX-License-Identifier: Apache-2.0

import pytest
from openai.types.chat import ChatCompletionToolParam

from tests.utils import get_model_path

from areal.api import ModelResponse
from areal.experimental.openai.client import (
    _concat_prompt_token_ids_with_parent,
    _prepare_prompt,
)
from areal.experimental.openai.prompt_renderer import (
    IncrementalPromptRenderer,
    _find_kth,
)
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer

QWEN3_MODEL_PATH = "Qwen/Qwen3-0.6B"
LOCAL_QWEN3_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B"

QWEN25_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_QWEN25_PATH = "/storage/openpsi/models/Qwen__Qwen2.5-0.5B-Instruct"

WEATHER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    },
}

CALCULATOR_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate math expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {"type": "string", "description": "Math expression"}
            },
            "required": ["expr"],
        },
    },
}


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return load_hf_tokenizer(get_model_path(LOCAL_QWEN3_PATH, QWEN3_MODEL_PATH))


@pytest.fixture(scope="module")
def qwen25_tokenizer():
    return load_hf_tokenizer(get_model_path(LOCAL_QWEN25_PATH, QWEN25_MODEL_PATH))


class TestPromptRendererCapability:
    def test_find_kth_helper(self):
        lst = [1, 2, 3, 2, 4, 2, 5]
        assert _find_kth(lst, 2, 1) == 1
        assert _find_kth(lst, 2, 2) == 3
        assert _find_kth(lst, 2, 3) == 5
        assert _find_kth(lst, 2, 4) == -1
        assert _find_kth(lst, 99, 1) == -1

    def test_supported_tokenizers(self, qwen3_tokenizer, qwen25_tokenizer):
        assert IncrementalPromptRenderer.is_supported(
            qwen3_tokenizer, tools=[WEATHER_TOOL]
        )
        assert IncrementalPromptRenderer.is_supported(
            qwen25_tokenizer, tools=[WEATHER_TOOL]
        )

    def test_tokenizer_without_chat_template(self):
        class DummyTokenizer:
            chat_template = None
            name_or_path = "dummy"

        dummy = DummyTokenizer()
        assert not IncrementalPromptRenderer.is_supported(dummy)  # type: ignore[arg-type]


class TestIncrementalPromptRenderingParity:
    def test_render_initial(self, qwen3_tokenizer):
        messages = [{"role": "user", "content": "What is the weather in Paris?"}]
        prompt_ids, base_ids = IncrementalPromptRenderer.render_initial(
            qwen3_tokenizer, messages, tools=[WEATHER_TOOL]
        )
        canonical_full = apply_chat_template(
            qwen3_tokenizer,
            messages,
            tools=[WEATHER_TOOL],
            add_generation_prompt=True,
            tokenize=True,
        )
        canonical_base = apply_chat_template(
            qwen3_tokenizer,
            messages,
            tools=[WEATHER_TOOL],
            add_generation_prompt=False,
            tokenize=True,
        )
        assert prompt_ids == canonical_full
        assert base_ids == canonical_base

    def test_single_tool_call_round(self, qwen3_tokenizer):
        messages = [{"role": "user", "content": "What is the weather in Paris?"}]
        _, base_ids = IncrementalPromptRenderer.render_initial(
            qwen3_tokenizer, messages, tools=[WEATHER_TOOL]
        )

        asst_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        }
        tool_msg = {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "get_weather",
            "content": '{"temp": "20C"}',
        }
        delta = [asst_msg, tool_msg]
        messages.extend(delta)

        # Canonical full render
        canonical_prompt = apply_chat_template(
            qwen3_tokenizer,
            messages,
            tools=[WEATHER_TOOL],
            add_generation_prompt=True,
            tokenize=True,
        )

        # Incremental render
        rendered = IncrementalPromptRenderer.render_incremental(
            qwen3_tokenizer,
            base_ids,
            delta,
            tools=[WEATHER_TOOL],
        )
        assert rendered is not None
        incr_prompt, _ = rendered

        assert incr_prompt == canonical_prompt

    def test_multi_turn_tool_sequence_20_turns(self, qwen3_tokenizer):
        messages = [{"role": "user", "content": "Start multi-turn episode"}]
        _, current_base = IncrementalPromptRenderer.render_initial(
            qwen3_tokenizer, messages, tools=[WEATHER_TOOL, CALCULATOR_TOOL]
        )

        for turn in range(1, 20):
            asst_msg = {
                "role": "assistant",
                "content": f"<think>Step {turn} analysis</think>",
                "tool_calls": [
                    {
                        "id": f"call_{turn}",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": f'{{"location": "City_{turn}"}}',
                        },
                    }
                ],
            }
            tool_msg = {
                "role": "tool",
                "tool_call_id": f"call_{turn}",
                "name": "get_weather",
                "content": f'{{"temp": "{20 + turn}C"}}',
            }
            delta = [asst_msg, tool_msg]
            messages.extend(delta)

            # Canonical full history
            canonical_prompt = apply_chat_template(
                qwen3_tokenizer,
                messages,
                tools=[WEATHER_TOOL, CALCULATOR_TOOL],
                add_generation_prompt=True,
                tokenize=True,
            )

            # Incremental
            rendered = IncrementalPromptRenderer.render_incremental(
                qwen3_tokenizer,
                current_base,
                delta,
                tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            )
            assert rendered is not None
            incr_prompt, current_base = rendered

            assert incr_prompt == canonical_prompt, f"Mismatch at turn {turn}"

    def test_parallel_tool_calls(self, qwen3_tokenizer):
        messages = [{"role": "user", "content": "Compare Paris and London"}]
        _, base_ids = IncrementalPromptRenderer.render_initial(
            qwen3_tokenizer, messages, tools=[WEATHER_TOOL]
        )

        asst_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "London"}',
                    },
                },
            ],
        }
        tool_1 = {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "get_weather",
            "content": "20C",
        }
        tool_2 = {
            "role": "tool",
            "tool_call_id": "c2",
            "name": "get_weather",
            "content": "15C",
        }
        delta = [asst_msg, tool_1, tool_2]
        messages.extend(delta)

        canonical_prompt = apply_chat_template(
            qwen3_tokenizer,
            messages,
            tools=[WEATHER_TOOL],
            add_generation_prompt=True,
            tokenize=True,
        )

        rendered = IncrementalPromptRenderer.render_incremental(
            qwen3_tokenizer,
            base_ids,
            delta,
            tools=[WEATHER_TOOL],
        )
        assert rendered is not None
        incr_prompt, _ = rendered

        assert incr_prompt == canonical_prompt

    def test_conversational_and_custom_system_prompt(self, qwen25_tokenizer):
        messages = [
            {"role": "system", "content": "You are a specialized math assistant."},
            {"role": "user", "content": "Solve 2+2"},
        ]
        _, base_ids = IncrementalPromptRenderer.render_initial(
            qwen25_tokenizer, messages, tools=[CALCULATOR_TOOL]
        )

        delta1 = [
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "Now compute 10 * 10"},
        ]
        messages.extend(delta1)

        canonical_prompt = apply_chat_template(
            qwen25_tokenizer,
            messages,
            tools=[CALCULATOR_TOOL],
            add_generation_prompt=True,
            tokenize=True,
        )

        rendered = IncrementalPromptRenderer.render_incremental(
            qwen25_tokenizer,
            base_ids,
            delta1,
            tools=[CALCULATOR_TOOL],
        )
        assert rendered is not None
        incr_prompt, _ = rendered

        assert incr_prompt == canonical_prompt


class TestIncrementalConcatPromptRendering:
    def test_render_concat_child_tokens_identity(self, qwen3_tokenizer):
        msg1 = [{"role": "user", "content": "Weather in Paris?"}]
        t1 = apply_chat_template(
            qwen3_tokenizer,
            msg1,
            tools=[WEATHER_TOOL],
            add_generation_prompt=True,
            tokenize=True,
        )
        eos_id = qwen3_tokenizer.eos_token_id
        out_tokens = qwen3_tokenizer.encode(
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>',
            add_special_tokens=False,
        ) + [eos_id]

        resp = ModelResponse(
            input_tokens=t1,
            output_tokens=out_tokens,
            output_logprobs=[0.0] * len(out_tokens),
            output_versions=[0] * len(out_tokens),
            stop_reason="stop",
            tokenizer=qwen3_tokenizer,
        )
        asst = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        }
        parent = InteractionWithTokenLogpReward(
            messages=msg1,
            output_message_list=[asst],
            model_response=resp,
            chat_template_type="concat",
        )
        tool_msg = [
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "get_weather",
                "content": "20C",
            }
        ]

        # Full history concat
        concat_full, _, _ = _concat_prompt_token_ids_with_parent(
            message_list=tool_msg,
            parent=parent,
            tokenizer=qwen3_tokenizer,
            tools=[WEATHER_TOOL],
        )

        # Incremental child tokens
        child_tokens = IncrementalPromptRenderer.render_concat_child_tokens(
            qwen3_tokenizer,
            parent_output_messages=parent.output_message_list,
            message_list=tool_msg,
            tools=[WEATHER_TOOL],
        )
        assert child_tokens is not None
        parent_tokens = (
            parent.model_response.input_tokens
            + parent.model_response.output_tokens_without_stop
            + [eos_id]
        )
        concat_incr = parent_tokens + child_tokens

        assert concat_incr == concat_full


@pytest.mark.asyncio
class TestPreparePromptIntegration:
    async def test_prepare_prompt_hf_multi_turn_parity(self, qwen3_tokenizer):
        messages = [{"role": "user", "content": "Weather query"}]
        inter1 = InteractionWithTokenLogpReward(
            messages=list(messages), chat_template_type="hf"
        )
        p1 = await _prepare_prompt(
            tokenizer=qwen3_tokenizer,
            processor=None,
            tokenizer_messages=messages,
            concat_messages=messages,
            image_data=[],
            parent=None,
            chat_template_type="hf",
            tools=[WEATHER_TOOL],
            extra_body={},
            interaction=inter1,
        )
        assert inter1.prompt_token_ids == p1.input_ids
        assert inter1.prompt_base_token_ids is not None

        asst = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        }
        tool = {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "get_weather",
            "content": "20C",
        }
        messages.extend([asst, tool])
        inter2 = InteractionWithTokenLogpReward(
            messages=list(messages), chat_template_type="hf", parent=inter1
        )

        p2 = await _prepare_prompt(
            tokenizer=qwen3_tokenizer,
            processor=None,
            tokenizer_messages=messages,
            concat_messages=[tool],
            image_data=[],
            parent=inter1,
            chat_template_type="hf",
            tools=[WEATHER_TOOL],
            extra_body={},
            interaction=inter2,
        )

        canonical = apply_chat_template(
            qwen3_tokenizer,
            messages,
            tools=[WEATHER_TOOL],
            add_generation_prompt=True,
            tokenize=True,
        )
        assert p2.input_ids == canonical
        assert inter2.prompt_token_ids == canonical

    async def test_prepare_prompt_hf_fallback_without_parent_base_ids(
        self, qwen3_tokenizer
    ):
        messages = [{"role": "user", "content": "Weather query"}]
        inter1 = InteractionWithTokenLogpReward(
            messages=list(messages), chat_template_type="hf"
        )
        # Simulate legacy interaction with prompt_base_token_ids as None
        inter1.prompt_base_token_ids = None

        asst = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        }
        tool = {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "get_weather",
            "content": "20C",
        }
        messages.extend([asst, tool])
        inter2 = InteractionWithTokenLogpReward(
            messages=list(messages), chat_template_type="hf", parent=inter1
        )

        p2 = await _prepare_prompt(
            tokenizer=qwen3_tokenizer,
            processor=None,
            tokenizer_messages=messages,
            concat_messages=[tool],
            image_data=[],
            parent=inter1,
            chat_template_type="hf",
            tools=[WEATHER_TOOL],
            extra_body={},
            interaction=inter2,
        )

        canonical = apply_chat_template(
            qwen3_tokenizer,
            messages,
            tools=[WEATHER_TOOL],
            add_generation_prompt=True,
            tokenize=True,
        )
        assert p2.input_ids == canonical
        assert inter2.prompt_token_ids == canonical
        assert inter1.prompt_base_token_ids is not None

    async def test_prepare_prompt_fallback_on_unsupported_template(
        self, qwen3_tokenizer, monkeypatch
    ):
        monkeypatch.setattr(
            IncrementalPromptRenderer, "is_supported", lambda *args, **kwargs: False
        )

        messages = [{"role": "user", "content": "Query 1"}]
        inter1 = InteractionWithTokenLogpReward(
            messages=list(messages), chat_template_type="hf"
        )
        _ = await _prepare_prompt(
            tokenizer=qwen3_tokenizer,
            processor=None,
            tokenizer_messages=messages,
            concat_messages=messages,
            image_data=[],
            parent=None,
            chat_template_type="hf",
            tools=[WEATHER_TOOL],
            extra_body={},
            interaction=inter1,
        )

        asst = {"role": "assistant", "content": "Response 1"}
        user2 = {"role": "user", "content": "Query 2"}
        messages.extend([asst, user2])
        inter2 = InteractionWithTokenLogpReward(
            messages=list(messages), chat_template_type="hf", parent=inter1
        )

        p2 = await _prepare_prompt(
            tokenizer=qwen3_tokenizer,
            processor=None,
            tokenizer_messages=messages,
            concat_messages=[user2],
            image_data=[],
            parent=inter1,
            chat_template_type="hf",
            tools=[WEATHER_TOOL],
            extra_body={},
            interaction=inter2,
        )

        canonical = apply_chat_template(
            qwen3_tokenizer,
            messages,
            tools=[WEATHER_TOOL],
            add_generation_prompt=True,
            tokenize=True,
        )
        assert p2.input_ids == canonical
