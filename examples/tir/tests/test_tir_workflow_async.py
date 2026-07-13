# SPDX-License-Identifier: Apache-2.0

"""Focused async-path regression tests for TIR workflow tool execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import tir_workflow as tir_workflow_module
from tir_workflow import TIRWorkflow
from tool_manager import ToolCallStatus

from areal.api.io_struct import ModelResponse


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


@pytest.mark.asyncio
async def test_tir_workflow_invokes_aexecute(monkeypatch):
    tokenizer = FakeTokenizer()
    workflow = object.__new__(TIRWorkflow)
    workflow.tokenizer = tokenizer
    workflow.max_turns = 3
    workflow.max_length = 512
    workflow.start_markers = ["```python"]
    workflow.end_markers = ["```"]
    workflow.async_reward_fn = lambda *args, **kwargs: _return_reward()

    async def _return_reward():
        return 1.0

    async def aexecute_tool_call(text: str):
        aexecute_calls.append(text)
        return "async-tool-result", ToolCallStatus.SUCCESS

    def execute_tool_call(text: str):
        raise AssertionError("sync execute_tool_call should not be used")

    aexecute_calls: list[str] = []

    workflow.tool_manager = SimpleNamespace(
        aexecute_tool_call=aexecute_tool_call,
        execute_tool_call=execute_tool_call,
    )

    responses = iter(
        [
            ModelResponse(
                output_tokens=tokenizer.encode("thinking```python"),
                output_logprobs=[0.0] * len("thinking```python"),
                output_versions=[1] * len("thinking```python"),
                stop_reason="stop",
            ),
            ModelResponse(
                output_tokens=tokenizer.encode("\nprint(1)\n```"),
                output_logprobs=[0.0] * len("\nprint(1)\n```"),
                output_versions=[1] * len("\nprint(1)\n```"),
                stop_reason="stop",
            ),
            ModelResponse(
                output_tokens=[tokenizer.eos_token_id],
                output_logprobs=[0.0],
                output_versions=[1],
                stop_reason="stop",
            ),
        ]
    )

    async def fake_generate_response(
        engine, context_ids, max_len, waiting_for_tool_start
    ):
        del engine, context_ids, max_len, waiting_for_tool_start
        response = next(responses)
        return response, response.stop_reason

    stats_calls: list[dict[str, int]] = []

    class FakeStatsTracker:
        def scalar(self, **kwargs):
            stats_calls.append(kwargs)

    monkeypatch.setattr(workflow, "_generate_response", fake_generate_response)
    monkeypatch.setattr(
        tir_workflow_module.stats_tracker, "get", lambda scope: FakeStatsTracker()
    )
    monkeypatch.setattr(
        tir_workflow_module.workflow_context, "stat_scope", lambda: "test"
    )

    result = await workflow._multi_round_response(
        engine=None, prompt_ids=tokenizer.encode("prompt"), data={}
    )

    assert result["rewards"].item() == 1.0
    assert stats_calls == [{"tool_call_count": 1, "tool_success_count": 1}]
    assert aexecute_calls == ["```python\nprint(1)\n```"]
