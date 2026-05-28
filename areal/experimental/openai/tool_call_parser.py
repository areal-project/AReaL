# SPDX-License-Identifier: Apache-2.0

import traceback
import uuid
from types import SimpleNamespace
from typing import Any

from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from areal.utils import logging

logger = logging.getLogger("ToolCallParser")

_SGLANG_TO_VLLM_TOOL_PARSER: dict[str, str] = {
    "qwen": "qwen3_xml",
    "qwen25": "qwen3_xml",
    "qwen3": "qwen3_xml",
    "qwen3_xml": "qwen3_xml",
    "qwen3_coder": "qwen3_coder",
    "hermes": "hermes",
    "llama3": "llama3_json",
    "llama3_json": "llama3_json",
    "llama4_json": "llama4_json",
    "mistral": "mistral",
    "openai": "openai",
    "deepseek_v3": "deepseek_v3",
}


def _detect_think_and_return_ori_think(
    text: str, think_start_token: str, think_end_token: str
) -> tuple[str, str]:
    """
    return think text(with <think> and </think>) and normal text
    """
    # This code is copies from sglang https://github.com/sgl-project/sglang/blob/cb30d056e3bc1b2f70fa7c00e0844cfe15716d65/python/sglang/srt/parser/reasoning_parser.py#L18
    in_reasoning = think_start_token in text

    if not in_reasoning:
        return "", text

    # The text is considered to be in a reasoning block.
    processed_text = text.replace(think_start_token, "")

    if think_end_token not in processed_text:
        # Assume reasoning was truncated before `</think>` token
        return think_start_token + processed_text, ""

    # Extract reasoning content
    splits = processed_text.split(think_end_token, maxsplit=1)
    reasoning_text = splits[0]
    normal_text = splits[1]

    return think_start_token + reasoning_text + think_end_token, normal_text


def _process_tool_calls_sglang(
    text: str,
    tools: list[Any],
    tool_call_parser: str,
    reasoning_parser: str,
    finish_reason: str,
    use_responses: bool = False,
) -> tuple[
    list[ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall] | None,
    str,
    str,
]:
    from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
    from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    if use_responses:
        tools = [
            SglTool(
                type=tool["type"],
                function=SglFunction(
                    name=tool.get("name"),
                    description=tool.get("description"),
                    parameters=tool.get("parameters"),
                ),
            )
            for tool in tools
        ]
    else:
        tools = [
            SglTool(type=tool["type"], function=SglFunction(**tool["function"]))
            for tool in tools
        ]

    parser_p = FunctionCallParser(tools, tool_call_parser)
    reasoning_parser_p = ReasoningParser(reasoning_parser)

    reasoning_text, content_text = _detect_think_and_return_ori_think(
        text,
        reasoning_parser_p.detector.think_start_token,
        reasoning_parser_p.detector.think_end_token,
    )

    if parser_p.has_tool_call(content_text):
        if finish_reason == "stop":
            finish_reason = "tool_calls"
        try:
            content_text, call_info_list = parser_p.parse_non_stream(content_text)

            if use_responses:
                tool_calls = [
                    ResponseFunctionToolCall(
                        type="function_call",
                        id=f"fc-{uuid.uuid4().hex[:24]}",
                        call_id=f"call_{uuid.uuid4().hex[:24]}",
                        name=call_info.name,
                        arguments=call_info.parameters,
                        status="completed",
                    )
                    for call_info in call_info_list
                ]
            else:
                tool_calls = [
                    ChatCompletionMessageFunctionToolCall(
                        type="function",
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        function=Function(
                            name=call_info.name, arguments=call_info.parameters
                        ),
                    )
                    for call_info in call_info_list
                ]

            return tool_calls, reasoning_text + content_text, finish_reason
        except Exception as e:
            logger.error(f"Tool call parsing error: {e}")
            traceback.print_exc()
            return None, text, finish_reason

    return None, text, finish_reason


def _process_tool_calls_vllm(
    text: str,
    tools: list[Any],
    tool_call_parser: str,
    reasoning_parser: str,
    finish_reason: str,
    use_responses: bool = False,
    tokenizer: Any = None,
) -> tuple[
    list[ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall] | None,
    str,
    str,
]:
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    # Use vllm's reasoning parser to get the think start/end tokens,
    # mirroring the sglang path which uses ReasoningParser.detector tokens.
    if tokenizer is not None and reasoning_parser:
        try:
            reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser(
                reasoning_parser
            )
            reasoning_parser_inst = reasoning_parser_cls(tokenizer)
            if hasattr(reasoning_parser_inst, "start_token") and hasattr(
                reasoning_parser_inst, "end_token"
            ):
                reasoning_text, content_text = _detect_think_and_return_ori_think(
                    text,
                    reasoning_parser_inst.start_token,
                    reasoning_parser_inst.end_token,
                )
            else:
                reasoning_text, content_text = "", text
        except Exception as e:
            logger.warning(
                "Failed to initialize vLLM reasoning parser '%s': %s. "
                "Skipping reasoning extraction.",
                reasoning_parser,
                e,
            )
            reasoning_text, content_text = "", text
    else:
        reasoning_text, content_text = "", text

    vllm_name = _SGLANG_TO_VLLM_TOOL_PARSER.get(tool_call_parser, tool_call_parser)
    try:
        tool_parser_cls = ToolParserManager.get_tool_parser(vllm_name)
    except KeyError:
        logger.warning(
            "vLLM tool parser '%s' (mapped from '%s') not found; skipping tool call parsing.",
            vllm_name,
            tool_call_parser,
        )
        return None, text, finish_reason

    if tokenizer is None:
        logger.warning(
            "vLLM tool parser requires a tokenizer but none was provided; skipping tool call parsing."
        )
        return None, text, finish_reason

    tool_parser = tool_parser_cls(tokenizer)
    request = SimpleNamespace(
        tools=tools,
        tool_choice=None,
        skip_special_tokens=True,
    )

    try:
        tool_call_info = tool_parser.extract_tool_calls(content_text, request)
    except Exception as e:
        logger.error("vLLM tool call parsing error: %s", e)
        traceback.print_exc()
        return None, text, finish_reason

    if not tool_call_info.tools_called:
        return None, text, finish_reason

    if finish_reason == "stop":
        finish_reason = "tool_calls"

    remaining_content = tool_call_info.content or ""

    if use_responses:
        result_tool_calls = [
            ResponseFunctionToolCall(
                type="function_call",
                id=f"fc-{uuid.uuid4().hex[:24]}",
                call_id=f"call_{uuid.uuid4().hex[:24]}",
                name=tc.function.name,
                arguments=tc.function.arguments,
                status="completed",
            )
            for tc in tool_call_info.tool_calls
        ]
    else:
        result_tool_calls = [
            ChatCompletionMessageFunctionToolCall(
                type="function",
                id=f"call_{uuid.uuid4().hex[:24]}",
                function=Function(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ),
            )
            for tc in tool_call_info.tool_calls
        ]

    return result_tool_calls, reasoning_text + remaining_content, finish_reason


def process_tool_calls(
    text: str,
    tools: list[Any],
    tool_call_parser: str,
    reasoning_parser: str,
    finish_reason: str,
    use_responses: bool = False,
    tokenizer: Any = None,
) -> tuple[
    list[ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall] | None,
    str,
    str,
]:
    """Process tool calls in the response"""
    try:
        return _process_tool_calls_sglang(
            text,
            tools,
            tool_call_parser,
            reasoning_parser,
            finish_reason,
            use_responses,
        )
    except ModuleNotFoundError:
        pass

    try:
        return _process_tool_calls_vllm(
            text,
            tools,
            tool_call_parser,
            reasoning_parser,
            finish_reason,
            use_responses,
            tokenizer=tokenizer,
        )
    except ModuleNotFoundError:
        pass

    logger.warning(
        "Neither sglang nor vllm is installed; skipping tool call parsing. Install one of them for tool call support."
    )
    return None, text, finish_reason
