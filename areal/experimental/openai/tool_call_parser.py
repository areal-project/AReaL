# SPDX-License-Identifier: Apache-2.0

import json
import re
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

_QWEN3_CODER_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>",
    re.DOTALL,
)
_QWEN3_CODER_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[^>\s]+)>\s*(?P<body>.*?)\s*</function>",
    re.DOTALL,
)
_QWEN3_CODER_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[^>\s]+)>(?P<value>.*?)</parameter>",
    re.DOTALL,
)

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


def _iter_tool_definitions(tools: list[Any]) -> list[dict[str, Any]]:
    """Return OpenAI chat/responses tool definitions in a common shape."""
    tool_defs: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if isinstance(tool.get("function"), dict):
            tool_defs.append(tool["function"])
        elif tool.get("type") == "function":
            tool_defs.append(tool)
    return tool_defs


def _tool_argument_schemas(tools: list[Any]) -> dict[str, dict[str, dict[str, Any]]]:
    schemas: dict[str, dict[str, dict[str, Any]]] = {}
    for tool_def in _iter_tool_definitions(tools):
        name = tool_def.get("name")
        parameters = tool_def.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")
        if isinstance(properties, dict):
            schemas[name] = {
                key: value
                for key, value in properties.items()
                if isinstance(value, dict)
            }
    return schemas


def _clean_qwen3_coder_parameter(raw_value: str) -> str:
    if raw_value.startswith("\n"):
        raw_value = raw_value[1:]
    if raw_value.endswith("\n"):
        raw_value = raw_value[:-1]
    return raw_value


def _coerce_qwen3_coder_parameter(
    value: str,
    schema: dict[str, Any] | None,
) -> Any:
    if not schema:
        return value

    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), None)
    stripped = value.strip()

    try:
        if typ == "boolean":
            lowered = stripped.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        if typ == "integer":
            return int(stripped)
        if typ == "number":
            return float(stripped)
        if typ in {"array", "object"}:
            return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value

    return value


def _build_tool_call(
    name: str,
    arguments: str,
    use_responses: bool,
) -> ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall:
    if use_responses:
        return ResponseFunctionToolCall(
            type="function_call",
            id=f"fc-{uuid.uuid4().hex[:24]}",
            call_id=f"call_{uuid.uuid4().hex[:24]}",
            name=name,
            arguments=arguments,
            status="completed",
        )
    return ChatCompletionMessageFunctionToolCall(
        type="function",
        id=f"call_{uuid.uuid4().hex[:24]}",
        function=Function(name=name, arguments=arguments),
    )


def _process_tool_calls_qwen3_coder_xml(
    text: str,
    tools: list[Any],
    finish_reason: str,
    use_responses: bool = False,
) -> tuple[
    list[ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall] | None,
    str,
    str,
]:
    """Parse Qwen3-Coder XML tool calls with ``<parameter=...>`` tags.

    SGLang's qwen3_coder parser recognizes the tool name in this format but
    can return empty arguments for Claude Code style parameters.  That makes
    Anthropic tool_use.input become ``{}``, so Claude Code rejects calls like
    Bash/Read/Write as missing required parameters.
    """

    reasoning_text, content_text = _detect_think_and_return_ori_think(
        text, "<think>", "</think>"
    )
    arg_schemas = _tool_argument_schemas(tools)

    tool_calls: list[ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall]
    tool_calls = []
    remove_spans: list[tuple[int, int]] = []

    for tool_match in _QWEN3_CODER_TOOL_CALL_RE.finditer(content_text):
        block_calls: list[
            ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall
        ] = []
        body = tool_match.group("body")
        for fn_match in _QWEN3_CODER_FUNCTION_RE.finditer(body):
            tool_name = fn_match.group("name")
            fn_body = fn_match.group("body")
            # A parameter value that itself contains a literal "</parameter>"
            # makes the non-greedy regex truncate at the wrong tag, silently
            # producing wrong arguments. When opening/closing tags are unbalanced
            # the block is ambiguous, so bail out and let the fallback parser run.
            if fn_body.count("<parameter=") != fn_body.count("</parameter>"):
                return None, text, finish_reason
            args: dict[str, Any] = {}
            for param_match in _QWEN3_CODER_PARAMETER_RE.finditer(fn_body):
                param_name = param_match.group("name")
                raw_param_value = param_match.group("value")
                # A balanced count of tags still hides nested pairs, e.g. a
                # value that contains its own "<parameter=...>...</parameter>".
                # The non-greedy regex truncates the outer value at the inner
                # closing tag, so treat any parameter marker inside the value as
                # ambiguous and fall back to the backend parser.
                if (
                    "<parameter=" in raw_param_value
                    or "</parameter>" in raw_param_value
                ):
                    return None, text, finish_reason
                param_value = _clean_qwen3_coder_parameter(raw_param_value)
                args[param_name] = _coerce_qwen3_coder_parameter(
                    param_value,
                    arg_schemas.get(tool_name, {}).get(param_name),
                )

            # No parsed arguments means either a genuinely empty block or a
            # truncated/malformed one. Skip it so process_tool_calls falls back
            # to the sglang parser instead of committing empty arguments.
            if not args:
                continue

            arguments = json.dumps(args, ensure_ascii=False)
            block_calls.append(_build_tool_call(tool_name, arguments, use_responses))

        if block_calls:
            tool_calls.extend(block_calls)
            remove_spans.append(tool_match.span())

    if not tool_calls:
        return None, text, finish_reason

    if finish_reason == "stop":
        finish_reason = "tool_calls"

    chunks: list[str] = []
    last = 0
    for start, end in remove_spans:
        chunks.append(content_text[last:start])
        last = end
    chunks.append(content_text[last:])
    cleaned_text = "".join(chunks).replace("<|im_end|>", "")

    return tool_calls, reasoning_text + cleaned_text, finish_reason


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
    if tool_call_parser == "qwen3_coder":
        tool_calls, output_text, finish_reason = _process_tool_calls_qwen3_coder_xml(
            text,
            tools,
            finish_reason,
            use_responses,
        )
        if tool_calls is not None:
            return tool_calls, output_text, finish_reason

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
