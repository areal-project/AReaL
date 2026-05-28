from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolParam

from areal.experimental.openai import tool_call_parser as parser_module

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "A powerful web search tool for accessing a vast range of external information beyond its training data. "
                "Use this tool when you need detailed information on highly specific, specialized, or niche topics, or when you need to verify information and fact-check claims by finding authoritative sources. "
                "It helps you answer complex questions that require deep knowledge or specific external data. The input should be a clear search query designed to find specific knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A precise search query for information retrieval.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "access",
            "description": (
                "Invokes the Jina AI Reader engine to intelligently access and parse a URL. "
                "This tool takes a webpage URL as input and returns its main article content in a clean Markdown format. "
                "Use this to perform a 'deep dive' on the most relevant link found via web_search to extract detailed evidence and data needed to answer a question. "
                "It automatically ignores advertisements and boilerplate code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the webpage to read.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

TEXT = (
    '<think>\nOkay, so the user is asking whether the director of "Scary Movie" and the director of "The Preacher\'s Wife" are from the same country. Let me think about how to approach this.\n\n'
    'First, I need to confirm the countries of these directors. I remember that "Scary Movie" is directed by Joe Anderson, and "The Preacher\'s Wife" is directed by David Fincher. Wait, but actually, "The Preacher\'s Wife" is directed by Christopher Nolan and David Fincher? No, I think I mixed up. Let me check my memory. \n\n'
    "Wait, no. The Preacher's Wife is a movie directed by Christopher Nolan. And David Fincher directed The Preacher. So the directors are different. Then the user is asking if they are both from the same country. The answer would be no, because the two directors are from different countries. \n\n"
    'But maybe I should verify this to be sure. Since I can use the web search function, I should use the "access" tool to get the URLs of the directors\' websites to confirm. So first, I\'ll search for "director of Scary Movie country" and "director of The Preacher\'s Wife country" to get precise data. Then, analyze the results to see if they have the same country affiliations.\n</think>\n\n'
    '<tool_call>\n{"name": "search", "arguments": {"query": "director of Scary Movie country"}}\n</tool_call>\n\n'
    '<tool_call>\n{"name": "search", "arguments": {"query": "director of The Preacher\'s Wife country"}}\n</tool_call><|im_end|>'
)

TEXT_WITH_TOOL_CALL_IN_THINKING = (
    '<think>\nOkay, so the user is asking whether the director of "Scary Movie" and the director of "The Preacher\'s Wife" are from the same country. Let me think about how to approach this.\n\n'
    'First, I need to confirm the countries of these directors. I remember that "Scary Movie" is directed by Joe Anderson, and "The Preacher\'s Wife" is directed by David Fincher. Wait, but actually, "The Preacher\'s Wife" is directed by Christopher Nolan and David Fincher? No, I think I mixed up. Let me check my memory. \n\n'
    'Wait, no. The Preacher\'s Wife is a movie directed by Christopher Nolan. And David Fincher directed The Preacher. <tool_call>\n{"name": "search", "arguments": {"query": "aaaa"}}\n</tool_call>\n\n So the directors are different. Then the user is asking if they are both from the same country. The answer would be no, because the two directors are from different countries. \n\n'
    'But maybe I should verify this to be sure. Since I can use the web search function, I should use the "access" tool to get the URLs of the directors\' websites to confirm. So first, I\'ll search for "director of Scary Movie country" and "director of The Preacher\'s Wife country" to get precise data. Then, analyze the results to see if they have the same country affiliations.\n</think>\n\n'
    '<tool_call>\n{"name": "search", "arguments": {"query": "director of Scary Movie country"}}\n</tool_call>\n\n'
    '<tool_call>\n{"name": "search", "arguments": {"query": "director of The Preacher\'s Wife country"}}\n</tool_call><|im_end|>'
)


def _assert_tool_calls(tool_calls, new_text: str, new_finish_reason: str) -> None:
    assert new_finish_reason == "tool_calls"
    assert tool_calls is not None, "Tool calls should be detected and returned"
    assert len(tool_calls) == 2, "Two tool calls should be parsed from the text"
    assert isinstance(tool_calls[0], ChatCompletionMessageToolCall)
    assert isinstance(tool_calls[1], ChatCompletionMessageToolCall)
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search"
    assert tool_calls[1].function.name == "search"
    assert (
        tool_calls[0].function.arguments
        == '{"query": "director of Scary Movie country"}'
    )
    assert (
        tool_calls[1].function.arguments
        == '{"query": "director of The Preacher\'s Wife country"}'
    )


def _run_process_tool_calls(text: str):
    return parser_module.process_tool_calls(
        text=text,
        tools=TOOLS,
        tool_call_parser="qwen25",
        reasoning_parser="qwen3",
        finish_reason="tool_calls",
        use_responses=False,
        tokenizer=object(),
    )


@pytest.mark.sglang
def test_process_tool_calls_qwen25_chat_completions_sglang():
    pytest.importorskip(
        "sglang.srt.function_call.function_call_parser",
        reason="sglang is required for sglang parser tests",
    )
    pytest.importorskip(
        "sglang.srt.parser.reasoning_parser",
        reason="sglang is required for sglang parser tests",
    )

    tool_calls, new_text, new_finish_reason = _run_process_tool_calls(TEXT)

    _assert_tool_calls(tool_calls, new_text, new_finish_reason)
    assert "<tool_call>" not in new_text


@pytest.mark.sglang
def test_process_tool_calls_qwen25_chat_completions_with_tool_call_in_thinking_sglang():
    pytest.importorskip(
        "sglang.srt.function_call.function_call_parser",
        reason="sglang is required for sglang parser tests",
    )
    pytest.importorskip(
        "sglang.srt.parser.reasoning_parser",
        reason="sglang is required for sglang parser tests",
    )

    tool_calls, new_text, new_finish_reason = _run_process_tool_calls(
        TEXT_WITH_TOOL_CALL_IN_THINKING
    )

    _assert_tool_calls(tool_calls, new_text, new_finish_reason)
    assert "<tool_call>" in new_text


def _raise_module_not_found(*args, **kwargs):
    raise ModuleNotFoundError


class FakeReasoningParser:
    start_token = "<think>"
    end_token = "</think>"

    def __init__(self, tokenizer, *args, **kwargs):
        pass


def _patch_vllm_parsers(monkeypatch):
    tool_parsers = pytest.importorskip(
        "vllm.tool_parsers",
        reason="vllm is required for vllm parser tests",
    )
    reasoning_mod = pytest.importorskip(
        "vllm.reasoning",
        reason="vllm is required for vllm parser tests",
    )
    monkeypatch.setattr(
        parser_module, "_process_tool_calls_sglang", _raise_module_not_found
    )
    monkeypatch.setattr(
        reasoning_mod.ReasoningParserManager,
        "get_reasoning_parser",
        staticmethod(lambda name: FakeReasoningParser),
    )
    return tool_parsers


@pytest.mark.vllm
def test_process_tool_calls_qwen25_chat_completions_vllm(
    monkeypatch: pytest.MonkeyPatch,
):
    tool_parsers = _patch_vllm_parsers(monkeypatch)

    class FakeParser:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def extract_tool_calls(self, content_text, request):
            assert request.skip_special_tokens is True
            return SimpleNamespace(
                tools_called=True,
                content=content_text.replace(
                    '<tool_call>\n{"name": "search", "arguments": {"query": "director of Scary Movie country"}}\n</tool_call>\n\n',
                    "",
                ).replace(
                    '<tool_call>\n{"name": "search", "arguments": {"query": "director of The Preacher\'s Wife country"}}\n</tool_call><|im_end|>',
                    "",
                ),
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="search",
                            arguments='{"query": "director of Scary Movie country"}',
                        )
                    ),
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="search",
                            arguments='{"query": "director of The Preacher\'s Wife country"}',
                        )
                    ),
                ],
            )

    monkeypatch.setattr(
        tool_parsers.ToolParserManager,
        "get_tool_parser",
        staticmethod(lambda name: FakeParser),
    )

    tool_calls, new_text, new_finish_reason = _run_process_tool_calls(TEXT)

    _assert_tool_calls(tool_calls, new_text, new_finish_reason)
    assert "<tool_call>" not in new_text


@pytest.mark.vllm
def test_process_tool_calls_qwen25_chat_completions_with_tool_call_in_thinking_vllm(
    monkeypatch: pytest.MonkeyPatch,
):
    tool_parsers = _patch_vllm_parsers(monkeypatch)

    class FakeParser:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def extract_tool_calls(self, content_text, request):
            assert request.skip_special_tokens is True
            return SimpleNamespace(
                tools_called=True,
                content=content_text.replace(
                    '<tool_call>\n{"name": "search", "arguments": {"query": "director of Scary Movie country"}}\n</tool_call>\n\n',
                    "",
                ).replace(
                    '<tool_call>\n{"name": "search", "arguments": {"query": "director of The Preacher\'s Wife country"}}\n</tool_call><|im_end|>',
                    "",
                ),
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="search",
                            arguments='{"query": "director of Scary Movie country"}',
                        )
                    ),
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="search",
                            arguments='{"query": "director of The Preacher\'s Wife country"}',
                        )
                    ),
                ],
            )

    monkeypatch.setattr(
        tool_parsers.ToolParserManager,
        "get_tool_parser",
        staticmethod(lambda name: FakeParser),
    )

    tool_calls, new_text, new_finish_reason = _run_process_tool_calls(
        TEXT_WITH_TOOL_CALL_IN_THINKING
    )

    _assert_tool_calls(tool_calls, new_text, new_finish_reason)
    assert "<tool_call>" in new_text
