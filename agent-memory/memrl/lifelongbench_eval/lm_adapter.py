from __future__ import annotations

"""
LLB 语言模型适配器：复用本地 memp OpenAILLM，适配 LifelongAgentBench 接口。

关键点：
- 将 LLB 的 ChatHistory 转换为 OpenAI Chat API 的 messages 列表
- 支持可选的 system_prompt 作为第一条消息
- 将 LLB 的推理参数转换为本地 provider 接口（如 max_completion_tokens -> max_tokens）
- 捕获异常并映射到 LLB 的异常类型
"""

from typing import Any, Mapping, Sequence

import os
import sys


def _ensure_llb_sys_path() -> None:
    """确保可以导入 LifelongAgentBench-main 下的 src 包。"""
    # Python 3.10 兼容：为 enum.StrEnum 提供兜底实现
    try:
        import enum as _enum

        if not hasattr(_enum, "StrEnum"):

            class _StrEnum(str, _enum.Enum):
                pass

            _enum.StrEnum = _StrEnum  # type: ignore[attr-defined]
        import typing as _typing

        if not hasattr(_typing, "reveal_type"):

            def _noop_reveal_type(x):
                return x

            _typing.reveal_type = _noop_reveal_type  # type: ignore[attr-defined]
        if not hasattr(_typing, "Self"):
            _typing.Self = object  # type: ignore[attr-defined]
    except Exception:
        pass
    root = os.path.abspath(os.path.join(os.getcwd(), "LifelongAgentBench-main"))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_llb_sys_path()

from src.language_models.language_model import LanguageModel  # type: ignore
from src.typings import (  # type: ignore
    Role,
    ChatHistory,
    ChatHistoryItem,
    LanguageModelContextLimitException,
    LanguageModelOutOfMemoryException,
    LanguageModelUnknownException,
)


class MempOpenAIAdapter(LanguageModel):
    """将本地 OpenAILLM 适配为 LLB 的 LanguageModel。"""

    def __init__(
        self,
        provider,
        role_dict: Mapping[str, str] | None = None,
    ) -> None:
        # LLB 要求传入 role 映射，用于渲染 chat_history 字符串。
        role_dict = role_dict or {Role.USER: "user", Role.AGENT: "assistant"}  # type: ignore[index]
        super().__init__(role_dict)
        self._provider = provider

    def _convert_batch(
        self, batch_chat_history: Sequence[ChatHistory], system_prompt: str
    ) -> list[list[dict[str, str]]]:
        messages_list: list[list[dict[str, str]]] = []
        for chat in batch_chat_history:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            # 将 LLB 的 ChatHistory 转为 OpenAI messages
            for i in range(chat.get_value_length()):
                item = chat.get_item_deep_copy(i)
                role = "user" if item.role == Role.USER else "assistant"
                messages.append({"role": role, "content": item.content})
            messages_list.append(messages)
        return messages_list

    def _map_infer_args(
        self, inference_config_dict: Mapping[str, Any]
    ) -> dict[str, Any]:
        """参数名映射：LLB 用 max_completion_tokens，本地用 max_tokens。"""
        kwargs = dict(inference_config_dict)
        if "max_completion_tokens" in kwargs and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        return kwargs

    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        try:
            messages_batch = self._convert_batch(batch_chat_history, system_prompt)
            kwargs = self._map_infer_args(inference_config_dict)
            outputs: list[ChatHistoryItem] = []
            for messages in messages_batch:
                content = self._provider.generate(messages, **kwargs) or ""
                outputs.append(ChatHistoryItem(role=Role.AGENT, content=content))
            return outputs
        except Exception as e:  # 细分异常映射（保守策略）
            msg = str(e)
            if "context" in msg and "length" in msg:
                raise LanguageModelContextLimitException(msg)
            if "OOM" in msg or "out of memory" in msg.lower():
                raise LanguageModelOutOfMemoryException(msg)
            raise LanguageModelUnknownException(msg)
