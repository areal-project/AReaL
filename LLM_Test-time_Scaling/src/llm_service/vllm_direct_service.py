"""Direct vLLM service implementation using chat-template -> completions flow."""

import asyncio
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer

from .base import LLMResponse, LLMService, Message


class VLLMDirectService(LLMService):
    """Direct vLLM client that applies chat templates client-side.

    This service converts chat messages into a single prompt via
    ``tokenizer.apply_chat_template`` and sends the prompt to the vLLM
    OpenAI-compatible ``/v1/completions`` endpoint.
    """

    _tokenizer_cache: Dict[str, Any] = {}

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, api_key, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.api_base = api_base
        self.tokenizer_name = tokenizer_name
        self._tokenizer = self._load_tokenizer(tokenizer_name, model_name)

        configured_context_limit = kwargs.get("context_limit")
        if isinstance(configured_context_limit, int) and configured_context_limit > 0:
            self.context_limit = configured_context_limit
        else:
            context_limit_env = os.getenv("LLM_CONTEXT_LIMIT_TOKENS")
            if context_limit_env and context_limit_env.isdigit() and int(context_limit_env) > 0:
                self.context_limit = int(context_limit_env)
            else:
                self.context_limit = 131072

        configured_max_tokens = kwargs.get("max_tokens")
        if isinstance(configured_max_tokens, int) and configured_max_tokens > 0:
            self.default_max_tokens: Optional[int] = configured_max_tokens
        else:
            self.default_max_tokens = None

    def _load_tokenizer(self, tokenizer_name: Optional[str], model_name: str) -> Optional[Any]:
        candidates: List[str] = []
        if tokenizer_name:
            candidates.append(tokenizer_name)
        candidates.append(model_name)
        if model_name.startswith("openai/"):
            stripped = model_name[len("openai/") :]
            candidates.append(stripped)

        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if candidate in self._tokenizer_cache:
                return self._tokenizer_cache[candidate]
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
                self._tokenizer_cache[candidate] = tokenizer
                return tokenizer
            except Exception:
                continue
        return None

    def _format_prompt_fallback(self, messages: List[Message]) -> str:
        lines = []
        for msg in messages:
            role = msg.role.upper()
            lines.append(f"{role}: {msg.content}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    def _apply_chat_template(self, messages: List[Message]) -> str:
        chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return self._format_prompt_fallback(messages)

    def _resolve_max_tokens(self, prompt: str, requested_max_tokens: Optional[int]) -> int:
        if isinstance(requested_max_tokens, int) and requested_max_tokens > 0:
            return requested_max_tokens

        safety_margin = 1000
        available_tokens: Optional[int] = None

        if self._tokenizer is not None:
            try:
                encoded = self._tokenizer(prompt, add_special_tokens=False)
                prompt_tokens = len(encoded.get("input_ids", []))
                available_tokens = max(1, self.context_limit - prompt_tokens - safety_margin)
            except Exception:
                available_tokens = None

        if isinstance(self.default_max_tokens, int) and self.default_max_tokens > 0:
            if available_tokens is None:
                return self.default_max_tokens
            return max(1, min(self.default_max_tokens, available_tokens))

        if available_tokens is not None:
            return available_tokens

        # Avoid vLLM/OpenAI-completions defaulting to 16 tokens when max_tokens is omitted.
        return 4096

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        prompt = self._apply_chat_template(messages)
        final_max_tokens = self._resolve_max_tokens(prompt, max_tokens)

        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        if "top_k" in kwargs:
            extra_body["top_k"] = kwargs.pop("top_k")

        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": final_max_tokens,
            **kwargs,
        }
        if extra_body:
            call_kwargs["extra_body"] = extra_body

        response = await self.client.completions.create(**call_kwargs)

        content = response.choices[0].text or ""
        reasoning_content = None

        if "qwen3" in self.model_name.lower() and "</think>" in content:
            reasoning_content = content.split("</think>")[0] + "</think>"
            content = content[content.rfind("</think>") + len("</think>") :]

        if (
            "gpt-oss" in self.model_name.lower()
            and "<|start|>assistant<|channel|>final<|message|>" in content
        ):
            marker = "<|start|>assistant<|channel|>final<|message|>"
            reasoning_content = content.split(marker)[0] + marker
            content = content.split(marker, 1)[1]

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "model": self.model_name,
            "api_base": self.api_base or "default",
        }

        return LLMResponse(
            content=content,
            model=getattr(response, "model", self.model_name),
            usage=usage,
            raw_response=response,
            finish_reason=response.choices[0].finish_reason,
            reasoning_content=reasoning_content,
        )

    async def generate_batch(
        self,
        messages_batch: List[List[Message]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> List[LLMResponse]:
        tasks = [
            self.generate(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
            for messages in messages_batch
        ]
        return await asyncio.gather(*tasks)
