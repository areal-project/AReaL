"""Raw vLLM service that manually constructs the prompt in Qwen chat format.

Unlike VLLMDirectService which uses tokenizer.apply_chat_template(), this
service manually builds the prompt string using <|im_start|>/<|im_end|>
markers — exactly matching the format used in training data and in
test_training_data_on_vllm.py.  The prompt is sent to /v1/completions
via requests.post().
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMResponse, LLMService, Message


class VLLMRawService(LLMService):
    """Raw vLLM client that manually formats the prompt.

    Constructs the chat prompt by concatenating messages with
    ``<|im_start|>``/``<|im_end|>`` markers (Qwen / ChatML format)
    and posts directly to ``/v1/completions`` using ``requests``,
    identical to what ``test_training_data_on_vllm.py`` does.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, api_key, **kwargs)
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        self.request_timeout = int(kwargs.get("request_timeout", 600))

        configured_max_tokens = kwargs.get("max_tokens")
        if isinstance(configured_max_tokens, int) and configured_max_tokens > 0:
            self.default_max_tokens: Optional[int] = configured_max_tokens
        else:
            self.default_max_tokens = None

    # ------------------------------------------------------------------
    # Prompt formatting — matches training data format exactly
    # ------------------------------------------------------------------
    @staticmethod
    def _format_prompt(messages: List[Message]) -> str:
        """Build a raw prompt string in Qwen / ChatML format.

        Produces the same layout as the pre-formatted training text::

            <|im_start|>system
            {system_content}<|im_end|>
            <|im_start|>user
            {user_content}<|im_end|>
            <|im_start|>assistant
        """
        parts: List[str] = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n")
        # Add the generation prompt for the assistant turn
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    # ------------------------------------------------------------------
    # HTTP call — mirrors test_training_data_on_vllm.py exactly
    # ------------------------------------------------------------------
    def _post_completions(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Synchronous POST to /v1/completions (same as the test script)."""
        url = f"{self.api_base}/completions"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, json=payload, timeout=self.request_timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # LLMService interface
    # ------------------------------------------------------------------
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
        prompt = self._format_prompt(messages)

        logger.debug(
            "[VLLMRawService] === Model Input ===\n"
            "model=%s, temperature=%s, max_tokens=%s\n"
            "--- prompt start ---\n%s\n--- prompt end ---",
            self.model_name, temperature, max_tokens, prompt,
        )
        print(
            f"\n[VLLMRawService DEBUG] === Model Input ===\n"
            f"model={self.model_name}, temperature={temperature}, max_tokens={max_tokens}\n"
            f"--- prompt start ---\n{prompt}\n--- prompt end ---\n",
            flush=True,
        )

        final_max_tokens = max_tokens
        if not isinstance(final_max_tokens, int) or final_max_tokens <= 0:
            final_max_tokens = self.default_max_tokens or 16384

        # Run blocking requests.post in a thread so we don't block the loop
        data = await asyncio.to_thread(
            self._post_completions, prompt, temperature, final_max_tokens
        )

        content = data["choices"][0]["text"]
        reasoning_content = None

        # Handle Qwen3 <think>...</think> reasoning blocks
        if "qwen3" in self.model_name.lower() and "</think>" in content:
            reasoning_content = content.split("</think>")[0] + "</think>"
            content = content[content.rfind("</think>") + len("</think>"):]

        usage_raw = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
            "model": self.model_name,
            "api_base": self.api_base,
        }

        return LLMResponse(
            content=content,
            model=data.get("model", self.model_name),
            usage=usage,
            raw_response=data,
            finish_reason=data["choices"][0].get("finish_reason"),
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
