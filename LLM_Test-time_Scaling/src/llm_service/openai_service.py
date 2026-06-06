"""OpenAI-compatible LLM service implementation."""

import asyncio
from typing import Any, List, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMResponse, LLMService, Message


class OpenAIService(LLMService):
    """OpenAI-compatible LLM service (works with OpenAI API and compatible endpoints)."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize OpenAI service.

        Args:
            model_name: Name of the model (e.g., "gpt-4", "gpt-oss-120b")
            api_key: OpenAI API key
            base_url: Base URL for API (for custom endpoints)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, api_key, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using OpenAI API."""
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw_response=response,
            finish_reason=response.choices[0].finish_reason,
        )

    async def generate_batch(
        self,
        messages_batch: List[List[Message]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> List[LLMResponse]:
        """Generate responses for a batch of message sequences."""
        tasks = [
            self.generate(messages, temperature, max_tokens, **kwargs)
            for messages in messages_batch
        ]
        return await asyncio.gather(*tasks)
