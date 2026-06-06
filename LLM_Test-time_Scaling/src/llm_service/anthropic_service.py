"""Anthropic Claude LLM service implementation."""

import asyncio
from typing import Any, List, Optional

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMResponse, LLMService, Message


class AnthropicService(LLMService):
    """Anthropic Claude LLM service."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs: Any):
        """Initialize Anthropic service.

        Args:
            model_name: Name of the Claude model
            api_key: Anthropic API key
            **kwargs: Additional configuration
        """
        super().__init__(model_name, api_key, **kwargs)
        self.client = AsyncAnthropic(api_key=api_key)

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
        """Generate a response using Anthropic API."""
        system_messages = [msg.content for msg in messages if msg.role == "system"]
        system_prompt = "\n\n".join(system_messages) if system_messages else None

        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role in ["user", "assistant"]
        ]

        response = await self.client.messages.create(
            model=self.model_name,
            messages=formatted_messages,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
            **kwargs,
        )

        return LLMResponse(
            content=response.content[0].text if response.content else "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            raw_response=response,
            finish_reason=response.stop_reason,
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
