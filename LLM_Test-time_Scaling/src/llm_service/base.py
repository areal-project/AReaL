"""Base classes for LLM service."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Response from an LLM service."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    finish_reason: Optional[str] = None
    reasoning_content: Optional[str] = None


@dataclass
class Message:
    """Message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str


class LLMService(ABC):
    """Abstract base class for LLM services."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs: Any):
        """Initialize the LLM service.

        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse containing the generated text
        """
        pass

    @abstractmethod
    async def generate_batch(
        self,
        messages_batch: List[List[Message]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> List[LLMResponse]:
        """Generate responses for a batch of message sequences.

        Args:
            messages_batch: List of message sequences
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            List of LLMResponse objects
        """
        pass
