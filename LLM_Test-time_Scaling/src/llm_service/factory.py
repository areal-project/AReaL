"""Factory for creating LLM service instances."""

from typing import Any, Optional

from .base import LLMService
from .litellm_service import LiteLLMService
from .openai_service import OpenAIService
from .vllm_direct_service import VLLMDirectService
from .vllm_raw_service import VLLMRawService


def create_llm_service(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> LLMService:
    """Create an LLM service instance.

    Args:
        provider: Service provider ("openai", "litellm", "vllm_direct")
        model_name: Name of the model
        api_key: API key for authentication
        **kwargs: Additional configuration (e.g., base_url for OpenAI, api_base for LiteLLM)

    Returns:
        LLMService instance

    Examples:
        # OpenAI
        service = create_llm_service("openai", "gpt-4", api_key="...")

        # Custom OpenAI-compatible endpoint (e.g., for Qwen3 or GPT-OSS)
        service = create_llm_service(
            "openai",
            "qwen3-235b",
            api_key="...",
            base_url="https://custom-endpoint.com/v1"
        )

        # LiteLLM (supports multiple providers)
        service = create_llm_service("litellm", "gpt-4", api_key="...")
        service = create_llm_service("litellm", "claude-3-opus", api_key="...")
        service = create_llm_service("litellm", "gemini/gemini-3-pro", api_key="...")

        # LiteLLM with custom endpoint
        service = create_llm_service(
            "litellm",
            "qwen3-235b",
            api_key="...",
            api_base="https://custom-endpoint.com/v1"
        )
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAIService(model_name, api_key, **kwargs)
    elif provider == "litellm":
        return LiteLLMService(model_name, api_key, **kwargs)
    elif provider == "vllm_direct":
        return VLLMDirectService(model_name, api_key, **kwargs)
    elif provider == "vllm_raw":
        return VLLMRawService(model_name, api_key, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported providers: openai, litellm, vllm_direct, vllm_raw"
        )
