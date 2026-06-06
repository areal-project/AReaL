"""Base class for reflection strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..base import Solution


class ReflectionStrategy(ABC):
    """Abstract base class for reflection strategies."""

    def __init__(self, llm_service: Any, temperature: float = 0.7,
                 reasoning_effort: Optional[str] = "auto", **kwargs: Any):
        """Initialize reflection strategy.

        Args:
            llm_service: LLM service for generation
            temperature: Sampling temperature for generation
            reasoning_effort: Reasoning effort ("auto", "low", "medium", "high", or None)
            **kwargs: Additional strategy-specific parameters
        """
        self.llm_service = llm_service
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    @abstractmethod
    async def reflect(
        self, problem: str, solutions: List[Solution], **kwargs: Any
    ) -> List[Solution]:
        """Apply reflection to improve solutions.

        Args:
            problem: The problem statement
            solutions: Current solutions to reflect on
            **kwargs: Additional parameters (e.g., ground_truth, test_cases)

        Returns:
            Improved solutions with feedback
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of the reflection strategy."""
        pass

    def _get_reasoning_effort(self, model_name: str, configured_effort: Optional[str]) -> Optional[str]:
        """Determine reasoning effort based on configuration and model.

        Args:
            model_name: Name of the model being used
            configured_effort: Configured reasoning effort ("auto", "low", "medium", "high", or None)

        Returns:
            Reasoning effort to use, or None if not applicable
        """
        if configured_effort == "auto":
            # Auto-detect: enable high effort for gpt-oss models
            return "high" if "gpt-oss" in model_name.lower() else None
        elif configured_effort in ["low", "medium", "high"]:
            return configured_effort
        else:
            # None or any other value disables reasoning effort
            return None

    def _build_generation_kwargs(self, temperature: Optional[float] = None,
                                  **base_kwargs: Any) -> Dict[str, Any]:
        """Build kwargs for LLM generation with reasoning effort if applicable.

        Args:
            temperature: Temperature override (uses self.temperature if None)
            **base_kwargs: Base kwargs to include

        Returns:
            Dictionary of kwargs for LLM generation
        """
        kwargs = dict(base_kwargs)

        # Set temperature
        if temperature is None:
            temperature = self.temperature
        
        kwargs["temperature"] = temperature

        # Add reasoning effort if applicable
        effort = self._get_reasoning_effort(self.llm_service.model_name, self.reasoning_effort)
        if effort:
            kwargs["extra_body"] = {"reasoning_effort": effort}

        return kwargs
