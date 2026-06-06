"""Base class for aggregation strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

try:
    from litellm import token_counter
except ImportError:
    token_counter = None

from ...llm_service import Message
from ..base import Solution


class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies."""

    def __init__(self, llm_service: Any, temperature: float = 0.0,
                 reasoning_effort: Optional[str] = "auto", **kwargs: Any):
        """Initialize aggregation strategy.

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
    async def aggregate(self, problem: str, solutions: List[Solution], **kwargs: Any) -> Solution:
        """Aggregate multiple solutions into a single best solution.

        Args:
            problem: The problem statement
            solutions: List of candidate solutions
            **kwargs: Additional parameters

        Returns:
            The aggregated/selected best solution
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of the aggregation strategy."""
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

    def _check_input_tokens(self, messages: List[Message], max_tokens: int = 128000) -> tuple[bool, int]:
        """Check if input tokens exceed the limit.
        
        Args:
            messages: List of messages to check
            max_tokens: Maximum allowed input tokens (default: 128000)
            
        Returns:
            Tuple of (exceeds_limit: bool, input_tokens: int)
        """
        if token_counter is None:
            # If litellm is not available, skip check
            return False, 0
        
        try:
            # Convert Message objects to dict format for token_counter
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            input_tokens = token_counter(model=self.llm_service.model_name, messages=formatted_messages)
            exceeds_limit = input_tokens > max_tokens
            return exceeds_limit, input_tokens
        except Exception:
            # If token counting fails, assume it's okay (let LLM service handle it)
            return False, 0

    def _create_skip_solution(self, problem: str, solutions: List[Solution], 
                             reason: str = "input_tokens_exceeded", 
                             input_tokens: int = 0) -> Solution:
        """Create a skip solution when input tokens exceed limit.
        
        Args:
            problem: The problem statement
            solutions: List of candidate solutions
            reason: Reason for skipping
            input_tokens: Number of input tokens (if known)
            
        Returns:
            A Solution object marked as skipped
        """
        # Return first solution with skip metadata
        skip_solution = solutions[0] if solutions else Solution(content="", metadata={})
        skip_solution.metadata["aggregation"] = self.get_strategy_name()
        skip_solution.metadata["skipped"] = True
        skip_solution.metadata["skip_reason"] = reason
        if input_tokens > 0:
            skip_solution.metadata["input_tokens"] = input_tokens
        skip_solution.metadata["n_candidates"] = len(solutions)
        return skip_solution
