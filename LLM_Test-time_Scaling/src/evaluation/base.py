"""Base classes for evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EvaluationResult:
    """Result of evaluating a solution."""

    is_correct: bool
    score: float  # 0.0 to 1.0
    feedback: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class Evaluator(ABC):
    """Abstract base class for evaluators."""

    @abstractmethod
    async def evaluate(
        self, problem: str, solution: str, ground_truth: Optional[str] = None, **kwargs: Any
    ) -> EvaluationResult:
        """Evaluate a solution.

        Args:
            problem: The problem statement
            solution: The solution to evaluate
            ground_truth: Optional ground truth answer
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with correctness and feedback
        """
        pass

    @abstractmethod
    async def evaluate_batch(
        self,
        problems: list[str],
        solutions: list[str],
        ground_truths: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[EvaluationResult]:
        """Evaluate a batch of solutions.

        Args:
            problems: List of problem statements
            solutions: List of solutions to evaluate
            ground_truths: Optional list of ground truth answers
            **kwargs: Additional evaluation parameters

        Returns:
            List of EvaluationResult objects
        """
        pass
