"""IMOBench evaluator using boxed-answer matching without LLM judging."""

import asyncio
import re
from typing import Any, Optional

from .base import EvaluationResult, Evaluator


class BoxedIMOBenchEvaluator(Evaluator):
    """Evaluate IMOBench solutions by comparing \boxed{} answer to ground truth."""

    def _extract_last_boxed(self, text: str) -> Optional[str]:
        marker = "\\boxed{"
        idx = text.rfind(marker)
        if idx == -1:
            return None

        start = idx + len(marker)
        depth = 1
        i = start
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
            i += 1
        return None

    def _normalize(self, value: str) -> str:
        normalized = value.strip()
        normalized = normalized.replace("$", "")
        normalized = re.sub(r"\\left|\\right", "", normalized)
        normalized = re.sub(r"\s+", "", normalized)
        return normalized

    async def evaluate(
        self,
        problem: str,
        solution: str,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        del problem
        del kwargs

        if ground_truth is None:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback="No ground truth provided",
                details={"error": "No ground truth"},
            )

        boxed_answer = self._extract_last_boxed(solution)
        if boxed_answer is None:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback="No \\boxed{} answer found in solution",
                details={
                    "boxed_answer": None,
                    "ground_truth": ground_truth,
                },
            )

        is_correct = self._normalize(boxed_answer) == self._normalize(str(ground_truth))
        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            feedback="Correct" if is_correct else "Incorrect",
            details={
                "boxed_answer": boxed_answer,
                "ground_truth": ground_truth,
            },
        )

    async def evaluate_batch(
        self,
        problems: list[str],
        solutions: list[str],
        ground_truths: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[EvaluationResult]:
        if ground_truths is None:
            ground_truths = [None] * len(problems)

        tasks = [
            self.evaluate(problem, solution, ground_truth, **kwargs)
            for problem, solution, ground_truth in zip(problems, solutions, ground_truths)
        ]
        return await asyncio.gather(*tasks)
