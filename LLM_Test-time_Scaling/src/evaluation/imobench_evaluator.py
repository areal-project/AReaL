"""IMOBench evaluator."""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import EvaluationResult, Evaluator
from .llm_judge import LLMJudge


class IMOBenchEvaluator(Evaluator):
    """Evaluator for IMOBench (AnswerBench) benchmark."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        llm_judge: Optional[LLMJudge] = None,
    ):
        """Initialize IMOBench evaluator.

        Args:
            data_path: Path to answerbench.csv or imobench.json
            llm_judge: Optional LLM judge for answer evaluation (if None, uses simple string comparison)
        """
        self.data_path = data_path
        self.llm_judge = llm_judge
        self._problems = None

    def load_problems(self, data_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Load problems from CSV or JSON file.

        Args:
            data_path: Path to data file (overrides self.data_path)

        Returns:
            List of problem dictionaries
        """
        if data_path is None:
            data_path = self.data_path

        if data_path is None:
            # Try default location
            default_path = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "benchmarks"
                / "imobench.json"
            )
            if default_path.exists():
                data_path = default_path
            else:
                raise ValueError("No data path provided and default not found")

        data_path = Path(data_path)

        if data_path.suffix == ".csv":
            return self._load_from_csv(data_path)
        elif data_path.suffix == ".json":
            return self._load_from_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    def _load_from_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Load problems from CSV file."""
        problems = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                problem = {
                    "id": row.get("id", ""),
                    "problem": row.get("problem", ""),
                    "short_answer": row.get("short_answer", ""),
                    "category": row.get("category", ""),
                    "subcategory": row.get("subcategory", ""),
                    "metadata": {k: v for k, v in row.items() if k not in ["id", "problem", "short_answer"]},
                }
                problems.append(problem)
        return problems

    def _load_from_json(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load problems from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "problems" in data:
            return data["problems"]
        else:
            raise ValueError("Invalid JSON format")

    def extract_answer(self, solution: str) -> str:
        """Extract final answer from solution text.
        
        Args:
            solution: Full solution text
            
        Returns:
            Extracted answer string (or original solution if no pattern matches)
        """
        # Try to extract final answer from solution (common pattern: "The answer is X")
        # Look for common answer patterns
        answer_patterns = [
            r"\\boxed\{([^}]+)\}",
            r"(?:the answer is|answer:|answer is|answer:)\s*([^\n\.]+)",
            r"final answer[:\s]+([^\n\.]+)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    return extracted
        
        # If no pattern matches, return the last line or the whole solution
        lines = solution.strip().split('\n')
        if lines:
            # Try to use the last non-empty line as answer
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('#'):
                    return line
        
        return solution.strip()

    def compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """Compare predicted answer with ground truth.

        This is a simplified version. For more sophisticated comparison,
        see the IMOBench evaluation script.

        Args:
            predicted: Predicted answer string
            ground_truth: Ground truth answer string

        Returns:
            True if answers match (with normalization)
        """
        # Normalize whitespace
        pred = " ".join(predicted.strip().split())
        gt = " ".join(ground_truth.strip().split())

        # Direct match
        if pred == gt:
            return True

        # Try normalized comparison
        if self._normalize_answer(pred) == self._normalize_answer(gt):
            return True

        return False

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove LaTeX formatting
        answer = re.sub(r"\\[a-zA-Z]+\{([^}]+)\}", r"\1", answer)
        answer = re.sub(r"\$([^$]+)\$", r"\1", answer)

        # Normalize whitespace and case
        answer = " ".join(answer.strip().split()).lower()

        # Remove common prefixes
        answer = re.sub(r"^(the answer is|answer:|answer is)\s*", "", answer, flags=re.IGNORECASE)

        return answer.strip()

    async def evaluate(
        self,
        problem: str,
        solution: str,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a solution.

        Args:
            problem: Problem statement
            solution: Solution/answer to evaluate
            ground_truth: Ground truth answer
            **kwargs: Additional parameters

        Returns:
            EvaluationResult
        """
        if ground_truth is None:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback="No ground truth provided",
                details={"error": "No ground truth"},
            )

        # Extract answer from solution
        extracted_answer = self.extract_answer(solution)
        
        # Use LLM judge if available, otherwise use simple comparison
        if self.llm_judge is not None:
            # Use LLM judge to evaluate the solution
            # Pass the full solution as context so LLM can see the reasoning process
            result = await self.llm_judge.evaluate(
                problem=problem,
                solution=solution,  # Pass full solution for context
                ground_truth=ground_truth,
                **kwargs,
            )
            
            # Add extracted answer to details (merge with existing details)
            if result.details is None:
                result.details = {}
            result.details.update({
                "extracted_answer": extracted_answer,
                "original_solution": solution,
                "ground_truth": ground_truth,
            })
            
            return result
        else:
            # Fallback to simple string comparison
            is_correct = self.compare_answers(extracted_answer, ground_truth)
            score = 1.0 if is_correct else 0.0

            feedback = "Correct" if is_correct else "Incorrect"
            if not is_correct:
                feedback += f"\nExtracted Answer: {extracted_answer[:200]}"
                feedback += f"\nExpected: {ground_truth[:200]}"

            return EvaluationResult(
                is_correct=is_correct,
                score=score,
                feedback=feedback,
                details={
                    "extracted_answer": extracted_answer,
                    "original_solution": solution,
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
        """Evaluate a batch of solutions."""
        if ground_truths is None:
            ground_truths = [None] * len(problems)

        import asyncio
        tasks = [
            self.evaluate(problem, solution, gt, **kwargs)
            for problem, solution, gt in zip(problems, solutions, ground_truths)
        ]

        return await asyncio.gather(*tasks)

