"""LLM scoring aggregation strategy."""

import asyncio
import re
from typing import Any, Dict, List, Optional

from ...llm_service import LLMService, Message
from ...prompts import PromptTemplate
from ..base import Solution
from .base import AggregationStrategy


class LLMScoringAggregation(AggregationStrategy):
    """Aggregation strategy that scores solutions using LLM and selects the best."""

    def __init__(
        self, llm_service: LLMService, scoring_prompt_template: Optional[PromptTemplate],
        temperature: float = 0.0, reasoning_effort: Optional[str] = "auto"
    ):
        """Initialize LLM scoring aggregation.

        Args:
            llm_service: LLM service for scoring
            scoring_prompt_template: PromptTemplate for scoring prompt
            temperature: Temperature for scoring (default: 0.0 for deterministic)
            reasoning_effort: Reasoning effort ("auto", "low", "medium", "high", or None)
        """
        super().__init__(llm_service, temperature=temperature, reasoning_effort=reasoning_effort)
        self.scoring_prompt_template = scoring_prompt_template

    async def aggregate(self, problem: str, solutions: List[Solution], **kwargs: Any) -> Solution:
        """Score solutions using LLM and select the highest scored one."""
        if len(solutions) == 1:
            return solutions[0]

        tasks = [self._score_solution(problem, sol, idx) for idx, sol in enumerate(solutions)]
        results = await asyncio.gather(*tasks)

        # Extract scores and token usage
        scores = []
        total_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        token_usage_list = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": []
        }
        llm_call_details = []  # Store detailed info for each LLM call
        
        for idx, result in enumerate(results):
            if isinstance(result, tuple):
                if len(result) == 4:
                    # Result is (score, token_usage, reasoning_content, content)
                    score, usage, reasoning, content = result
                    scores.append(score)
                    if usage:
                        total_token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        total_token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                        total_token_usage["total_tokens"] += usage.get("total_tokens", 0)
                        token_usage_list["prompt_tokens"].append(usage.get("prompt_tokens", 0))
                        token_usage_list["completion_tokens"].append(usage.get("completion_tokens", 0))
                        token_usage_list["total_tokens"].append(usage.get("total_tokens", 0))
                    llm_call_details.append({
                        "solution_idx": idx,
                        "token_usage": usage,
                        "reasoning_content": reasoning,
                        "content": content,
                    })
                elif len(result) == 2:
                    # Result is (score, token_usage) - backward compatibility
                    score, usage = result
                    scores.append(score)
                    if usage:
                        total_token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        total_token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                        total_token_usage["total_tokens"] += usage.get("total_tokens", 0)
                        token_usage_list["prompt_tokens"].append(usage.get("prompt_tokens", 0))
                        token_usage_list["completion_tokens"].append(usage.get("completion_tokens", 0))
                        token_usage_list["total_tokens"].append(usage.get("total_tokens", 0))
                    llm_call_details.append({
                        "solution_idx": idx,
                        "token_usage": usage,
                        "reasoning_content": "",
                        "content": "",
                    })
            else:
                # Backward compatibility if _score_solution returns just score
                scores.append(result)
                llm_call_details.append({
                    "solution_idx": idx,
                    "token_usage": {},
                    "reasoning_content": "",
                    "content": "",
                })

        for solution, score in zip(solutions, scores):
            solution.score = score

        best_idx = max(range(len(solutions)), key=lambda i: scores[i])
        best_solution = solutions[best_idx]
        best_solution.metadata["aggregation"] = "llm_scoring"
        best_solution.metadata["all_scores"] = scores
        best_solution.metadata["n_candidates"] = len(solutions)
        best_solution.metadata["token_usage"] = total_token_usage
        best_solution.metadata["token_usage_list"] = token_usage_list
        best_solution.metadata["llm_call_details"] = llm_call_details  # Store detailed info for each call

        return best_solution

    async def _score_solution(self, problem: str, solution: Solution, solution_idx: int) -> tuple[float, Optional[Dict[str, int]], str, str]:
        """Score a single solution.
        
        Args:
            problem: Problem statement
            solution: Solution to score
            solution_idx: Index of the solution in the solutions list
            
        Returns:
            Tuple of (score, token_usage_dict, reasoning_content, content)
        """
        if self.scoring_prompt_template:
            scoring_formatted = self.scoring_prompt_template.format_with_system(
                problem=problem, solution=solution.content
            )
            messages = [
                Message(role="system", content=scoring_formatted["system"]),
                Message(role="user", content=scoring_formatted["user"])
            ]
        else:
            # Fallback
            messages = [Message(role="user", content=f"Problem: {problem}\nSolution: {solution.content}\nScore (0-10):")]

        # Build kwargs with reasoning effort
        gen_kwargs = self._build_generation_kwargs()
        response = await self.llm_service.generate(messages, **gen_kwargs)

        # Extract token usage
        usage = response.usage or {}
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        score = self._parse_score(response.content)
        reasoning_content = response.reasoning_content or ""
        content = response.content
        return (score, token_usage, reasoning_content, content)

    def _parse_score(self, response: str) -> float:
        """Parse score from response."""
        match = re.search(r"[Ss]core:\s*(\d+(?:\.\d+)?)", response)
        if match:
            score = float(match.group(1))
            if score <= 10:
                return score / 10.0
            return score / 100.0

        return 0.5

    def get_strategy_name(self) -> str:
        """Get the name of the aggregation strategy."""
        return "llm_scoring"
