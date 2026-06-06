"""LLM-based judge for evaluating solutions."""

import asyncio
import re
from typing import Any, Optional

from ..llm_service import LLMService, Message
from .base import EvaluationResult, Evaluator


class LLMJudge(Evaluator):
    """LLM-based judge for evaluating solutions."""

    def __init__(
        self,
        llm_service: LLMService,
        benchmark: str = "general",
        system_prompt: Optional[str] = None,
        evaluation_prompt_template: Optional[str] = None,
    ):
        """Initialize the LLM judge.

        Args:
            llm_service: LLM service for evaluation
            benchmark: Benchmark name (imobench, lcb_pro, prbench, hle, general)
            system_prompt: Custom system prompt (overrides benchmark defaults)
            evaluation_prompt_template: Custom evaluation prompt template
        """
        self.llm_service = llm_service
        self.benchmark = benchmark
        self.system_prompt = system_prompt or self._get_default_system_prompt(benchmark)
        self.evaluation_prompt_template = (
            evaluation_prompt_template or self._get_default_evaluation_template(benchmark)
        )

    def _get_default_system_prompt(self, benchmark: str) -> str:
        """Get default system prompt for the benchmark."""
        prompts = {
            # "imobench": (
            #     "You are an expert mathematician evaluating mathematical solutions. "
            #     "Your task is to compare the final answer in the solution with the ground truth answer. "
            #     "Focus on:\n"
            #     "1. Whether the final answer matches the ground truth\n"
            #     "2. Whether the answer is mathematically equivalent (e.g., simplified forms)\n"
            #     "3. Brief assessment of solution correctness\n"
            #     "Be concise and focus on answer comparison rather than detailed step verification."
            # ),
            "imobench": (
                "# System Role: Deterministic Mathematical Autograder"
                "You are a precise, automated grading system."
                "Your sole function is to determine if the final answer provided in the Model Solution is mathematically equivalent to the Golden Answer."
                "You must NOT grade the reasoning or steps, only the final result."
            ), # this is from imobench paper
            "lcb_pro": (
                "You are an expert code evaluator for professional coding challenges. "
                "Evaluate code solutions considering:\n"
                "1. Correctness of the implementation\n"
                "2. Algorithm efficiency and optimization\n"
                "3. Edge case handling\n"
                "4. Code quality and readability\n"
                "Note: Actual execution results should be the primary factor."
            ),
            "prbench": (
                "You are an expert evaluator for professional reasoning in finance and law. "
                "Evaluate solutions based on:\n"
                "1. Logical reasoning and argumentation quality\n"
                "2. Accuracy of domain-specific knowledge\n"
                "3. Completeness of analysis\n"
                "4. Practical applicability and real-world considerations"
            ),
            "hle": (
                "You are an expert evaluator for multimodal understanding and reasoning. "
                "Evaluate solutions considering:\n"
                "1. Correct interpretation of visual and textual information\n"
                "2. Logical reasoning across modalities\n"
                "3. Completeness and accuracy of the answer\n"
                "4. Proper integration of all provided information"
            ),
            "gpqa_diamond": (
                "You are an expert evaluator. Assess whether the solution correctly "
                "selects the correct choice. Provide a detailed feedback."
            ),
            "general": (
                "You are an expert evaluator. Assess whether the solution correctly "
                "solves the given problem. Provide a score from 0 to 10 and detailed feedback."
            ),
        }
        return prompts.get(benchmark, prompts["general"])

    def _get_default_evaluation_template(self, benchmark: str) -> str:
        """Get default evaluation prompt template for the benchmark."""
        templates = {
#             "imobench": """Problem:
# {problem}

# Solution to Evaluate:
# {solution}
# {ground_truth_section}
# Please evaluate the mathematical solution:
# 1. Extract the final answer from the solution
# 2. Compare it with the ground truth answer
# 3. Determine if they are mathematically equivalent
# 4. Provide a brief assessment

# Format your response as:
# CORRECT: [Yes/No]
# SCORE: [0-10]
# FEEDBACK: [Brief comparison of final answer with ground truth]
# """,
            "imobench": """Problem:
{problem}

Solution to Evaluate:
{solution}
{ground_truth_section}
Please evaluate the mathematical solution:
1. Extract the final answer from the solution
2. Compare it with the ground truth answer using strict equivalence rules:
   - **Algebraic Equivalence:** e.g., 'n(n+1)/2' is equivalent to 'n^2/2 + n/2'. You must verify the algebra.
   - **Numerical Equivalence:** e.g., '1/2' is equivalent to '0.5'; 'sqrt(2)/2' is equivalent to '1/sqrt(2)'.
   - **Set/List Equivalence:** Unless specified as an ordered tuple/vector, the order of elements does not matter (e.g., {{1, 2}} is equivalent to {{2, 1}}).
   - **No Partial Credit:** If the answer is incomplete or partially incorrect, it is incorrect.
   - **No Answers:** If no clear, unambiguous final answer can be extracted, the solution must be graded as incorrect.
3. Determine if they are mathematically equivalent
4. Score the solution (0-10) based on correctness (10 if correct, 0 if incorrect)
5. Provide detailed feedback comparing the final answer with ground truth

Format your response as:
CORRECT: [Yes/No]
SCORE: [0-10]
FEEDBACK: [Brief comparison of final answer with ground truth, including equivalence analysis]
""",
            "lcb_pro": """Problem:
{problem}

Solution to Evaluate:
{solution}
{ground_truth_section}
Please evaluate the code solution:
1. Does it correctly solve the problem? (Yes/No)
2. Score the solution (0-10) considering:
   - Correctness (primary)
   - Algorithm efficiency
   - Edge case handling
   - Code quality
3. Provide specific feedback on the implementation

Format your response as:
CORRECT: [Yes/No]
SCORE: [0-10]
FEEDBACK: [your detailed code review]
""",
            "prbench": """Problem:
{problem}

Solution to Evaluate:
{solution}
{ground_truth_section}
Please evaluate the professional reasoning:
1. Is the reasoning sound and conclusion correct? (Yes/No)
2. Score the solution (0-10) based on:
   - Logical soundness
   - Domain knowledge accuracy
   - Completeness of analysis
   - Practical applicability
3. Provide detailed feedback on the reasoning quality

Format your response as:
CORRECT: [Yes/No]
SCORE: [0-10]
FEEDBACK: [your detailed evaluation of reasoning]
""",
            "hle": """Problem:
{problem}

Solution to Evaluate:
{solution}
{ground_truth_section}
Please evaluate the multimodal reasoning:
1. Is the answer correct? (Yes/No)
2. Score the solution (0-10) based on:
   - Correct interpretation of all modalities
   - Logical reasoning quality
   - Completeness and accuracy
   - Integration of information
3. Provide detailed feedback

Format your response as:
CORRECT: [Yes/No]
SCORE: [0-10]
FEEDBACK: [your detailed evaluation]
""",
            "gpqa_diamond": """Problem:
{problem}

Solution to Evaluate:
```
{solution}
```

{ground_truth_section}

Please evaluate the solution:
1. Does the solution selects the correct choice? (Yes/No)
2. Score (0 or 10):
3. Detailed feedback:

Format your response as:
CORRECT: [Yes/No]
SCORE: [0 or 10]
FEEDBACK: [your detailed feedback]
""",
            "general": """Problem:
{problem}

Solution to Evaluate:
{solution}
{ground_truth_section}
Please evaluate the solution:
1. Is it correct? (Yes/No)
2. Score (0-10):
3. Detailed feedback:

Format your response as:
CORRECT: [Yes/No]
SCORE: [0-10]
FEEDBACK: [your detailed feedback]
""",
        }
        return templates.get(benchmark, templates["general"])

    async def evaluate(
        self, problem: str, solution: str, ground_truth: Optional[str] = None, **kwargs: Any
    ) -> EvaluationResult:
        """Evaluate a solution using LLM."""
        ground_truth_section = ""
        if ground_truth:
            ground_truth_section = f"\nGround Truth Answer:\n{ground_truth}\n"

        user_prompt = self.evaluation_prompt_template.format(
            problem=problem, solution=solution, ground_truth_section=ground_truth_section
        )

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_prompt),
        ]

        response = await self.llm_service.generate(messages, temperature=0.0)
        return self._parse_evaluation_response(response.content)

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

        tasks = [
            self.evaluate(problem, solution, ground_truth)
            for problem, solution, ground_truth in zip(problems, solutions, ground_truths)
        ]

        return await asyncio.gather(*tasks)

    def _parse_evaluation_response(self, response: str) -> EvaluationResult:
        """Parse the LLM evaluation response."""
        correct_match = re.search(r"CORRECT:\s*(Yes|No)", response, re.IGNORECASE)
        score_match = re.search(r"SCORE:\s*(\d+)", response)
        feedback_match = re.search(r"FEEDBACK:\s*(.+)", response, re.DOTALL)

        is_correct = False
        if correct_match:
            is_correct = correct_match.group(1).lower() == "yes"

        score = 0.0
        if score_match:
            score = float(score_match.group(1)) / 10.0

        feedback = feedback_match.group(1).strip() if feedback_match else None

        return EvaluationResult(
            is_correct=is_correct, score=score, feedback=feedback, details={"raw_response": response}
        )
