"""LLM judge grading via eval vLLM server.

Constructs ChatML prompts for mathematical solution grading and sends them
to a vLLM eval server, then parses the CORRECT/SCORE/FEEDBACK response.
"""

from typing import Dict, Optional

import aiohttp

from vllm_client import post_completion, extract_response_text
from response_parser import parse_grading


GRADING_SYSTEM_PROMPT = (
    "# System Role: Deterministic Mathematical Autograder\n"
    "You are a precise, automated grading system.\n"
    "Your sole function is to determine if the final answer provided in the "
    "Model Solution is mathematically equivalent to the Golden Answer.\n"
    "You must NOT grade the reasoning or steps, only the final result."
)

GRADING_USER_TEMPLATE = """Problem:
{problem}

Solution to Evaluate:
{solution}

Ground Truth Answer:
{ground_truth}

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
"""


def build_grading_prompt(problem: str, solution: str, ground_truth: str) -> str:
    """Construct the full ChatML prompt for grading.

    Returns the raw prompt string ready to send to /v1/completions.
    """
    user_content = GRADING_USER_TEMPLATE.format(
        problem=problem, solution=solution, ground_truth=ground_truth
    )

    prompt = (
        f"<|im_start|>system\n{GRADING_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


async def grade_solution(
    session: aiohttp.ClientSession,
    api_base: str,
    model_name: str,
    problem: str,
    solution: str,
    ground_truth: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 300,
) -> Dict:
    """Grade a solution against ground truth using the eval vLLM server.

    Returns:
        {"is_correct": bool, "score": float, "feedback": str|None, "raw_response": str}
    """
    prompt = build_grading_prompt(problem, solution, ground_truth)

    response_json = await post_completion(
        session=session,
        api_base=api_base,
        model_name=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    raw_text = response_json.get("choices", [{}])[0].get("text", "")
    content, reasoning, usage = extract_response_text(response_json)
    is_correct, score, feedback = parse_grading(content)

    print(f"    [Grading] Raw response (first 500 chars): {raw_text[:500]}")
    if reasoning:
        print(f"    [Grading] Reasoning (first 300 chars): {reasoning[:300]}")
    print(f"    [Grading] Parsed content (first 300 chars): {content[:300]}")
    print(f"    [Grading] Result: correct={is_correct}, score={score}, feedback={feedback}")

    return {
        "is_correct": is_correct,
        "score": score,
        "feedback": feedback,
        "raw_response": content,
    }
