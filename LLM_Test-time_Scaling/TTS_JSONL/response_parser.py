"""Parse model responses for pairwise comparison and grading.

Contains the same parsing logic used by the existing PairwiseComparisonAggregation
and LLMJudge, extracted as standalone functions.
"""

import re
from typing import Optional, Tuple


def parse_comparison(response: str) -> int:
    """Parse a pairwise comparison response to determine the winner.

    Priority order:
    1. Markdown bold: **better solution:** **solution 1**
    2. Partial bold: **better solution:** solution 1
    3. Plain: Better Solution: Solution 1
    4. "solution X is better" patterns
    5. Conclusion-area heuristics (last 300 chars)
    6. First-200-char heuristics
    7. Fallback: 0 (tie)

    Returns:
        1 if Solution 1 is better, 2 if Solution 2 is better, 0 if tie/unparseable
    """
    response_lower = response.lower().strip()

    # Pattern 1a: "**better solution:** **solution 1**" (both parts bold)
    markdown_bold_match = re.search(
        r"\*\*better\s+solution:\*\*\s*\*\*(?:solution\s+)?([12])\*\*",
        response_lower,
        re.IGNORECASE | re.MULTILINE,
    )
    if markdown_bold_match:
        return int(markdown_bold_match.group(1))

    # Pattern 1b: "**better solution:** solution 1" (first part bold)
    markdown_match = re.search(
        r"\*\*better\s+solution:\*\*\s*(?:solution\s+)?([12])\b",
        response_lower,
        re.IGNORECASE | re.MULTILINE,
    )
    if markdown_match:
        return int(markdown_match.group(1))

    # Pattern 2: "Better Solution: Solution 1" or "Better Solution: [Solution 1 or ...]"
    better_solution_match = re.search(
        r"better\s+solution\s*:\s*(?:\[?\s*)?(?:solution\s*)?([12])\b",
        response_lower,
        re.IGNORECASE | re.MULTILINE,
    )
    if better_solution_match:
        return int(better_solution_match.group(1))

    # Pattern 3: "solution X is better"
    if re.search(r"solution\s*1\s+is\s+better", response_lower):
        return 1
    if re.search(r"solution\s*2\s+is\s+better", response_lower):
        return 2

    # Pattern 4: Conclusion area (last 300 chars)
    conclusion = response_lower[-300:]
    if re.search(r"(?:conclusion|final|answer|choose|select).*solution\s*1", conclusion):
        return 1
    if re.search(r"(?:conclusion|final|answer|choose|select).*solution\s*2", conclusion):
        return 2

    # Pattern 5: First 200 chars heuristic
    first_200 = response_lower[:200]
    has_sol1 = "solution 1" in first_200 or "solution1" in first_200
    has_sol2 = "solution 2" in first_200 or "solution2" in first_200
    if has_sol1 and not has_sol2:
        return 1
    if has_sol2 and not has_sol1:
        return 2

    return 0


def parse_grading(response: str) -> Tuple[bool, float, Optional[str]]:
    """Parse a grading/evaluation response.

    Expects format:
        CORRECT: [Yes/No]
        SCORE: [0-10]
        FEEDBACK: [text]

    Returns:
        (is_correct, score_0_to_1, feedback)
    """
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

    return (is_correct, score, feedback)
