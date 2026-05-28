"""
Reward function for the search scaffolding example.

Checks whether the model produced an ``<answer>...</answer>`` tag and does
basic string matching against the ground truth.  For production training
use an LLM-as-judge (as in tongyi_deepresearch); this simple version is
sufficient for the fake-tool demo.
"""

import re


def search_reward_fn(
    prompt_str: str,
    completion_str: str,
    input_tokens: list[int],
    output_tokens: list[int],
    **data,
) -> float:
    """Compute reward for a search agent trajectory.

    The function extracts the text inside ``<answer>...</answer>`` from
    *completion_str* and compares it against ``data["answer"]``.  A reward
    of 1.0 is given when the ground-truth answer string appears (case-
    insensitive) inside the predicted answer; otherwise the reward is 0.0.

    A small bonus (0.1) is awarded if an ``<answer>`` tag is present but
    the content does not match, to encourage the model to at least produce
    a structured answer.

    Parameters
    ----------
    prompt_str : str
        The prompt string (unused beyond signature compatibility).
    completion_str : str
        The model's full completion text.
    input_tokens : list[int]
        Input token IDs (unused beyond signature compatibility).
    output_tokens : list[int]
        Output token IDs (unused beyond signature compatibility).
    **data
        Must contain an ``"answer"`` key with the ground-truth answer.

    Returns
    -------
    float
        Reward value: 1.0 for correct, 0.1 for structured but wrong, 0.0
        for missing answer tag.
    """
    ground_truth = data.get("answer", "")
    if isinstance(ground_truth, list):
        ground_truth = str(ground_truth[0]) if ground_truth else ""
    ground_truth = str(ground_truth).strip()

    # Extract predicted answer from <answer>...</answer> tags
    match = re.search(r"<answer>(.*?)</answer>", completion_str, re.DOTALL)
    if match is None:
        return 0.0

    predicted = match.group(1).strip()
    if not predicted:
        return 0.0

    # Case-insensitive containment check (either direction)
    if ground_truth.lower() in predicted.lower():
        return 1.0
    if predicted.lower() in ground_truth.lower():
        return 1.0

    # Structured answer present but incorrect
    return 0.1
