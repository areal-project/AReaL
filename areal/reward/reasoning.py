# SPDX-License-Identifier: Apache-2.0

"""Composable reasoning rewards for LLM post-training (DeepSeek-R1 style RLVR).

Provides format verification for thinking/reasoning tags (<think>...</think>),
answer delimiter extraction (\\boxed{}, <answer>, ####), anti-length-hacking
penalties, and a composite reward wrapper that combines task accuracy, format
compliance, and length regulation with stats_tracker integration.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from areal.utils import logging, stats_tracker

logger = logging.getLogger("ReasoningReward")


def extract_boxed_content(text: str) -> str | None:
    """Extract content from the last \\boxed{...} with balanced brace matching.

    Handles nested braces properly (e.g. ``\\boxed{\\frac{1}{2}}``).

    Parameters
    ----------
    text : str
        Input string containing LaTeX boxed expressions.

    Returns
    -------
    str | None
        Extracted content within the last \\boxed{}, or None if no valid
        boxed expression is found.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    brace_start = idx + len("\\boxed{")
    depth = 1
    i = brace_start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[brace_start : i - 1].strip()
    return None


def extract_tag_content(text: str, tag: str = "answer") -> str | None:
    """Extract content from an XML-like tag (e.g. <answer>...</answer>).

    Parameters
    ----------
    text : str
        Input string.
    tag : str, optional
        Tag name to match, by default "answer".

    Returns
    -------
    str | None
        Content of the last matching tag pair, or None if not found.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = list(re.finditer(pattern, text, flags=re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def extract_hash_answer(text: str) -> str | None:
    """Extract answer following GSM8K-style '####' delimiter.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str | None
        Answer string following the final '####', or None if not found.
    """
    if "####" not in text:
        return None
    return text.split("####")[-1].strip()


def extract_reasoning_and_answer(
    text: str,
    think_start_tag: str = "<think>",
    think_end_tag: str = "</think>",
) -> tuple[str, str]:
    """Separate reasoning chain and visible answer from a completion.

    Handles complete, unclosed, and missing thinking tags gracefully.

    Parameters
    ----------
    text : str
        Model output text.
    think_start_tag : str, optional
        Opening thinking tag, by default "<think>".
    think_end_tag : str, optional
        Closing thinking tag, by default "</think>".

    Returns
    -------
    tuple[str, str]
        (reasoning_text, answer_text)
    """
    has_start = think_start_tag in text
    has_end = think_end_tag in text

    if has_start and has_end:
        start_idx = text.find(think_start_tag) + len(think_start_tag)
        end_idx = text.rfind(think_end_tag)
        if start_idx <= end_idx:
            reasoning = text[start_idx:end_idx].strip()
            answer = text[end_idx + len(think_end_tag) :].strip()
            return reasoning, answer
        # Malformed ordering (end before start)
        return "", text.strip()

    if has_start and not has_end:
        # Generation truncated inside thinking block
        start_idx = text.find(think_start_tag) + len(think_start_tag)
        return text[start_idx:].strip(), ""

    if not has_start and has_end:
        # Implicit thinking before closing tag
        end_idx = text.find(think_end_tag)
        reasoning = text[:end_idx].strip()
        answer = text[end_idx + len(think_end_tag) :].strip()
        return reasoning, answer

    # No thinking tags
    return "", text.strip()


@dataclass
class FormatRewardConfig:
    """Configuration for reasoning format reward verification."""

    think_start_tag: str = "<think>"
    think_end_tag: str = "</think>"
    require_think_tags: bool = True
    require_closed_think: bool = True
    allow_empty_think: bool = False
    answer_format: str = "boxed"  # "boxed", "xml", "hash", or "any"
    answer_tag: str = "answer"  # Used when answer_format == "xml"
    structure_reward: float = 0.5
    answer_format_reward: float = 0.5
    malformed_penalty: float = -0.5


class FormatReward:
    """Evaluates format compliance for reasoning models."""

    def __init__(self, config: FormatRewardConfig | None = None):
        self.config = config or FormatRewardConfig()

    def evaluate_detailed(self, completion: str) -> dict[str, Any]:
        """Evaluate format and return detailed breakdown and extracted contents."""
        cfg = self.config
        text = str(completion)

        start_count = text.count(cfg.think_start_tag)
        end_count = text.count(cfg.think_end_tag)

        # Check for multiple/mismatched tags
        if start_count > 1 or end_count > 1:
            return {
                "reward": cfg.malformed_penalty,
                "valid_think": False,
                "valid_answer_format": False,
                "reasoning": "",
                "extracted_answer": None,
                "error": "multiple_tags",
            }

        if cfg.require_think_tags:
            if start_count == 0:
                return {
                    "reward": cfg.malformed_penalty,
                    "valid_think": False,
                    "valid_answer_format": False,
                    "reasoning": "",
                    "extracted_answer": None,
                    "error": "missing_start_tag",
                }
            if cfg.require_closed_think and end_count == 0:
                return {
                    "reward": cfg.malformed_penalty,
                    "valid_think": False,
                    "valid_answer_format": False,
                    "reasoning": "",
                    "extracted_answer": None,
                    "error": "unclosed_think",
                }

        reasoning, answer = extract_reasoning_and_answer(
            text, cfg.think_start_tag, cfg.think_end_tag
        )

        if cfg.require_think_tags and not cfg.allow_empty_think and not reasoning:
            return {
                "reward": cfg.malformed_penalty,
                "valid_think": False,
                "valid_answer_format": False,
                "reasoning": "",
                "extracted_answer": None,
                "error": "empty_think",
            }

        # Check tag ordering if both exist
        if start_count == 1 and end_count == 1:
            if text.find(cfg.think_start_tag) > text.find(cfg.think_end_tag):
                return {
                    "reward": cfg.malformed_penalty,
                    "valid_think": False,
                    "valid_answer_format": False,
                    "reasoning": "",
                    "extracted_answer": None,
                    "error": "reversed_tags",
                }

        # Validate answer format
        extracted_ans: str | None = None
        valid_ans = False

        if cfg.answer_format == "boxed":
            extracted_ans = extract_boxed_content(answer or text)
            valid_ans = extracted_ans is not None
        elif cfg.answer_format == "xml":
            extracted_ans = extract_tag_content(answer or text, cfg.answer_tag)
            valid_ans = extracted_ans is not None
        elif cfg.answer_format == "hash":
            extracted_ans = extract_hash_answer(answer or text)
            valid_ans = extracted_ans is not None
        elif cfg.answer_format == "any":
            extracted_ans = answer if answer else text
            valid_ans = bool(extracted_ans.strip())
        else:
            raise ValueError(f"Unknown answer_format: {cfg.answer_format!r}")

        reward = 0.0
        if not cfg.require_think_tags or (start_count == 1 and end_count == 1):
            reward += cfg.structure_reward
        if valid_ans:
            reward += cfg.answer_format_reward

        return {
            "reward": reward,
            "valid_think": True,
            "valid_answer_format": valid_ans,
            "reasoning": reasoning,
            "extracted_answer": extracted_ans,
            "error": None,
        }

    def evaluate(self, completion: str) -> float:
        """Evaluate format and return format reward score."""
        return float(self.evaluate_detailed(completion)["reward"])

    def __call__(
        self,
        prompt: Any,
        completions: Any,
        prompt_ids: Any = None,
        completion_ids: Any = None,
        **kwargs: Any,
    ) -> float:
        return self.evaluate(str(completions))


@dataclass
class LengthPenaltyConfig:
    """Configuration for output length regulation."""

    target_length: int = 1024
    penalty_type: str = "threshold"  # "threshold", "linear", "soft_tanh"
    penalty_factor: float = 0.001
    max_penalty: float = 1.0


class LengthPenalty:
    """Penalizes excessively verbose reasoning to prevent length hacking."""

    def __init__(self, config: LengthPenaltyConfig | None = None):
        self.config = config or LengthPenaltyConfig()
        if self.config.target_length < 0:
            raise ValueError("target_length must be non-negative")
        if self.config.penalty_factor < 0:
            raise ValueError("penalty_factor must be non-negative")
        if self.config.max_penalty < 0:
            raise ValueError("max_penalty must be non-negative")

    def evaluate_length(self, length: int) -> float:
        """Compute penalty given a sequence length."""
        cfg = self.config
        if cfg.penalty_type == "threshold":
            excess = max(0, length - cfg.target_length)
            penalty = -cfg.penalty_factor * excess
        elif cfg.penalty_type == "linear":
            penalty = -cfg.penalty_factor * length
        elif cfg.penalty_type == "soft_tanh":
            excess = max(0, length - cfg.target_length)
            penalty = -cfg.max_penalty * math.tanh(cfg.penalty_factor * excess)
        else:
            raise ValueError(f"Unknown penalty_type: {cfg.penalty_type!r}")

        return max(-cfg.max_penalty, penalty)

    def evaluate(self, completion: Any, completion_ids: Any = None) -> float:
        """Evaluate length penalty from tokens or string."""
        if completion_ids is not None:
            length = len(completion_ids)
        elif isinstance(completion, (list, tuple)):
            length = len(completion)
        else:
            # Fallback to word count
            length = len(str(completion).split())
        return self.evaluate_length(length)

    def __call__(
        self,
        prompt: Any,
        completions: Any,
        prompt_ids: Any = None,
        completion_ids: Any = None,
        **kwargs: Any,
    ) -> float:
        return self.evaluate(completions, completion_ids=completion_ids)


class CompositeReward:
    """Composites task accuracy, format verification, and length regulation.

    Computes:
        R = w_acc * R_acc + w_format * R_format + w_length * R_length

    Automatically reports sub-reward components to stats_tracker.
    """

    def __init__(
        self,
        accuracy_fn: Callable[..., Any] | None = None,
        format_reward: FormatReward | None = None,
        length_penalty: LengthPenalty | None = None,
        acc_weight: float = 1.0,
        format_weight: float = 1.0,
        length_weight: float = 1.0,
        log_stats: bool = True,
    ):
        self.accuracy_fn = accuracy_fn
        self.format_reward = format_reward
        self.length_penalty = length_penalty
        self.acc_weight = acc_weight
        self.format_weight = format_weight
        self.length_weight = length_weight
        self.log_stats = log_stats

    def __call__(
        self,
        prompt: Any,
        completions: Any,
        prompt_ids: Any = None,
        completion_ids: Any = None,
        **kwargs: Any,
    ) -> float:
        acc_reward = 0.0
        if self.accuracy_fn is not None:
            try:
                sig = inspect.signature(self.accuracy_fn)
                # Filter kwargs matching the accuracy_fn signature if not **kwargs
                has_var_keyword = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if has_var_keyword:
                    call_kwargs = kwargs
                else:
                    call_kwargs = {
                        k: v for k, v in kwargs.items() if k in sig.parameters
                    }

                acc_result = self.accuracy_fn(
                    prompt,
                    completions,
                    prompt_ids,
                    completion_ids,
                    **call_kwargs,
                )
                acc_reward = float(acc_result)
            except Exception:
                logger.warning("Exception in accuracy_fn", exc_info=True)
                acc_reward = 0.0

        format_reward_val = 0.0
        if self.format_reward is not None:
            format_reward_val = self.format_reward(
                prompt,
                completions,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                **kwargs,
            )

        length_penalty_val = 0.0
        if self.length_penalty is not None:
            length_penalty_val = self.length_penalty(
                prompt,
                completions,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                **kwargs,
            )

        total_reward = (
            self.acc_weight * acc_reward
            + self.format_weight * format_reward_val
            + self.length_weight * length_penalty_val
        )

        if self.log_stats:
            try:
                stats_tracker.scalar(
                    reward_accuracy=float(acc_reward),
                    reward_format=float(format_reward_val),
                    reward_length_penalty=float(length_penalty_val),
                    reward_composite=float(total_reward),
                )
            except Exception:
                pass

        return float(total_reward)


def get_deepseek_r1_math_reward(
    accuracy_fn: Callable[..., Any],
    target_length: int = 2048,
    acc_weight: float = 1.0,
    format_weight: float = 0.5,
    length_weight: float = 0.1,
) -> CompositeReward:
    """Helper creating a standard DeepSeek-R1 style math reasoning reward.

    Enforces <think>...</think> with \\boxed{} answers and bounded length penalties.
    """
    format_reward = FormatReward(
        FormatRewardConfig(
            think_start_tag="<think>",
            think_end_tag="</think>",
            require_think_tags=True,
            require_closed_think=True,
            allow_empty_think=False,
            answer_format="boxed",
            structure_reward=0.5,
            answer_format_reward=0.5,
            malformed_penalty=-0.5,
        )
    )
    length_penalty = LengthPenalty(
        LengthPenaltyConfig(
            target_length=target_length,
            penalty_type="threshold",
            penalty_factor=0.0005,
            max_penalty=0.5,
        )
    )
    return CompositeReward(
        accuracy_fn=accuracy_fn,
        format_reward=format_reward,
        length_penalty=length_penalty,
        acc_weight=acc_weight,
        format_weight=format_weight,
        length_weight=length_weight,
    )
