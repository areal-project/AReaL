# SPDX-License-Identifier: Apache-2.0

import math

from areal.reward import (
    CompositeReward,
    FormatReward,
    FormatRewardConfig,
    LengthPenalty,
    LengthPenaltyConfig,
    extract_boxed_content,
    extract_hash_answer,
    extract_reasoning_and_answer,
    extract_tag_content,
    get_deepseek_r1_math_reward,
)


def test_extract_boxed_content():
    assert extract_boxed_content(r"The result is \boxed{42}.") == "42"
    assert extract_boxed_content(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert (
        extract_boxed_content(r"\boxed{\sqrt{\frac{a}{b}}}")
        == r"\sqrt{\frac{a}{b}}"
    )
    assert (
        extract_boxed_content(r"First \boxed{10}, then finally \boxed{20}")
        == "20"
    )
    assert extract_boxed_content("No boxed here") is None
    assert extract_boxed_content(r"\boxed{unclosed brace") is None


def test_extract_tag_content():
    assert (
        extract_tag_content("<answer>42</answer>", tag="answer")
        == "42"
    )
    assert (
        extract_tag_content(
            "<answer>\nline 1\nline 2\n</answer>", tag="answer"
        )
        == "line 1\nline 2"
    )
    assert (
        extract_tag_content(
            "<custom>my answer</custom>", tag="custom"
        )
        == "my answer"
    )
    assert extract_tag_content("No tag here", tag="answer") is None


def test_extract_hash_answer():
    assert extract_hash_answer("The final answer is #### 42") == "42"
    assert extract_hash_answer("Step 1 #### 10 \n Step 2 #### 20") == "20"
    assert extract_hash_answer("No delimiter") is None


def test_extract_reasoning_and_answer():
    # Normal case with <think> and </think>
    text = "<think>\nLet's calculate 2 + 2 = 4.\n</think>\nTherefore the answer is 4."
    reasoning, answer = extract_reasoning_and_answer(text)
    assert reasoning == "Let's calculate 2 + 2 = 4."
    assert answer == "Therefore the answer is 4."

    # Truncated inside thinking block
    text_unclosed = "<think>\nLet's calculate step 1..."
    reasoning_u, answer_u = extract_reasoning_and_answer(text_unclosed)
    assert reasoning_u == "Let's calculate step 1..."
    assert answer_u == ""

    # Implicit thinking before </think>
    text_implicit = "Let me think.\n</think>\nAnswer is 42."
    reasoning_i, answer_i = extract_reasoning_and_answer(text_implicit)
    assert reasoning_i == "Let me think."
    assert answer_i == "Answer is 42."

    # No thinking tags
    text_none = "Direct answer without tags."
    reasoning_n, answer_n = extract_reasoning_and_answer(text_none)
    assert reasoning_n == ""
    assert answer_n == "Direct answer without tags."


def test_format_reward_valid_boxed():
    formatter = FormatReward(
        FormatRewardConfig(
            think_start_tag="<think>",
            think_end_tag="</think>",
            require_think_tags=True,
            require_closed_think=True,
            answer_format="boxed",
            structure_reward=0.5,
            answer_format_reward=0.5,
            malformed_penalty=-0.5,
        )
    )
    completion = "<think>\nStep by step reasoning.\n</think>\nThe answer is \\boxed{42}."
    detail = formatter.evaluate_detailed(completion)
    assert detail["valid_think"] is True
    assert detail["valid_answer_format"] is True
    assert detail["extracted_answer"] == "42"
    assert detail["reward"] == 1.0
    assert formatter.evaluate(completion) == 1.0


def test_format_reward_unclosed_or_missing_think():
    formatter = FormatReward(
        FormatRewardConfig(
            require_think_tags=True,
            require_closed_think=True,
            malformed_penalty=-0.5,
        )
    )
    # Missing tags entirely
    assert (
        formatter.evaluate("Direct answer \\boxed{42}") == -0.5
    )
    # Unclosed think tag
    assert (
        formatter.evaluate("<think>Let's think \\boxed{42}")
        == -0.5
    )
    # Multiple start tags
    assert (
        formatter.evaluate("<think>A</think><think>B</think> \\boxed{42}")
        == -0.5
    )


def test_format_reward_empty_think_rejected():
    formatter = FormatReward(
        FormatRewardConfig(
            require_think_tags=True,
            allow_empty_think=False,
            malformed_penalty=-0.5,
        )
    )
    assert (
        formatter.evaluate("<think></think> \\boxed{42}") == -0.5
    )
    assert (
        formatter.evaluate("<think>   \n  </think> \\boxed{42}")
        == -0.5
    )


def test_format_reward_xml_and_hash_formats():
    xml_formatter = FormatReward(
        FormatRewardConfig(
            answer_format="xml",
            answer_tag="solution",
            structure_reward=0.3,
            answer_format_reward=0.7,
        )
    )
    comp_xml = "<think>reasoning</think><solution>x=5</solution>"
    assert xml_formatter.evaluate(comp_xml) == 1.0

    hash_formatter = FormatReward(
        FormatRewardConfig(
            answer_format="hash",
            structure_reward=0.4,
            answer_format_reward=0.6,
        )
    )
    comp_hash = "<think>reasoning</think>Final: #### 100"
    assert hash_formatter.evaluate(comp_hash) == 1.0


def test_length_penalty_threshold():
    penalty = LengthPenalty(
        LengthPenaltyConfig(
            target_length=100,
            penalty_type="threshold",
            penalty_factor=0.01,
            max_penalty=1.0,
        )
    )
    # Under or equal to target length -> 0 penalty
    assert penalty.evaluate_length(50) == 0.0
    assert penalty.evaluate_length(100) == 0.0

    # Over target length -> linear excess penalty
    assert math.isclose(penalty.evaluate_length(150), -0.5)
    # Capped at max_penalty
    assert penalty.evaluate_length(300) == -1.0


def test_length_penalty_linear_and_soft_tanh():
    linear_penalty = LengthPenalty(
        LengthPenaltyConfig(
            penalty_type="linear",
            penalty_factor=0.005,
            max_penalty=1.0,
        )
    )
    assert math.isclose(linear_penalty.evaluate_length(100), -0.5)

    tanh_penalty = LengthPenalty(
        LengthPenaltyConfig(
            target_length=50,
            penalty_type="soft_tanh",
            penalty_factor=0.02,
            max_penalty=0.8,
        )
    )
    assert tanh_penalty.evaluate_length(30) == 0.0
    expected = -0.8 * math.tanh(0.02 * (100 - 50))
    assert math.isclose(tanh_penalty.evaluate_length(100), expected)


def test_length_penalty_with_token_ids():
    penalty = LengthPenalty(
        LengthPenaltyConfig(
            target_length=5,
            penalty_type="threshold",
            penalty_factor=0.1,
            max_penalty=1.0,
        )
    )
    tokens = [101, 102, 103, 104, 105, 106, 107]  # len = 7, excess = 2
    res = penalty(prompt="Q", completions="A", completion_ids=tokens)
    assert math.isclose(res, -0.2)


def test_composite_reward_combination():
    def mock_accuracy(prompt, completions, *args, **kwargs):
        return 1.0 if "42" in completions else 0.0

    format_r = FormatReward(
        FormatRewardConfig(
            structure_reward=0.5,
            answer_format_reward=0.5,
            malformed_penalty=-0.5,
        )
    )
    length_p = LengthPenalty(
        LengthPenaltyConfig(
            target_length=10,
            penalty_type="threshold",
            penalty_factor=0.01,
        )
    )

    composite = CompositeReward(
        accuracy_fn=mock_accuracy,
        format_reward=format_r,
        length_penalty=length_p,
        acc_weight=1.0,
        format_weight=0.5,
        length_weight=1.0,
        log_stats=True,
    )

    # Correct answer, correct format, short length
    comp_good = "<think>ok</think> \\boxed{42}"
    tokens = [1, 2, 3]  # len=3 <= 10
    score = composite("Q", comp_good, completion_ids=tokens)
    # acc (1.0 * 1.0) + format (1.0 * 0.5) + length (0.0 * 1.0) = 1.5
    assert math.isclose(score, 1.5)

    # Wrong answer, correct format
    comp_wrong = "<think>ok</think> \\boxed{99}"
    score_wrong = composite("Q", comp_wrong, completion_ids=tokens)
    # acc (0.0) + format (0.5) + length (0.0) = 0.5
    assert math.isclose(score_wrong, 0.5)


def test_get_deepseek_r1_math_reward_factory():
    def mock_acc(prompt, completions, *args, **kwargs):
        return 1.0

    r1_reward = get_deepseek_r1_math_reward(
        accuracy_fn=mock_acc,
        target_length=100,
        acc_weight=1.0,
        format_weight=0.5,
        length_weight=0.1,
    )
    assert isinstance(r1_reward, CompositeReward)
    comp = "<think>reasoning</think>\\boxed{10}"
    val = r1_reward("Q", comp, completion_ids=[1] * 50)
    # acc (1.0 * 1.0) + format (1.0 * 0.5) - length (0) = 1.5
    assert math.isclose(val, 1.5)
