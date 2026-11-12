import builtins
import itertools
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from examples.countdown import reward_score
from examples.countdown.reward_score import (
    compute_score,
    evaluate_equation,
    validate_equation,
)


@pytest.mark.parametrize(
    "expression,numbers,target",
    [
        ("5 + 4 - 2 * 1", [1, 5, 2, 4], 7),
        ("8 / (3 - 8 / 3)", [8, 3, 8, 3], 24),
        ("(1 + 2) / 4", [1, 2, 4], Fraction(3, 4)),
        ("-2 + 3", [2, 3], 1),
        ("+2 * (3 - 1)", [2, 3, 1], 4),
        (" 12 / 3 + 4 ", [12, 3, 4], 8),
    ],
)
def test_valid_arithmetic_receives_full_reward(expression, numbers, target):
    """Preserve precedence, duplicate operands, unary signs and fractions."""
    assert validate_equation(expression, numbers)
    assert evaluate_equation(expression) == target
    assert (
        compute_score(
            f"<answer>{expression}</answer>", {"numbers": numbers, "target": target}
        )
        == 1.0
    )


@pytest.mark.parametrize(
    "expression,numbers,target",
    [
        ("1.5 * 2", [1, 5, 2], 3),
        ("1.5 * 2 + 4", [1, 5, 2, 4], 7),
        ("7 // 2", [7, 2], 3),
        ("3**2*5-7", [3, 2, 5, 7], 38),
        (".5 * 2", [5, 2], 1),
        ("5. + 2", [5, 2], 7),
        ("1e2 + 3", [1, 2, 3], 103),
        ("0x10 + 2", [0, 10, 2], 18),
        ("1_2 + 3", [1, 2, 3], 15),
        ("7 % 2", [7, 2], 1),
        ("(1, 2)", [1, 2], 3),
        ("True + 2", [2], 3),
        ("abs(-2)", [2], 2),
        ("2 # comment", [2], 2),
        ("2 +", [2], 2),
    ],
)
def test_invalid_syntax_receives_format_reward(expression, numbers, target):
    """Digit matching must not admit extra Python syntax or decimal operands."""
    assert not validate_equation(expression, numbers)
    assert evaluate_equation(expression) is None
    assert (
        compute_score(
            f"<answer>{expression}</answer>", {"numbers": numbers, "target": target}
        )
        == 0.1
    )


@pytest.mark.parametrize(
    "expression,numbers",
    [("2 + 3", [2, 3, 4]), ("2 + 2 + 3", [2, 3]), ("2 + 4", [2, 3]), ("23", [2, 3])],
)
def test_missing_reused_or_concatenated_numbers_receive_format_reward(
    expression, numbers
):
    """Require the operand multiset, not just a subset of supplied numbers."""
    assert not validate_equation(expression, numbers)
    assert (
        compute_score(
            f"<answer>{expression}</answer>", {"numbers": numbers, "target": 5}
        )
        == 0.1
    )


def test_zero_division_receives_format_reward():
    """Invalid arithmetic must not abort reward computation."""
    assert evaluate_equation("1 / (2 - 2)") is None
    assert (
        compute_score(
            "<answer>1 / (2 - 2)</answer>", {"numbers": [1, 2, 2], "target": 1}
        )
        == 0.1
    )


def test_score_levels_and_answer_extraction_are_preserved():
    """Preserve missing-answer, format and configurable success rewards."""
    truth = {"numbers": [2, 3], "target": 5}
    assert compute_score("2 + 3", truth) == 0
    assert compute_score("<answer>2 * 3</answer>", truth, format_score=0.25) == 0.25
    assert (
        compute_score("Assistant: reasoning\n<answer>2 + 3</answer>", truth, score=2)
        == 2
    )
    assert compute_score("<answer>2 * 3</answer> <answer>2 + 3</answer>", truth) == 1


@pytest.mark.parametrize(
    "expression", ["1" * 4097, " + ".join(["1"] * 100), "(" * 300 + "1" + ")" * 300]
)
def test_oversized_or_deep_expression_is_rejected(expression):
    """Bound parser input and arithmetic work for malformed completions."""
    assert not validate_equation(expression, [1])
    assert evaluate_equation(expression) is None


def test_power_chain_is_rejected_within_subprocess_timeout():
    """Guard the reported hang without risking an unbounded pytest process."""
    code = """
from examples.countdown.reward_score import compute_score, evaluate_equation, validate_equation
expression = '44**19**35'
assert not validate_equation(expression, [44, 19, 35])
assert evaluate_equation(expression) is None
assert compute_score('<answer>' + expression + '</answer>', {'numbers': [44, 19, 35], 'target': 1}) == 0.1
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=5,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def quiet_scores(monkeypatch):
    """Avoid random debug output while checking thousands of reward decisions."""
    monkeypatch.setattr(reward_score.random, "randint", lambda *_: 2)


def _all_arithmetic_trees(numbers):
    """Construct expressions and rational answers without parsing or eval.

    Enumerate every binary tree shape and operator assignment for this ordered
    operand tuple. None represents an undefined division anywhere in the tree.
    """
    if len(numbers) == 1:
        yield str(numbers[0]), Fraction(numbers[0])
        return
    for split in range(1, len(numbers)):
        for left, a in _all_arithmetic_trees(numbers[:split]):
            for right, b in _all_arithmetic_trees(numbers[split:]):
                for op in "+-*/":
                    value = None
                    if a is not None and b is not None:
                        if op == "+":
                            value = a + b
                        elif op == "-":
                            value = a - b
                        elif op == "*":
                            value = a * b
                        elif b != 0:
                            value = a / b
                    yield f"({left}{op}{right})", value


@pytest.mark.parametrize(
    "arity,expected_count", [(1, 3), (2, 36), (3, 864), (4, 25920)]
)
def test_exhaustive_small_integer_arithmetic(arity, expected_count, quiet_scores):
    """Check all trees/operators on 1-4 operands from {1,2,3}: 26,823 cases."""
    count = 0
    for numbers in itertools.product((1, 2, 3), repeat=arity):
        for expression, expected in _all_arithmetic_trees(numbers):
            count += 1
            assert validate_equation(expression, list(numbers)), expression
            assert evaluate_equation(expression) == expected, expression
            truth = {
                "numbers": list(numbers),
                "target": 0 if expected is None else expected,
            }
            answer = f"<answer>{expression}</answer>"
            assert compute_score(answer, truth) == (0.1 if expected is None else 1.0), (
                expression
            )
            if expected is not None:
                truth["target"] = expected + 1
                assert compute_score(answer, truth) == 0.1, expression
    # Guard against accidentally narrowing the exhaustive generator.
    assert count == expected_count


def test_exhaustive_operand_multisets(quiet_scores):
    """Check duplicates and order, plus missing, extra and replaced operands."""
    for arity in range(1, 5):
        for numbers in itertools.product((1, 2, 3, 4), repeat=arity):
            expression = "+".join(map(str, numbers))
            assert validate_equation(expression, list(reversed(numbers)))
            for invalid in (numbers[:-1], (*numbers, 99), (99, *numbers[1:])):
                assert not validate_equation(expression, list(invalid))
                assert (
                    compute_score(
                        f"<answer>{expression}</answer>",
                        {"numbers": list(invalid), "target": sum(numbers)},
                    )
                    == 0.1
                )


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("8-3-2", 3),
        ("8/4/2", 1),
        ("8/4*2", 4),
        ("8-3*2", 2),
        ("8/4+2", 4),
        ("8/(4/2)", 4),
        ("1--2", 3),
        ("1+-2", -1),
        ("-(1+2)*3", -9),
        ("1 / -2", Fraction(-1, 2)),
        ("1/(2-3)+1", 0),
        ("1/3+1/3+1/3", 1),
        ("(\n 1\t+\r\n2\n) / 3", 1),
        ("10000000000000001-10000000000000000", 1),
    ],
)
def test_precedence_signs_whitespace_and_exact_arithmetic(expression, expected):
    """Preserve Python arithmetic precedence without using Python evaluation."""
    assert evaluate_equation(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        " ",
        "()",
        "1 2",
        "1 + (2",
        "1 + 2)",
        "1/**/2",
        "1 << 2",
        "1 >> 2",
        "1 & 2",
        "1 | 2",
        "1 ^ 2",
        "~1",
        "1 @ 2",
        "1 == 2",
        "1 < 2",
        "1 != 2",
        "1 and 2",
        "1 or 2",
        "not 1",
        "1 if 2 else 3",
        "x",
        "x := 1",
        "[1,2]",
        "{1,2}",
        "{'x': 1}",
        "(1).__class__",
        "[1][0]",
        "(lambda: 1)()",
        "'1'",
        'f"{1}"',
        "__import__('os').getcwd()",
        "sum((1,2))",
        "False",
        "None",
        "1;2",
        "1#2",
        "1\\\n+2",
        "1\x00+2",
        "1\u200b+2",
        "１+２",
        "١+٢",
        "1−2",
        "1×2",
        "1÷2",
        "NaN",
        "inf",
        "0b10",
        "0o10",
        "2j",
        "1E2",
        "1e-2",
        "01+2",
        "1...2",
    ],
)
def test_non_arithmetic_syntax_is_rejected(expression, quiet_scores):
    """Reject Python syntax and alternate numeric spellings before evaluation."""
    assert not validate_equation(expression, [1, 2])
    assert evaluate_equation(expression) is None
    # Use a single-line answer wrapper: extraction is intentionally unchanged.
    if "\n" not in expression:
        assert (
            compute_score(
                f"<answer>{expression}</answer>",
                {"numbers": [1, 2], "target": 3},
                format_score=0.25,
            )
            == 0.25
        )


@pytest.mark.parametrize("value", [None, 1, 1.5, True, b"1+2", [], {}])
def test_non_string_equations_fail_closed(value):
    """The public arithmetic helpers must not raise for non-text input."""
    assert not validate_equation(value, [1, 2])
    assert evaluate_equation(value) is None


def test_equation_length_boundary():
    """Accept the documented length limit and reject one character beyond it."""
    expression = "1" + " " * (reward_score._MAX_EQUATION_LENGTH - 1)
    assert validate_equation(expression, [1])
    assert evaluate_equation(expression) == 1
    assert not validate_equation(expression + " ", [1])
    assert evaluate_equation(expression + " ") is None


def test_ast_node_limit_boundary():
    """Exercise the actual boundary without hardcoding the configured limit."""
    # Expression + N constants + (N-1) BinOp + (N-1) Add = 3N-1 nodes.
    count = (reward_score._MAX_AST_NODES + 1) // 3
    expression = "+".join(["1"] * count)
    assert validate_equation(expression, [1] * count)
    assert evaluate_equation(expression) == count
    assert not validate_equation(expression + "+1", [1] * (count + 1))
    assert evaluate_equation(expression + "+1") is None


def test_verifier_does_not_call_eval(monkeypatch, quiet_scores):
    """Keep successful scoring independent of executable-string evaluation."""

    def forbidden_eval(*args, **kwargs):
        raise AssertionError("Countdown must not execute eval")

    monkeypatch.setattr(builtins, "eval", forbidden_eval)
    assert evaluate_equation("8/(3-8/3)") == 24
    assert (
        compute_score(
            "<answer>8/(3-8/3)</answer>", {"numbers": [8, 3, 8, 3], "target": 24}
        )
        == 1.0
    )


@pytest.mark.parametrize(
    "prefix", ["", "Assistant: reasoning\n", "<|im_start|>assistant\n"]
)
def test_supported_answer_wrappers_preserve_rewards(prefix, quiet_scores):
    """Check the existing plain and chat-template completion entry points."""
    truth = {"numbers": [1, 2], "target": 3}
    assert compute_score(prefix + "<answer>1+2</answer>", truth) == 1.0
    assert compute_score(prefix + "<answer>1*2</answer>", truth) == 0.1
    assert compute_score(prefix + "no final answer", truth) == 0


def test_invalid_last_answer_cannot_reuse_earlier_valid_answer(quiet_scores):
    """Never fall back to an earlier answer when the final answer is invalid."""
    truth = {"numbers": [1, 5, 2], "target": 3}
    assert compute_score("<answer>5-2*1</answer> <answer>1.5*2</answer>", truth) == 0.1
