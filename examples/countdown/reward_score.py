import ast
import operator
import random
import re
from collections import Counter
from fractions import Fraction


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    solution_str = solution_str.split("\n")[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        final_answer = matches[-1].strip()
    else:
        final_answer = None
    return final_answer


# Bound parsing and exact arithmetic even when these helpers are called directly.
_MAX_EQUATION_LENGTH = 4096
_MAX_AST_NODES = 256
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _parse_equation(equation_str: str) -> ast.Expression:
    """Accept integer operands, arithmetic operators and parentheses only."""
    if len(equation_str) > _MAX_EQUATION_LENGTH or not re.fullmatch(
        r"[0-9+\-*/()\s]+", equation_str
    ):
        raise ValueError("Invalid characters or oversized equation.")
    tree = ast.parse(equation_str.strip(), mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        raise ValueError("Equation is too complex.")
    for node in nodes:
        if isinstance(node, ast.Constant):
            if type(node.value) is not int:
                raise ValueError("Only integer operands are allowed.")
        elif type(node) not in (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.UAdd,
            ast.USub,
            *_BINARY_OPERATORS,
        ):
            raise ValueError("Only +, -, * and / are allowed.")
    return tree


def validate_equation(equation_str: str, available_numbers: list[int]) -> bool:
    """Require valid arithmetic using each supplied integer exactly once."""
    try:
        tree = _parse_equation(equation_str)
        numbers = [
            node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
        ]
        return Counter(numbers) == Counter(available_numbers)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return False


def _evaluate_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant):
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        return _BINARY_OPERATORS[type(node.op)](
            _evaluate_node(node.left), _evaluate_node(node.right)
        )
    raise ValueError("Unsupported arithmetic expression.")


def evaluate_equation(equation_str: str) -> Fraction | None:
    """Evaluate restricted arithmetic without executing Python code.

    Division may produce fractional intermediate values, but decimal literals,
    exponentiation and floor division are not part of the Countdown rules.
    """
    try:
        return _evaluate_node(_parse_equation(equation_str).body)
    except (SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError):
        return None


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.1, score=1.0
):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print("No equation found")
        return 0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print("Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print("Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except Exception:
        if do_print:
            print("Error evaluating equation")
        return format_score
