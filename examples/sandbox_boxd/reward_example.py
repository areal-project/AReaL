"""Example: use BoxdRunner to grade model-generated Python code.

Usage:
    export BOXD_API_KEY=***
    uv run python examples/sandbox_boxd/reward_example.py
"""

from areal.infra.sandbox import BoxdRunner


def grade(runner: BoxdRunner, model_code: str, expected_stdout: str) -> float:
    result = runner.run(model_code, timeout=5)
    return 1.0 if result.stdout.strip() == expected_stdout.strip() else 0.0


if __name__ == "__main__":
    good = "print(sum(range(10)))"
    bad = "print('oops')"
    with BoxdRunner() as runner:
        print(f"Good reward: {grade(runner, good, '45')}")
        print(f"Bad reward:  {grade(runner, bad, '45')}")
