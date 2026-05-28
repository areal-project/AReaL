"""Example: use DaytonaRunner to grade model-generated Python code.

Usage:
    export DAYTONA_API_KEY=***
    uv run python examples/sandbox_daytona/reward_example.py
"""

from areal.infra.sandbox import DaytonaRunner


def grade(model_code: str, expected_stdout: str) -> float:
    with DaytonaRunner() as runner:
        result = runner.run(model_code, timeout=5)
        return 1.0 if result.stdout.strip() == expected_stdout.strip() else 0.0


if __name__ == "__main__":
    good = "print(sum(range(10)))"
    bad = "print('oops')"
    print(f"Good reward: {grade(good, '45')}")
    print(f"Bad reward:  {grade(bad, '45')}")
