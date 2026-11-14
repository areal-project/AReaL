"""Example: use RuntaRunner to grade model-generated Python code.

Usage:
    export RUNTA_TOKEN=***
    uv run python examples/sandbox_runta/reward_example.py
"""

from areal.infra.sandbox import RuntaRunner


def grade(runner: RuntaRunner, model_code: str, expected_stdout: str) -> float:
    result = runner.run(model_code, timeout=5)
    return 1.0 if result.stdout.strip() == expected_stdout.strip() else 0.0


if __name__ == "__main__":
    good = "print(sum(range(10)))"
    bad = "print('oops')"
    # One runner is one runtime; reuse it for every submission in a batch.
    with RuntaRunner() as runner:
        print(f"Good reward: {grade(runner, good, '45')}")
        print(f"Bad reward:  {grade(runner, bad, '45')}")
