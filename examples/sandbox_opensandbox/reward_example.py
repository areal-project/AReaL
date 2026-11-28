"""Example: use OpenSandboxRunner to grade model-generated Python code.

Usage:
    export OPEN_SANDBOX_DOMAIN=...        # management API host, e.g. sandbox.example.com:8080
    export OPEN_SANDBOX_API_KEY=***       # omit when the endpoint needs no key
    uv run python examples/sandbox_opensandbox/reward_example.py
"""

from areal.infra.sandbox import OpenSandboxRunner


def grade(model_code: str, expected_stdout: str) -> float:
    with OpenSandboxRunner() as runner:
        result = runner.run(model_code, timeout=5)
        return 1.0 if result.stdout.strip() == expected_stdout.strip() else 0.0


if __name__ == "__main__":
    good = "print(sum(range(10)))"
    bad = "print('oops')"
    print(f"Good reward: {grade(good, '45')}")
    print(f"Bad reward:  {grade(bad, '45')}")
