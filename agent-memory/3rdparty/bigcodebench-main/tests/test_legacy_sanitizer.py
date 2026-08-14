import pytest

from bigcodebench.lecacy_sanitize import sanitize


def test_inline_fn():
    assert (
        sanitize(
            """\
def f(n):
    def factorial(i):
        if i == 0:
            return 1
        else:
            return i * factorial(i-1)

    result = []
    for i in range(1, n+1):
        if i % 2 == 0:
            result.append(factorial(i))
        else:
            result.append(sum(range(1, i+1)))
    return result

# Test the function
print(f(5))""",
            entry_point="f",
        )
        == """\
def f(n):
    def factorial(i):
        if i == 0:
            return 1
        else:
            return i * factorial(i-1)

    result = []
    for i in range(1, n+1):
        if i % 2 == 0:
            result.append(factorial(i))
        else:
            result.append(sum(range(1, i+1)))
    return result"""
    )
