from areal.utils.wrapper import (
    wrap,
    wrap_get_method,
    wrap_get_method_name,
    wrap_remove_meta,
    wrapable,
)


class Calculator:
    def __init__(self):
        self.remaining = 2

    @wrapable()
    def add(self, a, b):
        return a + b + self.remaining

    @wrapable(name="multiply")
    def mul(self, x, y):
        return x * y * self.remaining


class EnhancedCalculator:
    def __init__(self, obj: Calculator):
        super().__init__()
        self.prefix = "Result:"
        self.calculator = obj
        wrap(self, obj, self._wrap_call)

    def _wrap_call(self, *args, **kwargs):
        method_name = wrap_get_method_name(kwargs)
        method = wrap_get_method(kwargs)
        kwargs = wrap_remove_meta(kwargs)
        print("wrap method: ", method_name)
        print("wrap method: ", method)
        return method(*args, **kwargs)


def test_wrapper():
    calc = Calculator()
    enhancer = EnhancedCalculator(calc)

    assert enhancer.add(2, 3) == 7
    assert enhancer.multiply(4, 5) == 40


if __name__ == "__main__":
    test_wrapper()
