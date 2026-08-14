"""
The function is copied from python-retry~=0.0.1.
The original code can be accessed using: from python_retry import retry.
The modified code provides better type hinting.
"""

import functools
import time
from typing import Callable, Optional, ParamSpec, TypeVar, final
from abc import ABC, abstractmethod

from .logger import SafeLogger


class BackoffStrategyInterface(ABC):
    def __init__(
        self, interval: tuple[Optional[int], Optional[int]] = (None, None)
    ) -> None:
        """
        The default value of `interval` is (None, None), which means that the interval is not set.
        """
        self.interval = interval

    @final
    def calculate(self, retry_count: int) -> float:
        backoff_time = self._calculate(retry_count)
        if self.interval[0] is not None:
            backoff_time = max(backoff_time, self.interval[0])
        if self.interval[1] is not None:
            backoff_time = min(backoff_time, self.interval[1])
        if backoff_time < 0:
            backoff_time = 0
        return backoff_time

    @abstractmethod
    def _calculate(self, retry_count: int) -> float:
        pass


class ExponentialBackoffStrategy(BackoffStrategyInterface):
    def __init__(
        self,
        exponent_base: float = 2,
        multiplier: float = 1,
        interval: tuple[Optional[int], Optional[int]] = (None, None),
    ) -> None:
        super().__init__(interval)
        self.exponent_base = exponent_base
        self.multiplier = multiplier

    def _calculate(self, retry_index: int) -> float:
        # retry_index starts from 0
        backoff_time = self.multiplier * (self.exponent_base**retry_index)
        return backoff_time


# https://stackoverflow.com/a/68290080
Param = ParamSpec("Param")
RetType = TypeVar("RetType")


class RetryHandler:
    @staticmethod
    def handle(
        max_retries: int = 3,
        waiting_strategy: BackoffStrategyInterface = ExponentialBackoffStrategy(),
        retry_on: Optional[tuple[type[Exception], ...]] = None,
    ) -> Callable[[Callable[Param, RetType]], Callable[Param, RetType]]:
        def decorator(func: Callable[Param, RetType]) -> Callable[Param, RetType]:
            # https://stackoverflow.com/a/309000
            @functools.wraps(func)
            def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
                for n in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except retry_on or Exception as e:
                        # return with/without Exception
                        if n == max_retries:
                            SafeLogger.error(f"{e}, retried has been exhausted...")
                            raise e
                        # time sleep
                        seconds = waiting_strategy.calculate(n)
                        SafeLogger.warning(f"{e}, retrying in {seconds} seconds...")
                        time.sleep(seconds)
                raise RuntimeError("This should never be reached")

            return wrapper

        return decorator
