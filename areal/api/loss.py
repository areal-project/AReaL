# SPDX-License-Identifier: Apache-2.0

"""Scalar means and independently normalized named numerators for engines.

Named callbacks return summed numerators, not local means. Their normalizers
must declare a fixed schema shared by all microbatches and ranks, including
transport-only ranks. Each mass is a batch-derived nonnegative scalar without
gradients. Empty terms retain their name and a graph-connected zero numerator.
"""

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, Protocol, TypeVar

import torch

LossValue = torch.Tensor | dict[str, torch.Tensor]
LossFunction = Callable[..., LossValue]
LossNormalizer = Callable[[dict[str, Any]], LossValue]
_F = TypeVar("_F", bound=Callable[..., Any])


def named_loss_normalizer(*, names: Sequence[str]) -> Callable[[_F], _F]:
    """Declare names without wrapping or executing the normalizer.

    Static names let engines construct zero masses on transport-only ranks
    without passing synthetic data to the objective. All ranks must register
    the same names before entering training or evaluation collectives.
    """
    if isinstance(names, str) or not names:
        raise ValueError("Named loss normalizers require a nonempty sequence of names")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Loss names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Loss names must be unique")
    schema = tuple(sorted(names))

    def decorate(function: _F) -> _F:
        setattr(function, "_areal_named_loss_names", schema)
        return function

    return decorate


def get_named_loss_names(function: Callable[..., Any]) -> tuple[str, ...] | None:
    """Read a schema through partials and bound methods without calling it."""
    while True:
        names = getattr(function, "_areal_named_loss_names", None)
        if names is not None:
            return names
        if not isinstance(function, partial):
            return None
        function = function.func


def is_named_loss_normalizer(function: Callable[..., Any]) -> bool:
    return get_named_loss_names(function) is not None


class TokenLossFn(Protocol):
    """Actor plugin returning differentiable, unreduced token losses.

    ``mask`` is batch-derived support fixed before optimizer splitting. Runtime
    filtering may zero numerator contributions but cannot change normalization.
    ``outputs`` and ``batch`` are read-only mappings; tensors must not be mutated.
    Algorithm parameters are bound once through the configured kwargs.
    """

    def __call__(
        self,
        *,
        outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        mask: torch.Tensor,
    ) -> torch.Tensor: ...
