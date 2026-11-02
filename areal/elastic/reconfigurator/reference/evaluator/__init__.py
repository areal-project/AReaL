# SPDX-License-Identifier: Apache-2.0

"""Reference candidate evaluation; event simulation is an internal detail."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from areal.elastic.reconfigurator.reference.evaluator.evaluator import Evaluator
__all__ = ["Evaluator"]


def __getattr__(name):
    if name == "Evaluator":
        from areal.elastic.reconfigurator.reference.evaluator.evaluator import Evaluator

        globals()[name] = Evaluator
        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
