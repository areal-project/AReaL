# SPDX-License-Identifier: Apache-2.0

"""Coordinate AWEX metadata construction with SGLang's GC object scans."""

import threading
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")
_metadata_gc_lock = threading.Lock()


def serialize_metadata_gc(func: Callable[_P, _T]) -> Callable[_P, _T]:
    """Keep GC scans from retaining partially constructed metadata tuples.

    CPython's tuple(iterator) may resize an unfinished tuple. gc.get_objects()
    in another thread can retain that tuple, violating the resize refcount
    requirement (CPython issue 15108). SGLang scans objects while freezing GC
    after server startup, concurrently with AWEX's metadata worker.
    """

    @wraps(func)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        with _metadata_gc_lock:
            return func(*args, **kwargs)

    return guarded
