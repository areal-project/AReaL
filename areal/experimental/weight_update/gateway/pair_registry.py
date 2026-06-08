# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading

from areal.experimental.weight_update.gateway.config import PairInfo
from areal.utils import logging

logger = logging.getLogger("PairRegistry")


class PairRegistry:
    def __init__(self) -> None:
        self._by_name: dict[str, PairInfo] = {}
        self._lock = threading.Lock()

    def register(self, pair_info: PairInfo) -> None:
        with self._lock:
            if pair_info.pair_name in self._by_name:
                raise ValueError(f"Pair '{pair_info.pair_name}' already registered")
            self._by_name[pair_info.pair_name] = pair_info
            logger.info("Registered pair '%s'", pair_info.pair_name)

    def get_by_name(self, pair_name: str) -> PairInfo | None:
        with self._lock:
            return self._by_name.get(pair_name)

    def unregister(self, pair_name: str) -> PairInfo | None:
        with self._lock:
            pair_info = self._by_name.pop(pair_name, None)
            if pair_info is not None:
                logger.info("Unregistered pair '%s'", pair_name)
            return pair_info

    def list_pairs(self) -> list[str]:
        with self._lock:
            return list(self._by_name.keys())
