# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading

from areal.utils import logging
from areal.v2.weight_update.gateway.config import PairInfo

logger = logging.getLogger("PairRegistry")


class PairRegistry:
    def __init__(self) -> None:
        self._by_name: dict[str, PairInfo] = {}
        self._pending: set[str] = set()
        self._lock = threading.Lock()

    def try_reserve(self, pair_name: str) -> bool:
        """Atomically reserve a name before asynchronous pair initialization."""
        with self._lock:
            if pair_name in self._by_name or pair_name in self._pending:
                return False
            self._pending.add(pair_name)
            return True

    def register_reserved(self, pair_info: PairInfo) -> None:
        """Commit a pair name previously acquired with :meth:`try_reserve`."""
        with self._lock:
            pair_name = pair_info.pair_name
            if pair_name not in self._pending:
                raise ValueError(f"Pair '{pair_name}' is not reserved")
            if pair_name in self._by_name:
                raise ValueError(f"Pair '{pair_name}' already registered")
            self._by_name[pair_name] = pair_info
            self._pending.remove(pair_name)
            logger.info("Registered pair '%s'", pair_name)

    def release_reservation(self, pair_name: str) -> None:
        """Release an uncommitted pair-name reservation after initialization."""
        with self._lock:
            self._pending.discard(pair_name)

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
