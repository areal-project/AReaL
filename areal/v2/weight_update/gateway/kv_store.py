# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from typing import Any

from areal.utils import logging

logger = logging.getLogger("WeightMetaStore")


class WeightMetaStore:
    """Thread-safe in-memory KV store for metadata exchange.

    Values are stored as-is (Python objects). Serialization/deserialization
    is handled at the HTTP boundary, not inside the store.
    Each key is scoped by pair_name to support multiple concurrent pairs.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._sets: dict[str, dict[str, set]] = {}
        self._lock = threading.Lock()

    def get(self, pair_name: str, key: str) -> Any | None:
        """Get value for key under pair_name. Returns None if not found."""
        with self._lock:
            return self._data.get(pair_name, {}).get(key)

    def put(self, pair_name: str, key: str, value: Any) -> None:
        """Store value under pair_name/key."""
        with self._lock:
            self._data.setdefault(pair_name, {})[key] = value

    def delete(self, pair_name: str, key: str) -> bool:
        """Delete key from pair. Returns True if key existed."""
        with self._lock:
            pair_data = self._data.get(pair_name, {})
            if key in pair_data:
                del pair_data[key]
                return True
            return False

    def add_to_set(self, pair_name: str, key: str, value: str) -> None:
        """Add value to a set under pair_name/key. Used for barrier sync."""
        with self._lock:
            self._sets.setdefault(pair_name, {}).setdefault(key, set()).add(value)

    def set_size(self, pair_name: str, key: str) -> int:
        """Get size of set under pair_name/key. Returns 0 if not found."""
        with self._lock:
            return len(self._sets.get(pair_name, {}).get(key, set()))

    def clear_pair(self, pair_name: str) -> None:
        """Remove all data and sets for a pair."""
        with self._lock:
            self._data.pop(pair_name, None)
            self._sets.pop(pair_name, None)
            logger.info("Cleared all data for pair '%s'", pair_name)

    def list_keys(self, pair_name: str) -> list[str]:
        """List all keys for a pair."""
        with self._lock:
            return list(self._data.get(pair_name, {}).keys())
