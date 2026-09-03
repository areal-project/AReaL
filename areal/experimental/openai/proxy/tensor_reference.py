# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from threading import Lock
from typing import Any, Literal

import torch
from pydantic import BaseModel

from areal.utils.data import is_multi_modal_key


class SharedTensorReference(BaseModel):
    """JSON marker for one tensor stored in a proxy-side rollout group."""

    type: Literal["areal_shared_tensor"] = "areal_shared_tensor"
    ref_id: str


class GroupTensorStore:
    """Store unique multimodal tensors and replace them with lightweight refs."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._ref_by_tensor_id: dict[int, str] = {}
        self._tensor_by_ref: dict[str, torch.Tensor] = {}

    def encode_multimodal_tensors(self, value: Any) -> Any:
        """Replace tensors below ``multi_modal_input*`` keys with references."""
        return self._encode(value, in_multimodal_value=False)

    def fetch(self, ref_ids: list[str]) -> dict[str, torch.Tensor]:
        """Return requested tensors, rejecting unknown group-scoped refs."""
        with self._lock:
            missing = [
                ref_id for ref_id in ref_ids if ref_id not in self._tensor_by_ref
            ]
            if missing:
                raise KeyError(f"Unknown shared tensor references: {missing}")
            return {ref_id: self._tensor_by_ref[ref_id] for ref_id in ref_ids}

    def _encode(self, value: Any, *, in_multimodal_value: bool) -> Any:
        if isinstance(value, torch.Tensor):
            if not in_multimodal_value:
                return value
            return self._reference(value).model_dump()

        if isinstance(value, dict):
            return {
                key: self._encode(
                    item,
                    in_multimodal_value=(
                        in_multimodal_value or is_multi_modal_key(str(key))
                    ),
                )
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self._encode(item, in_multimodal_value=in_multimodal_value)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                self._encode(item, in_multimodal_value=in_multimodal_value)
                for item in value
            ]

        return value

    def _reference(self, tensor: torch.Tensor) -> SharedTensorReference:
        tensor_id = id(tensor)
        with self._lock:
            ref_id = self._ref_by_tensor_id.get(tensor_id)
            if ref_id is None:
                ref_id = f"tensor-{len(self._tensor_by_ref)}"
                self._ref_by_tensor_id[tensor_id] = ref_id
                # Keep a strong reference so Python cannot reuse ``tensor_id``
                # during the lifetime of this rollout group.
                self._tensor_by_ref[ref_id] = tensor
        return SharedTensorReference(ref_id=ref_id)


class GroupTensorStoreRegistry:
    """Own proxy-side tensor stores until their rollout groups are finalized."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._stores: dict[str, GroupTensorStore] = {}
        self._last_access: dict[str, float] = {}

    def get_or_create(self, group_id: str) -> GroupTensorStore:
        with self._lock:
            store = self._stores.get(group_id)
            if store is None:
                store = GroupTensorStore()
                self._stores[group_id] = store
            self._last_access[group_id] = time.monotonic()
            return store

    def fetch(self, group_id: str, ref_ids: list[str]) -> dict[str, torch.Tensor]:
        with self._lock:
            store = self._stores.get(group_id)
            if store is None:
                raise KeyError(f"Unknown shared tensor group: {group_id}")
            self._last_access[group_id] = time.monotonic()
        return store.fetch(ref_ids)

    def discard(self, group_id: str) -> None:
        with self._lock:
            self._stores.pop(group_id, None)
            self._last_access.pop(group_id, None)

    def discard_stale(self, timeout_seconds: float) -> None:
        cutoff = time.monotonic() - timeout_seconds
        with self._lock:
            stale_ids = [
                group_id
                for group_id, last_access in self._last_access.items()
                if last_access < cutoff
            ]
            for group_id in stale_ids:
                self._stores.pop(group_id, None)
                self._last_access.pop(group_id, None)


FetchSharedTensors = Callable[[list[str]], Awaitable[dict[str, torch.Tensor]]]


class SharedTensorResolver:
    """Resolve each group-scoped tensor ref once across concurrent sessions."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], torch.Tensor] = {}
        self._inflight: dict[tuple[str, str], asyncio.Future[torch.Tensor]] = {}

    async def resolve(
        self,
        value: Any,
        *,
        group_id: str,
        fetch: FetchSharedTensors,
    ) -> Any:
        """Fetch missing references once, then restore aliases in ``value``."""
        ref_ids: list[str] = []
        self._collect_ref_ids(value, ref_ids, in_multimodal_value=False)
        if not ref_ids:
            return value

        owned: list[str] = []
        futures: dict[str, asyncio.Future[torch.Tensor]] = {}
        loop = asyncio.get_running_loop()
        for ref_id in dict.fromkeys(ref_ids):
            key = (group_id, ref_id)
            cached = self._values.get(key)
            if cached is not None:
                future = loop.create_future()
                future.set_result(cached)
            else:
                future = self._inflight.get(key)
                if future is None:
                    future = loop.create_future()
                    self._inflight[key] = future
                    owned.append(ref_id)
            futures[ref_id] = future

        if owned:
            try:
                fetched = await fetch(owned)
                missing = [ref_id for ref_id in owned if ref_id not in fetched]
                if missing:
                    raise RuntimeError(
                        f"Proxy omitted shared tensor references: {missing}"
                    )
                invalid = [
                    ref_id
                    for ref_id in owned
                    if not isinstance(fetched[ref_id], torch.Tensor)
                ]
                if invalid:
                    raise TypeError(
                        f"Proxy returned non-tensor shared references: {invalid}"
                    )
            except BaseException as exc:
                for ref_id in owned:
                    key = (group_id, ref_id)
                    future = self._inflight.pop(key)
                    if not future.done():
                        future.set_exception(exc)
                        # The owner raises directly; retrieving the exception
                        # prevents an unobserved-future warning without hiding it
                        # from concurrent waiters.
                        future.exception()
                raise

            for ref_id in owned:
                key = (group_id, ref_id)
                tensor = fetched[ref_id]
                self._values[key] = tensor
                future = self._inflight.pop(key)
                future.set_result(tensor)

        resolved = {
            ref_id: await asyncio.shield(future) for ref_id, future in futures.items()
        }
        return self._replace_refs(value, resolved, in_multimodal_value=False)

    def discard(self, group_id: str) -> None:
        keys = [key for key in self._values if key[0] == group_id]
        for key in keys:
            self._values.pop(key, None)

        inflight_keys = [key for key in self._inflight if key[0] == group_id]
        if inflight_keys:
            raise RuntimeError(
                f"Cannot discard shared tensor group {group_id} while fetches are active"
            )

    @classmethod
    def _collect_ref_ids(
        cls,
        value: Any,
        result: list[str],
        *,
        in_multimodal_value: bool,
    ) -> None:
        if in_multimodal_value and cls._is_reference(value):
            result.append(value["ref_id"])
            return
        if isinstance(value, dict):
            for key, item in value.items():
                cls._collect_ref_ids(
                    item,
                    result,
                    in_multimodal_value=(
                        in_multimodal_value or is_multi_modal_key(str(key))
                    ),
                )
        elif isinstance(value, list):
            for item in value:
                cls._collect_ref_ids(
                    item,
                    result,
                    in_multimodal_value=in_multimodal_value,
                )

    @classmethod
    def _replace_refs(
        cls,
        value: Any,
        resolved: dict[str, torch.Tensor],
        *,
        in_multimodal_value: bool,
    ) -> Any:
        if in_multimodal_value and cls._is_reference(value):
            return resolved[value["ref_id"]]
        if isinstance(value, dict):
            return {
                key: cls._replace_refs(
                    item,
                    resolved,
                    in_multimodal_value=(
                        in_multimodal_value or is_multi_modal_key(str(key))
                    ),
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                cls._replace_refs(
                    item,
                    resolved,
                    in_multimodal_value=in_multimodal_value,
                )
                for item in value
            ]
        return value

    @staticmethod
    def _is_reference(value: Any) -> bool:
        return (
            isinstance(value, dict)
            and value.get("type") == "areal_shared_tensor"
            and isinstance(value.get("ref_id"), str)
        )
