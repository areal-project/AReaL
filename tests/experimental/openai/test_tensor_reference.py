# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from areal.experimental.openai.proxy.server import deserialize_interactions
from areal.experimental.openai.proxy.tensor_reference import (
    GroupTensorStore,
    GroupTensorStoreRegistry,
    SharedTensorResolver,
)
from areal.utils.data import concat_padded_tensors


def _trajectory(
    *, input_ids: torch.Tensor, pixel_values: torch.Tensor
) -> dict[str, Any]:
    return {
        "input_ids": input_ids,
        "multi_modal_input": [
            {
                "pixel_values": pixel_values,
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        ],
    }


def test_group_tensor_store_shared_image_returns_same_reference():
    """Shared image tensors should use one ref while text remains untouched."""
    store = GroupTensorStore()
    pixel_values = torch.arange(12).reshape(1, 3, 2, 2)
    first_ids = torch.tensor([[1, 2]])
    second_ids = torch.tensor([[1, 3]])

    first = store.encode_multimodal_tensors(
        _trajectory(input_ids=first_ids, pixel_values=pixel_values)
    )
    second = store.encode_multimodal_tensors(
        _trajectory(input_ids=second_ids, pixel_values=pixel_values)
    )

    first_ref = first["multi_modal_input"][0]["pixel_values"]
    second_ref = second["multi_modal_input"][0]["pixel_values"]
    assert first_ref == second_ref
    assert first["input_ids"] is first_ids
    assert second["input_ids"] is second_ids
    fetched = store.fetch([first_ref["ref_id"]])
    assert fetched[first_ref["ref_id"]] is pixel_values


def test_group_tensor_store_equal_distinct_images_return_distinct_references():
    """Equal-valued image tensors must not be merged without shared identity."""
    store = GroupTensorStore()
    first_image = torch.arange(4)
    second_image = first_image.clone()

    first = store.encode_multimodal_tensors(
        {"multi_modal_input": [{"pixel_values": first_image}]}
    )
    second = store.encode_multimodal_tensors(
        {"multi_modal_input": [{"pixel_values": second_image}]}
    )

    assert (
        first["multi_modal_input"][0]["pixel_values"]["ref_id"]
        != second["multi_modal_input"][0]["pixel_values"]["ref_id"]
    )


@pytest.mark.asyncio
async def test_shared_tensor_resolver_concurrent_sessions_fetches_reference_once():
    """Concurrent session exports should share one fetch and one tensor object."""
    resolver = SharedTensorResolver()
    ref = {"type": "areal_shared_tensor", "ref_id": "tensor-0"}
    tensor = torch.arange(8)
    fetch_calls: list[list[str]] = []
    allow_fetch = asyncio.Event()

    async def fetch(ref_ids: list[str]) -> dict[str, torch.Tensor]:
        fetch_calls.append(ref_ids)
        await allow_fetch.wait()
        return {"tensor-0": tensor}

    def serialized_interaction(interaction_id: str) -> dict[str, Any]:
        return {
            interaction_id: {
                "tensor_dict": {
                    "input_ids": torch.tensor([[1, 2]]),
                    "multi_modal_input": [{"pixel_values": ref}],
                },
                "reward": 1.0,
                "interaction_id": interaction_id,
            }
        }

    first_task = asyncio.create_task(
        resolver.resolve(
            serialized_interaction("interaction-0"),
            group_id="train:task-1",
            fetch=fetch,
        )
    )
    await asyncio.sleep(0)
    second_task = asyncio.create_task(
        resolver.resolve(
            serialized_interaction("interaction-1"),
            group_id="train:task-1",
            fetch=fetch,
        )
    )
    await asyncio.sleep(0)
    allow_fetch.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert fetch_calls == [["tensor-0"]]
    first_interaction = deserialize_interactions(first)["interaction-0"]
    second_interaction = deserialize_interactions(second)["interaction-1"]
    first_image = first_interaction.to_tensor_dict()["multi_modal_input"][0][
        "pixel_values"
    ]
    second_image = second_interaction.to_tensor_dict()["multi_modal_input"][0][
        "pixel_values"
    ]
    assert first_image is tensor
    assert second_image is tensor
    assert first_image is second_image

    grouped = concat_padded_tensors(
        [first_interaction.to_tensor_dict(), second_interaction.to_tensor_dict()]
    )
    assert grouped["multi_modal_input"][0]["pixel_values"] is tensor
    assert grouped["multi_modal_input"][1]["pixel_values"] is tensor


@pytest.mark.asyncio
async def test_shared_tensor_resolver_ignores_reference_marker_in_text_payload():
    """Reference-like user text must not enter multimodal tensor resolution."""
    resolver = SharedTensorResolver()
    value = {
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "areal_shared_tensor",
                    "ref_id": "user-provided-value",
                },
            }
        ]
    }

    async def unexpected_fetch(_ref_ids: list[str]) -> dict[str, torch.Tensor]:
        raise AssertionError("text payload must not fetch shared tensors")

    resolved = await resolver.resolve(
        value,
        group_id="train:task-1",
        fetch=unexpected_fetch,
    )

    assert resolved is value


def test_group_tensor_store_registry_discard_removes_references():
    """Finalizing a group should release all proxy-side tensor references."""
    registry = GroupTensorStoreRegistry()
    store = registry.get_or_create("train:task-1")
    encoded = store.encode_multimodal_tensors(
        {"multi_modal_input": [{"pixel_values": torch.arange(4)}]}
    )
    ref_id = encoded["multi_modal_input"][0]["pixel_values"]["ref_id"]

    registry.discard("train:task-1")

    with pytest.raises(KeyError, match="Unknown shared tensor group"):
        registry.fetch("train:task-1", [ref_id])


@pytest.mark.asyncio
async def test_staggered_exports_after_cleanup_resolve_new_store_tensors(monkeypatch):
    """A recreated store must not collide with an earlier export's cached refs."""
    now = 0.0
    monkeypatch.setattr(
        "areal.experimental.openai.proxy.tensor_reference.time.monotonic", lambda: now
    )
    registry = GroupTensorStoreRegistry()
    resolver = SharedTensorResolver()
    group_id = "train:staggered"
    fetch_calls = []

    async def fetch(ref_ids: list[str]) -> dict[str, torch.Tensor]:
        fetch_calls.append(ref_ids)
        return registry.fetch(group_id, ref_ids)

    first = registry.get_or_create(group_id).encode_multimodal_tensors(
        {"multi_modal_input": [{"pixel_values": torch.tensor([1])}]}
    )
    resolved_first = await resolver.resolve(first, group_id=group_id, fetch=fetch)

    now = 61.0
    registry.discard_stale(timeout_seconds=60)
    with pytest.raises(KeyError, match="Unknown shared tensor group"):
        registry.fetch(group_id, [])

    later = registry.get_or_create(group_id).encode_multimodal_tensors(
        {"multi_modal_input": [{"pixel_values": torch.tensor([99])}]}
    )
    first_ref = first["multi_modal_input"][0]["pixel_values"]["ref_id"]
    later_ref = later["multi_modal_input"][0]["pixel_values"]["ref_id"]
    assert later_ref != first_ref
    with pytest.raises(KeyError, match="Unknown shared tensor references"):
        registry.fetch(group_id, [first_ref])

    resolved_later = await resolver.resolve(later, group_id=group_id, fetch=fetch)
    resolved_again = await resolver.resolve(later, group_id=group_id, fetch=fetch)

    assert fetch_calls == [[first_ref], [later_ref]]
    torch.testing.assert_close(
        resolved_first["multi_modal_input"][0]["pixel_values"],
        torch.tensor([1]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        resolved_later["multi_modal_input"][0]["pixel_values"],
        torch.tensor([99]),
        rtol=0,
        atol=0,
    )
    assert (
        resolved_again["multi_modal_input"][0]["pixel_values"]
        is resolved_later["multi_modal_input"][0]["pixel_values"]
    )
