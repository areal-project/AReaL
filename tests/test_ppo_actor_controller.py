from types import SimpleNamespace

import pytest
import torch

from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.infra.rpc.serialization import deserialize_value
from areal.trainer.ppo.actor import PPOActorController, PPOActorControllerV2


def _make_controller(backend: str) -> PPOActorController:
    controller = object.__new__(PPOActorController)
    controller.train_alloc = SimpleNamespace(backend=backend)
    return controller


def test_megatron_compute_advantages_bypasses_multimodal_payload(monkeypatch):
    """Megatron should keep vision RTensor references on the controller."""
    controller = _make_controller("megatron")
    pixel_values = torch.arange(4)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    batch = [
        {
            "input_ids": torch.tensor([1, 2]),
            "multi_modal_input": [
                {
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                }
            ],
        },
        {
            "input_ids": torch.tensor([3, 4]),
            "multi_modal_input": [
                {
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                }
            ],
        },
    ]
    captured = {}

    def fake_call(method, rpc_batch, *, rpc_meta):
        captured["method"] = method
        captured["batch"] = rpc_batch
        captured["rpc_meta"] = rpc_meta
        return [dict(item, advantages=torch.tensor([1.0])) for item in rpc_batch]

    monkeypatch.setattr(controller, "_custom_function_call", fake_call)

    result = controller.compute_advantages(batch)

    assert captured["method"] == "compute_advantages"
    assert captured["rpc_meta"] == {"broadcast": True}
    assert all("multi_modal_input" not in item for item in captured["batch"])
    assert result[0]["multi_modal_input"] is batch[0]["multi_modal_input"]
    assert result[1]["multi_modal_input"] is batch[1]["multi_modal_input"]
    assert "multi_modal_input" in batch[0]


def test_non_megatron_compute_advantages_keeps_existing_payload(monkeypatch):
    """Other v1 backends should retain their existing controller behavior."""
    controller = _make_controller("fsdp")
    batch = [{"multi_modal_input": [{"pixel_values": torch.arange(4)}]}]
    expected = [{"status": "unchanged"}]
    captured = {}

    def fake_call(method, *args, rpc_meta, **kwargs):
        captured["method"] = method
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["rpc_meta"] = rpc_meta
        return expected

    monkeypatch.setattr(controller, "_custom_function_call", fake_call)

    result = controller.compute_advantages(batch)

    assert result is expected
    assert captured["method"] == "compute_advantages"
    assert captured["args"][0] is batch
    assert captured["rpc_meta"] == {"broadcast": True}


def test_v2_megatron_advantages_preserves_group_vision_references(monkeypatch):
    """Multiple groups must not receive copies of the complete DP image list."""
    controller = object.__new__(PPOActorControllerV2)
    controller.train_alloc = SimpleNamespace(backend="megatron")
    images = [
        RTensor(
            shard=TensorShardInfo(shard_id=f"image-{group}", node_addr="test.invalid"),
            data=torch.empty((2, 3), dtype=torch.float32, device="meta"),
        )
        for group in range(3)
    ]
    batch = [
        {
            "input_ids": torch.full((4, 2), group, dtype=torch.long),
            "multi_modal_input": [{"pixel_values": images[group]} for _ in range(4)],
        }
        for group in range(3)
    ]

    def fake_call(path, payload):
        assert path == "/ppo/actor/compute_advantages"
        received = deserialize_value(payload["args"])[0]
        assert all("multi_modal_input" not in group for group in received)
        return [dict(group, advantages=torch.ones(4, 2)) for group in received]

    monkeypatch.setattr(controller, "_gateway_post_result", fake_call)
    result = controller.compute_advantages(batch)

    for original, updated in zip(batch, result, strict=True):
        assert updated["multi_modal_input"] is original["multi_modal_input"]
        assert len(updated["multi_modal_input"]) == 4
        torch.testing.assert_close(
            updated["input_ids"], original["input_ids"], rtol=0, atol=0
        )

    sent = {}

    def fake_update(path, payload):
        assert path == "/ppo/actor/update"
        sent["batch"] = deserialize_value(payload["args"])[0]

    monkeypatch.setattr(controller, "_gateway_post", fake_update)
    controller.ppo_update(result)
    for image, group in zip(images, sent["batch"], strict=True):
        for item in group["multi_modal_input"]:
            assert item["pixel_values"].shard == image.shard
            assert item["pixel_values"].data.is_meta


@pytest.mark.parametrize("invalid", [None, [], [{}], [{}, {}, None]])
def test_v2_megatron_advantages_rejects_invalid_group_results(monkeypatch, invalid):
    """Never silently attach images to a shortened or malformed result batch."""
    controller = object.__new__(PPOActorControllerV2)
    controller.train_alloc = SimpleNamespace(backend="megatron")
    batch = [
        {"multi_modal_input": [{"pixel_values": torch.zeros(1)}]} for _ in range(3)
    ]
    monkeypatch.setattr(controller, "_gateway_post_result", lambda *_: invalid)

    with pytest.raises(RuntimeError, match="Megatron compute_advantages returned"):
        controller.compute_advantages(batch)
    assert all("multi_modal_input" in item for item in batch)


@pytest.mark.parametrize("backend,with_images", [("fsdp", True), ("megatron", False)])
def test_v2_advantages_unrelated_payloads_keep_existing_behavior(
    monkeypatch, backend, with_images
):
    """Text-only and non-Megatron calls retain their previous payloads/results."""
    controller = object.__new__(PPOActorControllerV2)
    controller.train_alloc = SimpleNamespace(backend=backend)
    batch = [{"input_ids": torch.ones(1, 2, dtype=torch.long)}]
    if with_images:
        batch[0]["multi_modal_input"] = [{"pixel_values": torch.zeros(1)}]
    expected = [{"unchanged": True}]

    def fake_call(_path, payload):
        received = deserialize_value(payload["args"])[0]
        assert ("multi_modal_input" in received[0]) is with_images
        return expected

    monkeypatch.setattr(controller, "_gateway_post_result", fake_call)
    assert controller.compute_advantages(batch) is expected
