from types import SimpleNamespace

import torch

from areal.api import ParallelStrategy
from areal.trainer.ppo.actor import PPOActorController


def _make_controller(backend: str) -> PPOActorController:
    controller = object.__new__(PPOActorController)
    controller.train_alloc = SimpleNamespace(backend=backend)
    controller.config = SimpleNamespace(packing_algorithm="ffd")
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


def test_megatron_advantages_reattach_images_after_balanced_dispatch(monkeypatch):
    """Vision payloads follow original trajectory order after DP result merging."""
    controller = _make_controller("megatron")
    controller.train_alloc.parallel = ParallelStrategy(data_parallel_size=2)
    controller.workers_is_dp_head = [True, False, True, False]
    batch = [
        {
            "input_ids": torch.full((1, length), idx),
            "attention_mask": torch.ones(1, length, dtype=torch.bool),
            "multi_modal_input": [{"pixel_values": torch.tensor([idx])}],
        }
        for idx, length in enumerate([2, 9, 4, 7])
    ]

    def dispatch(method, rpc_batch, *, rpc_meta):
        args, kwargs, indices = controller._prepare_dispatch(rpc_batch)
        assert [idx for group in indices for idx in group] != list(range(4))
        assert not kwargs
        assert all(
            "multi_modal_input" not in item for group in args[0] for item in group
        )
        worker_results = []
        for group in args[0]:
            worker_results.extend(
                (
                    [
                        dict(item, advantages=item["input_ids"].float())
                        for item in group
                    ],
                    None,
                )
            )
        return controller._collect_results(worker_results, indices)

    monkeypatch.setattr(controller, "_custom_function_call", dispatch)
    result = controller.compute_advantages(batch)
    for idx, item in enumerate(result):
        assert item["multi_modal_input"] is batch[idx]["multi_modal_input"]
        torch.testing.assert_close(
            item["advantages"], batch[idx]["input_ids"].float(), rtol=0, atol=0
        )
