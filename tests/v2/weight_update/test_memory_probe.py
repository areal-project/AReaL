# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

from areal.v2.inference_service.sglang import scheduler as scheduler_module
from areal.v2.weight_update import memory_probe


def test_physical_device_id_respects_visible_device_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5, 7")

    assert memory_probe._physical_device_id(0) == "5"
    assert memory_probe._physical_device_id(1) == "7"
    assert memory_probe._physical_device_id(2) is None


def test_memory_probe_is_useful_without_cuda(monkeypatch):
    monkeypatch.setattr(memory_probe.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(memory_probe.socket, "gethostname", lambda: "node-a")
    monkeypatch.setattr(memory_probe.os, "getpid", lambda: 42)

    probe = memory_probe.collect_awex_memory_probe(
        role="training",
        rank=3,
        pair_names=["pair-b", "pair-a", "pair-b"],
    )

    assert probe == {
        "role": "training",
        "rank": 3,
        "host": "node-a",
        "pid": 42,
        "pair_names": ["pair-a", "pair-b"],
        "cuda_available": False,
    }


def test_scheduler_memory_probe_collects_every_distributed_rank(monkeypatch):
    scheduler = SimpleNamespace(tp_rank=0, dp_rank=0, pp_rank=0)
    bridge = scheduler_module.AwexSchedulerBridge(scheduler)
    bridge._push_result = MagicMock()
    local_probe = {"rank": "dp=0 pp=0 tp=0"}
    monkeypatch.setattr(
        memory_probe,
        "collect_awex_memory_probe",
        lambda **kwargs: local_probe,
    )
    monkeypatch.setattr(scheduler_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(scheduler_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(scheduler_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(scheduler_module.dist, "get_rank", lambda: 0)

    def gather(probes, probe):
        probes[:] = [probe, {"rank": "dp=0 pp=1 tp=0"}]

    monkeypatch.setattr(scheduler_module.dist, "all_gather_object", gather)

    bridge.awex_report_memory_probe(["actor-rollout-v2"])

    bridge._push_result.assert_called_once_with(
        [
            local_probe,
            {"rank": "dp=0 pp=1 tp=0"},
        ]
    )
