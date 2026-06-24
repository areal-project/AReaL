# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from areal.experimental.cli.inference.scheduler import TaskHandle
from areal.experimental.cli.inference.state import (
    ModelEntry,
    ModelReplica,
    ModelState,
    ServiceState,
)


def _service_state(gateway_pid: int = 30, router_pid: int = 40) -> ServiceState:
    return ServiceState(
        service="svc",
        backend="local",
        gateway_handle=TaskHandle(
            host="127.0.0.1", ports=[8080], gpu_devices=[], ref={"pid": gateway_pid}
        ),
        router_handle=TaskHandle(
            host="127.0.0.1", ports=[9000], gpu_devices=[], ref={"pid": router_pid}
        ),
        admin_api_key="admin",
        started_at=1.0,
    )


def _replica(*, worker_pid: int, proxy_pid: int) -> ModelReplica:
    return ModelReplica(
        data_proxy=TaskHandle(
            host="127.0.0.1", ports=[5001], gpu_devices=[], ref={"pid": proxy_pid}
        ),
        worker=TaskHandle(
            host="127.0.0.1", ports=[5000], gpu_devices=[0], ref={"pid": worker_pid}
        ),
    )


def test_terminate_runtime_state_kills_in_order(monkeypatch):
    from areal.experimental.cli.inference import common
    from areal.experimental.cli.inference.state import RuntimeState

    service_state = _service_state(gateway_pid=30, router_pid=40)
    model_state = ModelState(
        service="svc",
        models={
            "m": ModelEntry(
                backend="sglang:tp=1,dp=1",
                replicas=[_replica(worker_pid=10, proxy_pid=20)],
            )
        },
    )
    calls = []

    def fake_kill_pids(pids, grace_s):
        calls.append((list(pids), grace_s))

    monkeypatch.setattr(common, "kill_pids", fake_kill_pids)

    common.terminate_runtime_state(
        RuntimeState(service_state=service_state, model_state=model_state),
        grace_s=7.0,
    )

    # Data-flow order: data_proxy → worker → gateway → router
    assert calls == [([20], 7.0), ([10], 7.0), ([30], 7.0), ([40], 7.0)]


def test_prepare_service_slot_force_recovers_raw_pids(tmp_path, monkeypatch):
    from areal.experimental.cli.inference.commands import run
    from areal.experimental.cli.inference.state import (
        models_state_path,
        service_state_path,
    )

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    # Omit the ``backend`` field so ServiceState.load raises KeyError and
    # _prepare_service_slot falls back to recover_pids_from_raw_state.
    service_state_path("svc").write_text(
        json.dumps(
            {
                "service": "svc",
                "gateway_handle": {
                    "host": "127.0.0.1",
                    "ports": [8080],
                    "gpu_devices": [],
                    "ref": {"pid": 100},
                },
                "router_handle": {
                    "host": "127.0.0.1",
                    "ports": [9000],
                    "gpu_devices": [],
                    "ref": {"pid": 101},
                },
                "admin_api_key": "admin",
                "started_at": 1.0,
            }
        )
    )
    models_state_path("svc").write_text(
        json.dumps(
            {
                "service": "svc",
                "models": {
                    "m": {
                        "backend": "sglang:tp=1,dp=1",
                        "replicas": [
                            {
                                "data_proxy": {
                                    "host": "127.0.0.1",
                                    "ports": [5001],
                                    "gpu_devices": [],
                                    "ref": {"pid": 300},
                                },
                                "worker": {
                                    "host": "127.0.0.1",
                                    "ports": [5000],
                                    "gpu_devices": [0],
                                    "ref": {"pid": 200},
                                },
                            }
                        ],
                    }
                },
            }
        )
    )
    calls = []

    def fake_kill_pids(pids, grace_s):
        calls.append((list(pids), grace_s))

    monkeypatch.setattr(run, "kill_pids", fake_kill_pids)

    run._prepare_service_slot(service="svc", force=True)

    assert calls == [([100, 101, 300, 200], 5.0)]
    assert not service_state_path("svc").exists()
    assert not models_state_path("svc").exists()


def test_foreground_cleanup_uses_latest_model_state(tmp_path, monkeypatch):
    from areal.experimental.cli.inference import common
    from areal.experimental.cli.inference.commands import run

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    service_state = _service_state(gateway_pid=30, router_pid=40)
    latest = ModelState(service="svc")
    latest.models["later"] = ModelEntry(
        backend="sglang:tp=1,dp=1",
        replicas=[_replica(worker_pid=10, proxy_pid=20)],
    )
    latest.save()
    calls = []

    def fake_kill_pids(pids, grace_s):
        calls.append((list(pids), grace_s))

    monkeypatch.setattr(common, "kill_pids", fake_kill_pids)

    run._cleanup_runtime("svc", service_state, ModelState(service="svc"), grace_s=3.0)

    # Same data-flow order as terminate_runtime_state.
    assert calls == [([20], 3.0), ([10], 3.0), ([30], 3.0), ([40], 3.0)]
