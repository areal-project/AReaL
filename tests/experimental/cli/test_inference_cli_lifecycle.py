# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json


def test_terminate_runtime_state_kills_in_order(monkeypatch):
    from areal.experimental.cli.inference import common
    from areal.experimental.cli.inference.state import (
        ModelEntry,
        ModelState,
        RuntimeState,
        ServiceState,
    )

    service_state = ServiceState(
        service="svc",
        gateway_pid=30,
        gateway_url="http://127.0.0.1:8080",
        router_pid=40,
        router_url="http://127.0.0.1:9000",
        admin_api_key="admin",
        started_at=1.0,
    )
    model_state = ModelState(
        service="svc",
        models={
            "m": ModelEntry(
                kind="internal",
                engine_pids=[10],
                proxy_pids=[20],
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

    assert calls == [([10], 7.0), ([20], 7.0), ([30], 7.0), ([40], 7.0)]


def test_prepare_service_slot_force_recovers_raw_pids(tmp_path, monkeypatch):
    from areal.experimental.cli.inference.commands import run
    from areal.experimental.cli.inference.state import (
        models_state_path,
        service_state_path,
    )

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    service_state_path("svc").write_text(
        json.dumps(
            {
                "service": "svc",
                "gateway_pid": 100,
                "gateway_url": "http://127.0.0.1:8080",
                "router_pid": 101,
                "router_url": "http://127.0.0.1:9000",
                "admin_api_key": "admin",
                "started_at": 1.0,
                "future_key": "forces ServiceState.load to fail",
            }
        )
    )
    models_state_path("svc").write_text(
        json.dumps(
            {
                "models": {
                    "m": {
                        "engine_pids": [200],
                        "proxy_pids": [300],
                        "pids": [200, 300],
                    }
                }
            }
        )
    )
    calls = []

    def fake_kill_pids(pids, grace_s):
        calls.append((list(pids), grace_s))

    monkeypatch.setattr(run, "kill_pids", fake_kill_pids)

    run._prepare_service_slot(service="svc", force=True)

    assert calls == [([100, 101, 200, 300], 5.0)]
    assert not service_state_path("svc").exists()
    assert not models_state_path("svc").exists()


def test_foreground_cleanup_uses_latest_model_state(tmp_path, monkeypatch):
    from areal.experimental.cli.inference import common
    from areal.experimental.cli.inference.commands import run
    from areal.experimental.cli.inference.state import (
        ModelEntry,
        ModelState,
        ServiceState,
    )

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    service_state = ServiceState(
        service="svc",
        gateway_pid=30,
        gateway_url="http://127.0.0.1:8080",
        router_pid=40,
        router_url="http://127.0.0.1:9000",
        admin_api_key="admin",
        started_at=1.0,
    )
    latest = ModelState(service="svc")
    latest.models["later"] = ModelEntry(
        kind="internal",
        engine_pids=[10],
        proxy_pids=[20],
    )
    latest.save()
    calls = []

    def fake_kill_pids(pids, grace_s):
        calls.append((list(pids), grace_s))

    monkeypatch.setattr(common, "kill_pids", fake_kill_pids)

    run._cleanup_runtime("svc", service_state, ModelState(service="svc"), grace_s=3.0)

    assert calls == [([10], 3.0), ([20], 3.0), ([30], 3.0), ([40], 3.0)]
