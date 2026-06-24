# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from areal.experimental.cli.agent.commands import run
from areal.experimental.cli.agent.state import (
    PairState,
    ProcessState,
    ServiceState,
    service_state_path,
)


def test_run_persists_service_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))

    def fake_launch_agent_stack(**kwargs):
        service = kwargs["service"]
        return ServiceState(
            service=service,
            launch_mode="detached",
            agent=kwargs["agent"],
            admin_api_key=kwargs["admin_api_key"],
            gateway=ProcessState(
                component="gateway",
                pid=11,
                url="http://127.0.0.1:1",
                log_file="gateway.log",
            ),
            router=ProcessState(
                component="router",
                pid=12,
                url="http://127.0.0.1:2",
                log_file="router.log",
            ),
            pairs=[
                PairState(
                    index=0,
                    worker=ProcessState(
                        component="worker-0",
                        pid=13,
                        url="http://127.0.0.1:3",
                        log_file="worker.log",
                    ),
                    data_proxy=ProcessState(
                        component="proxy-0",
                        pid=14,
                        url="http://127.0.0.1:4",
                        log_file="proxy.log",
                    ),
                )
            ],
            inf_addr=kwargs["inf_addr"],
            inf_api_key=kwargs["inf_api_key"],
            inf_model=kwargs["inf_model"],
            session_timeout=kwargs["session_timeout"],
            health_poll_interval=kwargs["health_poll_interval"],
            drain_timeout=kwargs["drain_timeout"],
        )

    monkeypatch.setattr(run, "launch_agent_stack", fake_launch_agent_stack)

    rc = run.do_run(
        agent="pkg.Agent",
        service="svc",
        num_pairs=1,
        admin_api_key="agent-admin",
        setup_timeout=1.0,
        health_poll_interval=1.0,
        drain_timeout=1.0,
        session_timeout=60.0,
        log_level="info",
        force=False,
        inf_addr="http://inf",
        inf_api_key="inf-admin",
        inf_model="model",
    )

    assert rc == 0
    assert service_state_path("svc").exists()
