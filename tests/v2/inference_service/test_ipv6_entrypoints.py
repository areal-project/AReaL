from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock, patch


def test_data_proxy_main_formats_ipv6_serving_addr():
    from areal.v2.inference_service.data_proxy import (
        __main__ as data_proxy_main,
    )

    with (
        patch.object(
            sys,
            "argv",
            [
                "data-proxy",
                "--host",
                "::1",
                "--port",
                "8082",
                "--tokenizer-path",
                "mock-tokenizer",
                "--admin-api-key",
                "admin-key",
                "--callback-server-addr",
                "http://[::1]:19000",
            ],
        ),
        patch.object(data_proxy_main, "create_app") as mock_create_app,
        patch.object(data_proxy_main.uvicorn, "run") as mock_run,
    ):
        mock_create_app.return_value = MagicMock()

        data_proxy_main.main()

    config = mock_create_app.call_args.args[0]
    assert config.serving_addr == "[::1]:8082"
    assert config.deterministic_sampling is False
    assert config.message_preprocessors == ()
    assert config.prefix_matcher is None
    mock_run.assert_called_once()


def test_guard_main_registers_ipv6_worker_addr():
    from areal.v2.inference_service.guard import __main__ as guard_main

    args = argparse.Namespace(
        port=0,
        host="::1",
        experiment_name="test-exp",
        trial_name="test-trial",
        role="guard",
        worker_index=0,
        name_resolve_type="nfs",
        nfs_record_root="/tmp/areal/name_resolve",
        etcd3_addr="localhost:2379",
        fileroot=None,
    )

    with (
        patch.object(
            guard_main,
            "make_base_parser",
            return_value=MagicMock(
                parse_known_args=MagicMock(return_value=(args, [])),
            ),
        ),
        patch.object(
            guard_main,
            "configure_state_from_args",
            return_value="::1",
        ) as mock_configure,
        patch.object(guard_main, "run_server") as mock_run_server,
    ):
        guard_main.main()

    mock_configure.assert_called_once()
    called_state = mock_configure.call_args.args[0]
    assert called_state is guard_main._state

    mock_run_server.assert_called_once()
    rs_args = mock_run_server.call_args
    assert rs_args.args[2] == "::1"
    assert rs_args.args[3] == 0
