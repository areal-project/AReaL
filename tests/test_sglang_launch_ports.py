# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from areal.engine import sglang_remote


@pytest.mark.parametrize(
    "communication_args",
    [
        {},
        {"nccl_port": None},
        {"nccl_port": 19001},
        {"dist_init_addr": "localhost:19002"},
    ],
)
def test_launch_server_allocates_only_unspecified_communication_port(
    monkeypatch, communication_args
):
    build_cmd = MagicMock(return_value=["sglang-server"])
    monkeypatch.setattr(sglang_remote.SGLangConfig, "build_cmd_from_args", build_cmd)
    monkeypatch.setattr(sglang_remote.subprocess, "Popen", MagicMock())
    choose_port = MagicMock(return_value=[19003])
    monkeypatch.setattr(sglang_remote, "find_free_ports", choose_port)
    args = {"port": 19000, **communication_args}

    sglang_remote.SGLangBackend().launch_server(args)

    launched_args = build_cmd.call_args.args[0]
    if communication_args.get("nccl_port") or communication_args.get("dist_init_addr"):
        choose_port.assert_not_called()
        for key, value in communication_args.items():
            assert launched_args[key] == value
    else:
        choose_port.assert_called_once_with(1, exclude_ports={19000})
        assert launched_args["nccl_port"] == 19003
    assert launched_args["port"] == 19000
