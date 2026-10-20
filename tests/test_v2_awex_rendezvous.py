from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from areal.v2.inference_service.controller.controller import RolloutControllerV2
from areal.v2.training_service.controller.controller import GatewayTrainController


@pytest.mark.parametrize("colocate", [False, True])
def test_connect_engine_awex_allocates_rendezvous_only_for_separation(colocate):
    actor = object.__new__(GatewayTrainController)
    actor._ensure_initialized = MagicMock()
    actor._worker_addrs = ["http://train:8000"]
    actor._role = "actor"
    actor.config = SimpleNamespace(
        admin_api_key="test", log_level="info", request_timeout=30
    )
    rollout = MagicMock(spec=RolloutControllerV2)
    rollout.inference_worker_urls = ["http://infer:8000"]
    rollout.inference_guard_addrs = ["http://infer:9000"]
    response = MagicMock()
    response.json.return_value = {"host": "infer", "ports": [12345]}

    with (
        patch("requests.post", return_value=response) as post,
        patch(
            "areal.v2.weight_update.controller.controller.WeightUpdateController"
        ) as controller,
    ):
        actor.connect_engine(rollout, SimpleNamespace(type="awex", colocate=colocate))

    kwargs = controller.return_value.connect.call_args.kwargs
    assert kwargs["colocate"] is colocate
    if colocate:
        post.assert_not_called()
        assert kwargs["nccl_master_addr"] == ""
        assert kwargs["nccl_master_port"] == 0
    else:
        post.assert_called_once_with(
            "http://infer:9000/alloc_ports", json={"count": 1}, timeout=30
        )
        response.raise_for_status.assert_called_once()
        assert kwargs["nccl_master_addr"] == "infer"
        assert kwargs["nccl_master_port"] == 12345
