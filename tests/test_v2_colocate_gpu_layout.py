import asyncio
from unittest.mock import ANY, AsyncMock, call, patch

import pytest


def test_physical_base_from_cvd_maps_guard_mask_to_physical_base():
    from areal.v2.inference_service.sglang.launch_server import physical_base_from_cvd

    assert physical_base_from_cvd("4,5,6,7") == 4
    assert physical_base_from_cvd("0,1,2,3") == 0
    assert physical_base_from_cvd("0,1,2,3,4,5,6,7") == 0


def test_physical_base_from_cvd_rejects_empty_and_noncontiguous():
    from areal.v2.inference_service.sglang.launch_server import physical_base_from_cvd

    with pytest.raises(ValueError):
        physical_base_from_cvd("")
    with pytest.raises(ValueError):
        physical_base_from_cvd("0,2,3,4")


def test_separation_liveness_collective_does_not_retry_per_worker():
    from areal.v2.weight_update.gateway import app as gw_app

    post_once = AsyncMock()
    with patch.object(gw_app, "_post_once", post_once):
        asyncio.run(
            gw_app._initialize_awex_liveness(
                object(), ["http://infer", "http://train"], 30.0, 4
            )
        )

    assert post_once.await_args_list == [
        call(
            ANY,
            "http://infer/awex/batch_isend_irecv",
            30.0,
            json_data={"world_size": 4},
        ),
        call(
            ANY,
            "http://train/awex/batch_isend_irecv",
            30.0,
            json_data={"world_size": 4},
        ),
    ]
