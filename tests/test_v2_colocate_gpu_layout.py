import inspect

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


def test_colocate_execute_dispatch_does_not_retry():
    from areal.v2.weight_update.gateway import app as gw_app

    # The general poster keeps its retry wrapper; the execute dispatcher must
    # use a single-shot poster because reader/writer init is not reentrant.
    assert hasattr(gw_app._post, "retry")
    assert not hasattr(gw_app._post_once, "retry")

    src = inspect.getsource(gw_app)
    seg = src.split("async def _colocate_execute_weight_update", 1)[1]
    seg = seg.split("async def ", 1)[0]
    assert "_post_once(" in seg
    assert "_post(" not in seg.replace("_post_once(", "")
