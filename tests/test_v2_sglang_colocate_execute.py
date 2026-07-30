from types import SimpleNamespace
from unittest import mock

from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter


def test_execute_colocate_update_resumes_weights_before_reader():
    events = []

    adapter = AwexSGLangAdapter.__new__(AwexSGLangAdapter)
    adapter.wait_for_training_offloaded = lambda version: events.append("wait")
    adapter.resume_memory = lambda tags: events.append(("resume", tuple(tags)))
    adapter._rebuild_derived_weights = lambda: events.append("post_load")

    reader = SimpleNamespace(
        update_weights=lambda step_id: events.append(("transfer", step_id))
    )

    def ensure_reader():
        events.append("build_reader")
        return reader

    adapter._ensure_reader = ensure_reader

    with mock.patch("areal.v2.weight_update.awex.sglang_adapter.torch") as torch_mock:
        torch_mock.cuda.synchronize.side_effect = lambda: events.append("quiesce")
        AwexSGLangAdapter.execute_colocate_weight_update(adapter, version=3)

    assert events == [
        "wait",
        ("resume", ("weights",)),
        "quiesce",
        "build_reader",
        ("transfer", 3),
        "post_load",
    ]
