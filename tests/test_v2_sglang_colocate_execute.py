from types import SimpleNamespace
from unittest import mock

from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter


def test_sglang_memory_tags_include_cuda_graph():
    assert AwexSGLangAdapter._SGLANG_MEMORY_TAGS == {
        "kv_cache",
        "weights",
        "cuda_graph",
    }


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


class TestResumeMemoryIgnoresLocallyTrackedTags:
    """The colocate handover releases inference memory outside this adapter."""

    @staticmethod
    def _adapter():
        adapter = AwexSGLangAdapter.__new__(AwexSGLangAdapter)
        adapter._released_tags = set()
        adapter._scheduler = mock.Mock()
        return adapter

    def test_resume_issues_request_when_release_was_not_observed(self):
        adapter = self._adapter()

        AwexSGLangAdapter.resume_memory(adapter, ["weights"])

        adapter._scheduler.resume_memory_occupation.assert_called_once()
        req = adapter._scheduler.resume_memory_occupation.call_args.args[0]
        assert req.tags == ["weights"]

    def test_resume_skips_only_tags_sglang_cannot_serve(self):
        adapter = self._adapter()

        AwexSGLangAdapter.resume_memory(adapter, ["optimizer"])

        adapter._scheduler.resume_memory_occupation.assert_not_called()

    def test_resume_passes_through_every_supported_tag(self):
        adapter = self._adapter()

        AwexSGLangAdapter.resume_memory(adapter, ["weights", "kv_cache", "cuda_graph"])

        req = adapter._scheduler.resume_memory_occupation.call_args.args[0]
        assert set(req.tags) == AwexSGLangAdapter._SGLANG_MEMORY_TAGS
