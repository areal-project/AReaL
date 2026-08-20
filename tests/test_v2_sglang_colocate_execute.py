import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

from areal.engine.weight_update.awex.colocate_protocol import ColocateTopology
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
    adapter.wait_for_training_offloaded = lambda: events.append("wait")
    adapter.resume_memory = lambda tags: events.append(("resume", tuple(tags)))
    topology = ColocateTopology(
        transfer_rank=2,
        infer_world_size=4,
        train_world_size=4,
        instance_world_size=1,
    )
    client = SimpleNamespace(
        add_object_to_set=lambda key, value: events.append(("finished", key, value))
    )
    adapter._colocate_backend = SimpleNamespace(
        topology=topology,
        meta_server_client=client,
        update_weights=lambda version: events.extend(
            [("transfer", version), "post_load"]
        ),
    )

    with mock.patch("areal.v2.weight_update.awex.sglang_adapter.torch") as torch_mock:
        torch_mock.cuda.synchronize.side_effect = lambda: events.append("quiesce")
        AwexSGLangAdapter.execute_colocate_weight_update(adapter, version=3)

    assert events == [
        "wait",
        ("resume", ("weights",)),
        "quiesce",
        ("transfer", 3),
        "post_load",
        ("finished", "finished_weights_update_engines", 2),
    ]


class TestResumeMemoryIgnoresLocallyTrackedTags:
    """The colocate handover releases inference memory outside this adapter."""

    @staticmethod
    def _adapter():
        adapter = AwexSGLangAdapter.__new__(AwexSGLangAdapter)
        adapter._scheduler = mock.Mock()
        return adapter

    @staticmethod
    def _resume(adapter, tags):
        io_struct = ModuleType("sglang.srt.managers.io_struct")
        io_struct.ResumeMemoryOccupationReqInput = SimpleNamespace
        with mock.patch.dict(sys.modules, {"sglang.srt.managers.io_struct": io_struct}):
            AwexSGLangAdapter.resume_memory(adapter, tags)

    def test_resume_issues_request_when_release_was_not_observed(self):
        adapter = self._adapter()

        self._resume(adapter, ["weights"])

        adapter._scheduler.resume_memory_occupation.assert_called_once()
        req = adapter._scheduler.resume_memory_occupation.call_args.args[0]
        assert req.tags == ["weights"]

    def test_resume_skips_only_tags_sglang_cannot_serve(self):
        adapter = self._adapter()

        self._resume(adapter, ["optimizer"])

        adapter._scheduler.resume_memory_occupation.assert_not_called()

    def test_resume_passes_through_every_supported_tag(self):
        adapter = self._adapter()

        self._resume(adapter, ["weights", "kv_cache", "cuda_graph"])

        req = adapter._scheduler.resume_memory_occupation.call_args.args[0]
        assert set(req.tags) == AwexSGLangAdapter._SGLANG_MEMORY_TAGS
