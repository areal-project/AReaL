from types import SimpleNamespace

from areal.engine.awex.adapters.sglang_adapter import AwexSGLangAdapter
from areal.engine.awex.plugins.sglang_plugin import AwexSchedulerPlugin
from areal.v2.inference_service.sglang.scheduler import AwexSchedulerBridge


def test_v1_and_v2_use_the_shared_sglang_adapter():
    bridge = AwexSchedulerBridge.__new__(AwexSchedulerBridge)
    bridge._adapter = None
    bridge._scheduler = SimpleNamespace()

    adapter = bridge._require_adapter()

    plugin = AwexSchedulerPlugin.__new__(AwexSchedulerPlugin)
    plugin._receiver = None
    plugin._scheduler = bridge._scheduler
    assert type(plugin._require_receiver()) is AwexSGLangAdapter
    assert type(adapter) is AwexSGLangAdapter
