# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType

from areal.utils.mbridge_compat import _patch_qwen3vl_position_embedding_type


def test_qwen3vl_position_embedding_type_guard_preserves_requested_value(monkeypatch):
    for package_name in ("mbridge", "mbridge.models", "mbridge.models.qwen3_vl"):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    class Qwen3VLGPTModel:
        def __init__(self, *args, position_embedding_type="rope", **kwargs):
            del args, kwargs, position_embedding_type
            self.position_embedding_type = "rope"

    module_name = "mbridge.models.qwen3_vl.gpt_model"
    module = ModuleType(module_name)
    module.Qwen3VLGPTModel = Qwen3VLGPTModel
    monkeypatch.setitem(sys.modules, module_name, module)

    _patch_qwen3vl_position_embedding_type()
    model = Qwen3VLGPTModel(position_embedding_type="mrope")

    assert model.position_embedding_type == "mrope"
    wrapped = Qwen3VLGPTModel.__init__
    _patch_qwen3vl_position_embedding_type()
    assert Qwen3VLGPTModel.__init__ is wrapped
