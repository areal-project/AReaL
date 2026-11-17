# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.npu, pytest.mark.slow]


@pytest.mark.parametrize("train_draft", [False, True])
def test_native_reload_multiple_buckets_preserves_npu_graph_output(train_draft):
    """Checkpoint-layout reload must update storage captured before the reload."""
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    import vllm_ascend.ops  # noqa: F401
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

    from areal.engine.vllm_ext.npu_weight_reload import NpuReloadModel, NpuWeightReload

    class LayoutMethod(QuantizeMethodBase):
        def create_weights(self, layer, *args, **kwargs):
            pass

        def apply(self, layer, *args, **kwargs):
            raise NotImplementedError

        def process_weights_after_loading(self, layer):
            for parameter in layer.parameters():
                parameter.data = parameter.data.t().contiguous()

    def weight_loader(param, loaded_weight):
        param.copy_(loaded_weight)

    class Model(nn.Module):
        def __init__(self, prefix):
            super().__init__()
            self.prefix = prefix
            self.layer = nn.Module()
            self.layer.weight = nn.Parameter(
                torch.ones(3, 8, device="npu"), requires_grad=False
            )
            self.layer.bias_weight = nn.Parameter(
                torch.ones(3, 8, device="npu"), requires_grad=False
            )
            self.layer.quant_method = LayoutMethod()
            for parameter in self.layer.parameters():
                parameter.weight_loader = weight_loader

        def load_weights(self, weights):
            loaded = set()
            for name, value in weights:
                prefix, name = name.split(".", 1)
                if prefix != self.prefix:
                    continue
                parameter = getattr(self.layer, name)
                parameter.weight_loader(parameter, value)
                loaded.add(f"layer.{name}")
            return loaded

        def forward(self, inputs):
            return inputs @ self.layer.weight + inputs @ self.layer.bias_weight

    torch.npu.set_device(0)
    with torch.no_grad(), torch.device("npu"):
        model, draft = Model("target"), Model("draft")
        for part in (model, draft):
            record_metadata_for_reloading(part)
            part.layer.quant_method.process_weights_after_loading(part.layer)
        inputs = torch.ones(16, 8, device="npu")
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                model(inputs)
                draft(inputs)
        torch.npu.current_stream().wait_stream(stream)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = model(inputs) + draft(inputs)
        torch.npu.synchronize()

        wrapper = ACLGraphWrapper.__new__(ACLGraphWrapper)
        wrapper.runnable = model
        reload_model = NpuReloadModel(wrapper, draft)
        assert model in reload_model.modules()
        for value in (2.0, 3.0):
            reload = NpuWeightReload(reload_model, SimpleNamespace(dtype=torch.float32))
            reload.start()
            reload.load([("target.weight", torch.full((3, 8), value, device="npu"))])
            reload.load(
                [("target.bias_weight", torch.full((3, 8), value + 1, device="npu"))]
            )
            if train_draft:
                reload.load(
                    [("draft.weight", torch.full((3, 8), value + 2, device="npu"))]
                )
                reload.load(
                    [("draft.bias_weight", torch.full((3, 8), value + 3, device="npu"))]
                )
            reload.finish()
            graph.replay()
            torch.npu.synchronize()
            draft_sum = 2 * value + 5 if train_draft else 2
            expected = torch.full(
                (16, 3), 8 * (2 * value + 1 + draft_sum), device="npu"
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=1e-5)
