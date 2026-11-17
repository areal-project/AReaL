# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from areal.engine.vllm_ext.npu_weight_reload import (
    NpuReloadModel,
    NpuWeightReload,
    needs_native_moe_reload,
    weight_update_models,
)

pytestmark = pytest.mark.vllm


@pytest.mark.parametrize("moe", [False, True])
def test_native_reload_gate_matches_installed_ascend_method(moe):
    """Select native reload for the installed Ascend MoE method only."""
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

    model = nn.Sequential(nn.Linear(2, 2, device="cpu"))
    if moe:
        # Only the method type is needed; avoid distributed initialization.
        model[0].quant_method = AscendUnquantizedFusedMoEMethod.__new__(
            AscendUnquantizedFusedMoEMethod
        )

    assert needs_native_moe_reload(model) is moe


@pytest.mark.parametrize("same_parameter", [False, True])
def test_native_tied_target_parameters_reload_through_alias(same_parameter):
    pytest.importorskip("vllm.model_executor.model_loader.reload")
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(2, 2, bias=False)
            self.b = nn.Linear(2, 2, bias=False)
            self.b.weight = (
                self.a.weight
                if same_parameter
                else nn.Parameter(self.a.weight.detach())
            )

        def load_weights(self, weights):
            for name, value in weights:
                layer = getattr(self, name)
                layer.weight.weight_loader(layer.weight, value)

    with torch.no_grad(), torch.device("cpu"):
        model = Model()
        record_metadata_for_reloading(model)
        pointer = model.a.weight.data_ptr()
        for value in (2.0, 3.0):
            reload = NpuWeightReload(model, SimpleNamespace(dtype=torch.float32))
            reload.start()
            reload.load([("b", torch.full((2, 2), value))])
            reload.finish()
            for layer in (model.a, model.b):
                assert layer.weight.data_ptr() == pointer
                torch.testing.assert_close(
                    layer.weight, torch.full((2, 2), value), rtol=0, atol=0
                )


@pytest.mark.parametrize("load_draft", [False, True])
def test_native_reload_preserves_frozen_draft_and_runtime_buffers(load_draft):
    pytest.importorskip("vllm.model_executor.model_loader.reload")
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

    class Model(nn.Module):
        def __init__(self, prefix):
            super().__init__()
            self.prefix = prefix
            self.layer = nn.Linear(2, 2, bias=False)

        def load_weights(self, weights):
            loaded = set()
            for name, value in weights:
                if name != f"{self.prefix}.weight":
                    continue
                self.layer.weight.weight_loader(self.layer.weight, value)
                loaded.add("layer.weight")
            return loaded

    with torch.no_grad(), torch.device("cpu"):
        target, draft = Model("target"), Model("draft")
        target.layer.weight.fill_(1)
        draft.layer.weight.fill_(2)
        # Runtime-only buffers need no actor checkpoint entry.
        target.runtime = nn.Module()
        target.runtime.register_buffer("state", torch.ones(2))
        model = NpuReloadModel(target, draft)
        record_metadata_for_reloading(model)
        pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
        reload = NpuWeightReload(model, SimpleNamespace(dtype=torch.float32))
        reload.start()
        for enabled, prefix in ((True, "target"), (load_draft, "draft")):
            if enabled:
                reload.load([(f"{prefix}.weight", torch.full((2, 2), 3.0))])
        reload.finish()
        torch.testing.assert_close(
            target.layer.weight, torch.full((2, 2), 3.0), rtol=0, atol=0
        )
        torch.testing.assert_close(
            draft.layer.weight,
            torch.full((2, 2), 3.0 if load_draft else 2.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(target.runtime.state, torch.ones(2), rtol=0, atol=0)
        assert {name: p.data_ptr() for name, p in model.named_parameters()} == pointers


@pytest.mark.parametrize("draft_only_alias", [False, True])
def test_native_generic_shared_padded_head_reloads_once(draft_only_alias):
    """Model-owned mappings handle shared layers without Qwen names or classes."""
    pytest.importorskip("vllm.model_executor.model_loader.reload")
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

    class Model(nn.Module):
        def __init__(self, path, layer, aliases):
            super().__init__()
            self.add_module(path, layer)
            self.path = path
            self.aliases = aliases

        def load_weights(self, weights):
            loaded = set()
            for name, value in weights:
                prefix, parameter_name = name.rsplit(".", 1)
                if prefix not in self.aliases:
                    continue
                name = f"{self.path}.{parameter_name}"
                parameter = self.get_parameter(name)
                parameter.weight_loader(parameter, value)
                loaded.add(name)
            return loaded

    with torch.no_grad(), torch.device("cpu"):
        layer = ParallelLMHead(
            100, 4, bias=True, params_dtype=torch.float32, disable_tp=True
        )
        target = Model("output", layer, {"checkpoint.output"})
        draft = Model("input", layer, {"checkpoint.output", "next_token.shared"})
        model = NpuReloadModel(target, draft)
        record_metadata_for_reloading(model)
        pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
        prefix = "next_token.shared" if draft_only_alias else "checkpoint.output"
        for value in (2.0, 3.0):
            transaction = NpuWeightReload(model, SimpleNamespace(dtype=torch.float32))
            transaction.start()
            transaction.load([(f"{prefix}.weight", torch.full((100, 4), value))])
            transaction.load([(f"{prefix}.bias", torch.full((100,), value + 1))])
            transaction.finish()
            for name, parameter in model.named_parameters():
                assert parameter.data_ptr() == pointers[name]
                expected = torch.zeros_like(parameter)
                expected[:100] = value + 1 if name.endswith("bias") else value
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "head,tp_size,tp_rank,tied", [(False, 1, 0, False), (True, 2, 1, True)]
)
def test_native_reload_padded_vocab_preserves_values_and_storage(
    monkeypatch, head, tp_size, tp_rank, tied
):
    pytest.importorskip("vllm.model_executor.model_loader.reload")
    from vllm.model_executor.layers import vocab_parallel_embedding as vocab
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

    monkeypatch.setattr(vocab, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_world_size", lambda: tp_size)
    layer_class = vocab.ParallelLMHead if head else vocab.VocabParallelEmbedding

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = layer_class(100, 4, params_dtype=torch.float32)
            if tied:
                self.alias = nn.Module()
                self.alias.weight = self.embed_tokens.weight

        def load_weights(self, weights):
            for name, value in weights:
                parameter = self.get_parameter(name)
                parameter.weight_loader(parameter, value)

    with torch.no_grad(), torch.device("cpu"):
        model = Model()
        record_metadata_for_reloading(model)
        model.load_weights([("embed_tokens.weight", torch.ones(100, 4))])
        pointer = model.embed_tokens.weight.data_ptr()
        for offset in (1, 2):
            values = torch.arange(400.0).reshape(100, 4) + offset
            transaction = NpuWeightReload(model, SimpleNamespace(dtype=torch.float32))
            transaction.start()
            transaction.load([("embed_tokens.weight", values)])
            transaction.finish()
            indices = model.embed_tokens.shard_indices
            expected = torch.zeros_like(model.embed_tokens.weight)
            expected[: indices.num_org_elements] = values[
                indices.org_vocab_start_index : indices.org_vocab_end_index
            ]
            assert model.embed_tokens.weight.data_ptr() == pointer
            torch.testing.assert_close(
                model.embed_tokens.weight, expected, rtol=0, atol=0
            )
            if tied:
                assert model.alias.weight.data_ptr() == pointer
                torch.testing.assert_close(model.alias.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("moe", [False, True])
def test_native_mtp_lookup_unwraps_breakable_graph_and_reloads(moe):
    pytest.importorskip("vllm.model_executor.model_loader.reload")
    pytest.importorskip("vllm_ascend")
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP, Qwen3_5MTP
    from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper

    with torch.no_grad(), torch.device("cpu"):
        draft_class = Qwen3_5MoeMTP if moe else Qwen3_5MTP
        draft = draft_class.__new__(draft_class)
        nn.Module.__init__(draft)
        draft.model = nn.Module()
        draft.model.fc = nn.Linear(4, 2, bias=False)
        draft.model.embed_tokens = nn.Embedding(8, 2)
        draft.lm_head = nn.Linear(2, 8, bias=False)
        wrapper = BreakableACLGraphWrapper.__new__(BreakableACLGraphWrapper)
        wrapper.runnable = draft
        config = SimpleNamespace(dtype=torch.float32)
        target = nn.Module()
        target.load_weights = lambda weights: set()
        runner = SimpleNamespace(
            model=target,
            model_config=config,
            drafter=SimpleNamespace(model=wrapper),
            vllm_config=SimpleNamespace(
                speculative_config=SimpleNamespace(
                    method="mtp", draft_model_config=config
                )
            ),
        )
        models = weight_update_models(runner)
        assert models[1] == (draft, config)
        model = NpuReloadModel(*(module for module, _ in models))
        record_metadata_for_reloading(model)
        pointer = draft.model.fc.weight.data_ptr()
        for value in (2.0, 3.0):
            transaction = NpuWeightReload(model, config)
            transaction.start()
            transaction.load([("mtp.fc.weight", torch.full((2, 4), value))])
            transaction.finish()
            assert draft.model.fc.weight.data_ptr() == pointer
            torch.testing.assert_close(
                draft.model.fc.weight, torch.full((2, 4), value), rtol=0, atol=0
            )


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("train_mtp", [False, True])
@pytest.mark.parametrize("mtp_first", [False, True])
def test_native_reload_target_and_mtp_preserves_storage(shared, train_mtp, mtp_first):
    """Exercise real native loaders with shared modules and two successive updates."""
    pytest.importorskip("vllm.model_executor.model_loader.reload")
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP
    from vllm.model_executor.utils import replace_parameter

    class LayoutMethod(QuantizeMethodBase):
        def create_weights(self, layer, *args, **kwargs):
            pass

        def apply(self, layer, *args, **kwargs):
            raise NotImplementedError

        def process_weights_after_loading(self, layer):
            for name, parameter in list(layer.named_parameters()):
                replace_parameter(layer, name, parameter.data.t().contiguous())
                getattr(layer, name).weight_loader = weight_loader

    class Target(nn.Module):
        def load_weights(self, weights):
            loaded = set()
            for name, tensor in weights:
                if name.startswith("mtp."):
                    continue
                parameter = self.get_parameter(name)
                parameter.weight_loader(parameter, tensor)
                loaded.add(name)
            return loaded

    def weight_loader(param, loaded_weight):
        param.copy_(loaded_weight)

    def layout_layer():
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.ones(3, 8), requires_grad=False)
        layer.bias_weight = nn.Parameter(torch.ones(3, 8), requires_grad=False)
        layer.quant_method = LayoutMethod()
        return layer

    with torch.no_grad(), torch.device("cpu"):
        target = Target()
        target.model = nn.Module()
        target.model.embed_tokens = nn.Embedding(4, 2)
        target.lm_head = nn.Linear(2, 4, bias=False)
        target.layer = layout_layer()
        draft = Qwen3_5MoeMTP.__new__(Qwen3_5MoeMTP)
        nn.Module.__init__(draft)
        draft.model = nn.Module()
        draft.model.embed_tokens = (
            target.model.embed_tokens if shared else nn.Embedding(4, 2)
        )
        draft.lm_head = target.lm_head if shared else nn.Linear(2, 4, bias=False)
        draft.model.fc = layout_layer()

        class GraphWrapper:
            def __init__(self, runnable):
                self.runnable = runnable

            def __getattr__(self, name):
                return getattr(self.runnable, name)

            def unwrap(self):
                return self.runnable

        config = SimpleNamespace(dtype=torch.float32)
        runner = SimpleNamespace(
            model=GraphWrapper(target),
            model_config=config,
            drafter=SimpleNamespace(model=GraphWrapper(draft)),
            vllm_config=SimpleNamespace(
                speculative_config=SimpleNamespace(
                    method="mtp", draft_model_config=config
                )
            ),
        )
        model = NpuReloadModel(*(module for module, _ in weight_update_models(runner)))
        assert target in model.modules()
        for parameter in model.parameters():
            parameter.weight_loader = weight_loader
        record_metadata_for_reloading(model)
        for layer in (target.layer, draft.model.fc):
            layer.quant_method.process_weights_after_loading(layer)
        snapshots = {name: p.detach() for name, p in model.named_parameters()}
        frozen = draft.model.fc.weight.clone()

        for value in (2.0, 3.0):
            reload = NpuWeightReload(model, SimpleNamespace(dtype=torch.float32))
            reload.start()
            target_weights = [
                ("model.embed_tokens.weight", torch.full((4, 2), value)),
                ("lm_head.weight", torch.full((4, 2), value)),
                ("layer.weight", torch.full((3, 8), value)),
                ("layer.bias_weight", torch.full((3, 8), value + 1)),
            ]
            draft_weights = (
                [
                    ("mtp.fc.weight", torch.full((3, 8), value + 2)),
                    ("mtp.fc.bias_weight", torch.full((3, 8), value + 3)),
                ]
                if train_mtp
                else []
            )
            weights = (
                draft_weights + target_weights
                if mtp_first
                else target_weights + draft_weights
            )
            for weight in weights:
                reload.load([weight])
            reload.finish()
            for name, parameter in model.named_parameters():
                assert parameter.data_ptr() == snapshots[name].data_ptr(), name
            torch.testing.assert_close(
                target.layer.weight, torch.full((8, 3), value), rtol=0, atol=0
            )
            torch.testing.assert_close(
                draft.model.fc.weight,
                torch.full((8, 3), value + 2) if train_mtp else frozen,
                rtol=0,
                atol=0,
            )
            for module in (target, draft):
                for parameter in (
                    module.model.embed_tokens.weight,
                    module.lm_head.weight,
                ):
                    torch.testing.assert_close(
                        parameter, torch.full((4, 2), value), rtol=0, atol=0
                    )


def test_native_reload_preserves_ascend_routing_buffers():
    pytest.importorskip("vllm.model_executor.model_loader.reload")
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

    class AscendRoutedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 3), requires_grad=False)
            self.register_buffer("expert_ids_per_ep_rank", torch.tensor([0, 1, 2]))
            self.register_buffer("log2phy", torch.tensor([2, 0, 1]))
            self.register_buffer("moe_load", torch.tensor([7, 5, 3]))
            self.register_buffer("load_counter", torch.tensor(9))

        def load_weights(self, weights):
            for _, value in weights:
                self.weight.weight_loader(self.weight, value)

    with torch.no_grad(), torch.device("cpu"):
        model = AscendRoutedExperts()
        record_metadata_for_reloading(model)
        expected = {name: buf.clone() for name, buf in model.named_buffers()}
        pointers = {name: buf.data_ptr() for name, buf in model.named_buffers()}
        transaction = NpuWeightReload(model, SimpleNamespace(dtype=torch.float32))
        transaction.start()
        assert not model.log2phy.is_meta
        transaction.load([("weight", torch.full((2, 3), 4.0))])
        transaction.finish()
        torch.testing.assert_close(
            model.weight, torch.full((2, 3), 4.0), rtol=0, atol=0
        )
        for name, buf in model.named_buffers():
            assert buf.data_ptr() == pointers[name]
            torch.testing.assert_close(buf, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_native_reload_packed_experts_across_buckets(ep_rank):
    pytest.importorskip("vllm.model_executor.model_loader.reload")
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
    from vllm.model_executor.utils import replace_parameter

    class LayoutMethod(QuantizeMethodBase):
        def create_weights(self, *args, **kwargs):
            pass

        def apply(self, *args, **kwargs):
            raise NotImplementedError

        def process_weights_after_loading(self, layer):
            for name, parameter in list(layer.named_parameters()):
                replace_parameter(layer, name, parameter.transpose(1, 2).contiguous())
                getattr(layer, name).weight_loader = layer.weight_loader

    class AscendRoutedExperts(RoutedExperts):
        pass

    experts, hidden, intermediate = 4, 8, 2
    local_experts = experts // 2
    first_expert = ep_rank * local_experts
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    nn.Module.__init__(layer)
    layer.layer_name = "experts"
    layer.quant_config = None
    layer.quant_method = LayoutMethod()
    layer.moe_config = SimpleNamespace(
        is_act_and_mul=True,
        tp_rank=0,
        moe_parallel_config=SimpleNamespace(tp_size=1),
        num_experts=experts,
        num_logical_experts=experts,
    )
    layer.expert_map_manager = SimpleNamespace(num_fused_shared_experts=0)
    layer._map_global_expert_id_to_local_expert_id = (
        lambda index: index - first_expert
        if first_expert <= index < first_expert + local_experts
        else -1
    )
    layer.ckpt_gate_proj_name = "gate_proj"
    layer.ckpt_down_proj_name = "down_proj"
    layer.ckpt_up_proj_name = "up_proj"
    layer.lora_base_layer_prefix = ""
    layer.is_fused_checkpoint_transposed = False
    layer.w13_weight = nn.Parameter(
        torch.ones(local_experts, 2 * intermediate, hidden), requires_grad=False
    )
    layer.w2_weight = nn.Parameter(
        torch.ones(local_experts, hidden, intermediate), requires_grad=False
    )
    layer.register_buffer("expert_ids_per_ep_rank", torch.arange(experts))
    for parameter in layer.parameters():
        parameter.weight_loader = layer.weight_loader

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = layer

        def load_weights(self, weights):
            return set(self.experts.load_weights(weights))

    with torch.no_grad(), torch.device("cpu"):
        model = Model()
        record_metadata_for_reloading(model)
        gate_up = torch.arange(experts * 2 * intermediate * hidden).reshape(
            experts, 2 * intermediate, hidden
        )
        down = torch.arange(experts * hidden * intermediate).reshape(
            experts, hidden, intermediate
        )
        model.load_weights([("gate_up_proj", gate_up), ("down_proj", down)])
        layer.quant_method.process_weights_after_loading(layer)
        snapshots = {name: p.detach() for name, p in model.named_parameters()}
        for offset in (1, 2):
            transaction = NpuWeightReload(model, SimpleNamespace(dtype=torch.float32))
            transaction.start()
            transaction.load([("gate_up_proj", gate_up + offset)])
            transaction.load([("down_proj", down + offset)])
            transaction.finish()
            for name, expected in (("w13_weight", gate_up), ("w2_weight", down)):
                parameter = getattr(layer, name)
                assert parameter.data_ptr() == snapshots[f"experts.{name}"].data_ptr()
                expected = expected[first_expert : first_expert + local_experts]
                torch.testing.assert_close(
                    parameter,
                    (expected + offset).transpose(1, 2).float(),
                    rtol=0,
                    atol=0,
                )


def test_native_reload_repacks_ascend_gdn_derived_weight():
    pytest.importorskip("vllm.model_executor.model_loader.reload")
    pytest.importorskip("vllm_ascend")
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
    from vllm_ascend.ops.gdn import initialize_packed_conv_weight

    class LayoutMethod(QuantizeMethodBase):
        def create_weights(self, *args, **kwargs):
            pass

        def apply(self, *args, **kwargs):
            raise NotImplementedError

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_config = SimpleNamespace(dtype=torch.float32)
            self.conv1d = nn.Module()
            self.conv1d.weight = nn.Parameter(torch.zeros(4, 1, 3), requires_grad=False)
            self.conv1d.weight.weight_loader = lambda param, value: param.copy_(value)
            self.conv1d.quant_method = LayoutMethod()
            initialize_packed_conv_weight(self)

        def load_weights(self, weights):
            for _, value in weights:
                weight = self.conv1d.weight
                weight.weight_loader(weight, value)

    with torch.no_grad(), torch.device("cpu"):
        model = Model()
        record_metadata_for_reloading(model)
        model.conv1d.quant_method.process_weights_after_loading(model.conv1d)
        snapshots = {name: p.detach() for name, p in model.named_parameters()}
        for offset in (1, 2):
            expected = torch.arange(12.0).reshape(4, 1, 3) + offset
            transaction = NpuWeightReload(model, model.model_config)
            transaction.start()
            transaction.load([("conv1d.weight", expected)])
            transaction.finish()
            for name, parameter in model.named_parameters():
                assert parameter.data_ptr() == snapshots[name].data_ptr()
            torch.testing.assert_close(model.conv1d.weight, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                model.conv1d.ascend_conv1d_weight,
                expected.squeeze(1).t(),
                rtol=0,
                atol=0,
            )
