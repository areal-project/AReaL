# SPDX-License-Identifier: Apache-2.0
"""Weight conversion, tensor layouts, and native AWEX transfer planning."""

from types import SimpleNamespace

import pytest
import torch

from areal.models.mcore.qwen4_exp_awex import (
    _MIXER_WEIGHTS,
    _REPLICATED_LAYER_WEIGHTS,
    build_mcore_converter,
    build_sglang_converter,
    build_sharding_strategy,
)
from areal.models.mcore.qwen4_exp_awex_layout import Qwen4ExpGDNLayout


@pytest.fixture
def writer():
    cls = build_mcore_converter()
    instance = cls.__new__(cls)
    instance.rank_info = SimpleNamespace(
        pp_rank=1, pp_size=2, attn_tp_size=4, attn_tp_rank=2
    )
    instance.hf_config = SimpleNamespace(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    instance.tf_config = SimpleNamespace()
    instance.infer_atten_tp_size = 4
    instance._pp_stage_layer_id_map = {(1, 0): {0: 24}, (1, 1): {0: 36}}
    return instance


@pytest.mark.parametrize("suffix", sorted(_REPLICATED_LAYER_WEIGHTS))
@pytest.mark.parametrize("vp,global_layer", [(0, 24), (1, 36)])
def test_replicated_mapping_pp_vpp_matches_inference(writer, suffix, vp, global_layer):
    parameter = torch.randn(3, 5)
    mcore_suffix = suffix.replace("self_attn.indexer.", "self_attention.indexer.")
    actual = writer.convert_param(
        f"module.language_model.decoder.layers.0.{mcore_suffix}", parameter, vp_stage=vp
    )
    canonical = f"model.layers.{global_layer}.{suffix}"
    assert actual[0][0] == canonical
    assert actual[0][1] is parameter
    cls = build_sglang_converter()
    reader = cls.__new__(cls)
    sglang_suffix = suffix.replace("self_attn.indexer.", "indexer.")
    received = reader.convert_param(
        f"model.language_model.layers.{global_layer}.{sglang_suffix}", parameter
    )
    assert received[0][0] == canonical
    assert received[0][1] is parameter
    strategy_cls = build_sharding_strategy()
    strategy = strategy_cls.__new__(strategy_cls)
    from awex.sharding.param_sharding import ShardingType

    assert strategy.get_sharding_strategy(canonical) == (ShardingType.NO_SHARDING, 0, 1)


@pytest.mark.parametrize("suffix", sorted(_MIXER_WEIGHTS))
def test_final_mixer_keeps_weights_without_norm_offset(writer, suffix):
    parameter = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.bfloat16)
    actual = writer.convert_param(f"language_model.decoder.{suffix}", parameter)
    assert actual[0][0] == f"model.{suffix}"
    torch.testing.assert_close(actual[0][1], parameter, rtol=0, atol=0)


def test_gdn_norm_does_not_inherit_qwen35_offset(writer):
    parameter = torch.tensor([0.5, 1.0, 1.5], dtype=torch.bfloat16)
    actual = writer._convert_attention_param(
        "self_attention.out_norm.weight", parameter, "0"
    )
    assert actual[0][0] == "linear_attn.norm.weight"
    torch.testing.assert_close(actual[0][1], parameter, rtol=0, atol=0)


@pytest.mark.parametrize(
    "suffix",
    [
        "ple.ple_embedding.ngram_embedding.weight",
        "self_attention.indexer.unknown.weight",
        "attn_hyper_connection.unknown.weight",
    ],
)
def test_unimplemented_state_rejected_before_generic_fallback(writer, suffix):
    with pytest.raises(NotImplementedError):
        writer.convert_param(f"decoder.layers.0.{suffix}", torch.ones(2, 2))


def test_missing_pp_map_fails_instead_of_using_local_layer_id(writer):
    with pytest.raises(ValueError, match="Missing pp stage"):
        writer.convert_param(
            "decoder.layers.0.attn_hyper_connection.hc_norm.weight",
            torch.ones(4),
            vp_stage=2,
        )


def test_gdn_writer_slices_repacked_tensor_using_training_tp(writer):
    from areal.models.mcore.qwen4_exp_awex_layout import Qwen4ExpGDNLayout

    full = torch.arange(16480 * 3, dtype=torch.float32).reshape(16480, 3)
    # Stand in only for the TP gather; conversion and training-rank selection
    # execute the actual production methods inherited from AWEX.
    writer._full_tp_tensor = lambda parameter: full
    actual = writer._convert_attention_param(
        "self_attention.in_proj.weight", full.chunk(4)[2], "0"
    )
    qkvz, ba = Qwen4ExpGDNLayout(16, 48, 128, 128).pack_input(full, 4, 4)
    assert [name for name, _ in actual] == [
        "linear_attn.in_proj_qkvz.weight",
        "linear_attn.in_proj_ba.weight",
    ]
    for (_, parameter), reference in zip(actual, (qkvz, ba)):
        torch.testing.assert_close(parameter, reference.chunk(4)[2], rtol=0, atol=0)


@pytest.mark.parametrize("component,rows", [("qkvz", 16384), ("ba", 96)])
def test_actual_decoupled_gdn_entry_points(writer, component, rows):
    from areal.models.mcore.qwen4_exp_awex_layout import Qwen4ExpGDNLayout

    full = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    writer._full_tp_tensor = lambda parameter: full
    actual = writer._convert_attention_param(
        f"self_attention.in_proj_{component}.weight", full.chunk(4)[2], "0"
    )
    expected = Qwen4ExpGDNLayout(16, 48, 128, 128).pack_decoupled(full, 4, 4, component)
    assert actual[0][0] == f"linear_attn.in_proj_{component}.weight"
    torch.testing.assert_close(actual[0][1], expected.chunk(4)[2], rtol=0, atol=0)


def test_gdn_a_log_expands_bf16_values_to_native_sglang_float32(writer):
    parameter = torch.tensor([-2.25, 0.5, 1.75], dtype=torch.bfloat16)
    name, actual = writer._convert_attention_param(
        "self_attention.A_log", parameter, "0"
    )[0]
    assert name == "linear_attn.A_log"
    assert actual.dtype == torch.float32
    assert parameter.dtype == torch.bfloat16
    torch.testing.assert_close(actual, torch.tensor([-2.25, 0.5, 1.75]), rtol=0, atol=0)


def test_explicit_registration_resolves_native_factories_without_fallback(monkeypatch):
    from awex.models.registry import (
        ModelRegistry,
        _resolve_converter,
        get_sharding_strategy,
    )

    from areal.models.mcore.qwen4_exp_awex import register_qwen4_exp_awex

    monkeypatch.setattr(ModelRegistry, "models", {})
    register_qwen4_exp_awex()
    register_qwen4_exp_awex()
    config = ModelRegistry.get_model_config("Qwen4ExpForConditionalGeneration")
    assert (
        _resolve_converter(config["mcore_converter"], None) is build_mcore_converter()
    )
    assert (
        _resolve_converter(config["sglang_converter"], None) is build_sglang_converter()
    )
    assert (
        get_sharding_strategy("Qwen4ExpForConditionalGeneration")
        is build_sharding_strategy()
    )


def test_registration_rejects_unrelated_upstream_adapter(monkeypatch):
    from awex.models.registry import ModelRegistry

    from areal.models.mcore.qwen4_exp_awex import register_qwen4_exp_awex

    existing = {"mcore_converter": object()}
    monkeypatch.setattr(
        ModelRegistry, "models", {"Qwen4ExpForConditionalGeneration": existing}
    )
    with pytest.raises(ValueError, match="already registered"):
        register_qwen4_exp_awex()
    assert ModelRegistry.models["Qwen4ExpForConditionalGeneration"] is existing


def test_bound_frozen_contract_excludes_exact_table_and_preserved_visual(writer):
    from areal.models.mcore.qwen4_exp_awex_contract import Qwen4ExpFrozenContract

    table_name = "model.layers.24.ple.ple_embedding.ngram_embedding.weight"
    visual_name = "model.visual.patch_embed.proj.weight"
    contract = Qwen4ExpFrozenContract(
        "a" * 64, frozenset({table_name}), frozenset({visual_name}), True, True
    )
    parameter = torch.nn.Parameter(
        torch.ones(4, 3, dtype=torch.bfloat16), requires_grad=False
    )
    original = {table_name: parameter}
    writer.bind_frozen_contract(contract, original, frozenset({table_name}))
    mcore_name = "module.language_model.decoder.layers.0.ple.ple_embedding.ngram_embedding.weight"
    assert writer.convert_param(mcore_name, parameter.detach(), vp_stage=0) == []
    reader_cls = build_sglang_converter()
    reader = reader_cls.__new__(reader_cls)
    visual = torch.nn.Parameter(torch.ones(3, 4))
    reader.bind_frozen_contract(
        contract, {**original, visual_name: visual}, frozenset({visual_name})
    )
    assert reader.convert_param(table_name, parameter.detach()) == []
    assert reader.convert_param("visual.patch_embed.proj.weight", visual) == []
    qsa_name = "model.layers.24.self_attn.indexer.index_qk_proj.weight"
    result = reader.convert_param(
        "model.layers.24.indexer.index_qk_proj.weight", visual
    )
    assert result[0][0] == qsa_name
    assert result[0][1] is visual
    # Writer payloads are detached, so checking their requires_grad would miss this.
    parameter.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable"):
        writer.convert_param(mcore_name, parameter.detach(), vp_stage=0)


def test_native_registry_binds_every_reader_and_refreshes_original_parameters(
    monkeypatch,
):
    from awex.models.registry import ModelRegistry, get_infer_weights_converter

    from areal.models.mcore.qwen4_exp_awex import register_qwen4_exp_awex
    from areal.models.mcore.qwen4_exp_awex_contract import Qwen4ExpFrozenContract

    table_name = "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    visual_name = "model.visual.patch_embed.proj.weight"
    contract = Qwen4ExpFrozenContract(
        "a" * 64, frozenset({table_name}), frozenset({visual_name}), True, True
    )
    parameters = {
        table_name: torch.nn.Parameter(
            torch.ones(2, 3, dtype=torch.bfloat16), requires_grad=False
        ),
        visual_name: torch.nn.Parameter(torch.ones(3, 4)),
    }
    bound = []

    def bind(converter):
        converter.bind_frozen_contract(
            contract, dict(parameters), frozenset({visual_name})
        )
        bound.append(converter)

    monkeypatch.setattr(ModelRegistry, "models", {})
    register_qwen4_exp_awex(sglang_binder=bind)
    register_qwen4_exp_awex(sglang_binder=bind)
    args = (
        "sglang",
        "Qwen4ExpForConditionalGeneration",
        SimpleNamespace(num_attention_heads=24, num_key_value_heads=2),
        SimpleNamespace(tp_rank=0, ep_rank=0),
        SimpleNamespace(tp_size=4, ep_size=1, device_backend="cpu"),
    )
    metadata = get_infer_weights_converter(*args)
    payload = get_infer_weights_converter(*args)
    assert bound == [metadata, payload]
    for converter in (metadata, payload):
        assert (
            converter.convert_param(table_name, parameters[table_name].detach()) == []
        )
    replacement = torch.nn.Parameter(
        torch.zeros(2, 3, dtype=torch.bfloat16), requires_grad=False
    )
    parameters[table_name] = replacement
    payload.refresh_frozen_contract()
    assert payload._qwen4_original_parameters[table_name] is replacement
    # Recovery to an invalid model must invalidate the previously valid binding.
    parameters[table_name] = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="trainable"):
        payload.refresh_frozen_contract()
    with pytest.raises(NotImplementedError):
        payload.convert_param(table_name, replacement.detach())


def test_native_registry_rejects_binder_that_does_not_bind(monkeypatch):
    from awex.models.registry import ModelRegistry, get_infer_weights_converter

    from areal.models.mcore.qwen4_exp_awex import register_qwen4_exp_awex

    monkeypatch.setattr(ModelRegistry, "models", {})
    register_qwen4_exp_awex(sglang_binder=lambda converter: None)
    with pytest.raises(ValueError, match="did not bind"):
        get_infer_weights_converter(
            "sglang",
            "Qwen4ExpForConditionalGeneration",
            SimpleNamespace(num_attention_heads=24, num_key_value_heads=2),
            SimpleNamespace(tp_rank=0, ep_rank=0),
            SimpleNamespace(tp_size=4, ep_size=1, device_backend="cpu"),
        )


def _labels(heads, widths, tail):
    # Encode each semantic coordinate independently of the implementation's
    # reshape/split operations. MCore concatenates whole head groups.
    rows = []
    lookup = {}
    for head in range(heads):
        for category, width in enumerate(widths):
            for channel in range(width):
                value = category * 100000 + head * 1000 + channel
                rows.append(value)
                lookup[category, head, channel] = value
    tensor = torch.tensor(rows, dtype=torch.int64)
    return tensor.reshape(-1, *([1] * len(tail))).expand(-1, *tail).clone(), lookup


def _expected(lookup, heads, widths, categories, infer_tp, tail):
    rows = []
    for rank in range(infer_tp):
        for category in categories:
            for head in range(rank * heads // infer_tp, (rank + 1) * heads // infer_tp):
                for channel in range(widths[category]):
                    rows.append(lookup[category, head, channel])
    return (
        torch.tensor(rows, dtype=torch.int64)
        .reshape(-1, *([1] * len(tail)))
        .expand(-1, *tail)
    )


@pytest.mark.parametrize("train_tp", [1, 2, 4, 8])
@pytest.mark.parametrize("infer_tp", [1, 2, 4, 8])
def test_gdn_packing_multiple_heads_preserves_semantic_coordinates(train_tp, infer_tp):
    """Actual model head geometry; narrow hidden width keeps this CPU test small."""
    layout = Qwen4ExpGDNLayout(16, 48, 128, 128)
    widths = (128, 128, 384, 384, 3, 3)
    source, lookup = _labels(16, widths, (3,))
    original = source.clone()
    qkvz, ba = layout.pack_input(source, train_tp, infer_tp)
    for actual, categories in ((qkvz, range(4)), (ba, range(4, 6))):
        expected = _expected(lookup, 16, widths, categories, infer_tp, (3,))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(source, original, rtol=0, atol=0)

    conv, conv_lookup = _labels(16, widths[:3], (1, 4))
    actual_conv = layout.pack_conv(conv, train_tp, infer_tp)
    expected_conv = _expected(conv_lookup, 16, widths[:3], range(3), infer_tp, (1, 4))
    torch.testing.assert_close(actual_conv, expected_conv, rtol=0, atol=0)
    for component, sizes in (("qkvz", widths[:4]), ("ba", widths[4:])):
        decoupled, labels = _labels(16, sizes, (3,))
        actual = layout.pack_decoupled(decoupled, train_tp, infer_tp, component)
        expected = _expected(labels, 16, sizes, range(len(sizes)), infer_tp, (3,))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("train_tp,infer_tp", [(0, 4), (4, 0), (3, 4), (4, 32)])
def test_gdn_invalid_tp_raises(train_tp, infer_tp):
    layout = Qwen4ExpGDNLayout(16, 48, 128, 128)
    with pytest.raises(ValueError, match="TP sizes"):
        layout.pack_input(torch.empty(16480, 3), train_tp, infer_tp)


def test_gdn_local_tensor_rejected_as_full_tensor():
    layout = Qwen4ExpGDNLayout(16, 48, 128, 128)
    with pytest.raises(ValueError, match="full GDN tensor"):
        layout.pack_input(torch.empty(4120, 3), 4, 4)


@pytest.mark.parametrize(
    "heads,values,key_dim,value_dim",
    [(0, 48, 128, 128), (16, 47, 128, 128), (16, 48, 0, 128)],
)
def test_gdn_invalid_geometry_raises(heads, values, key_dim, value_dim):
    with pytest.raises(ValueError):
        Qwen4ExpGDNLayout(heads, values, key_dim, value_dim)


@pytest.mark.parametrize("infer_tp", [1, 2, 4, 8])
@pytest.mark.parametrize("tail", [(), (3,)])
def test_gated_qkv_preserves_head_gate_pairs_and_replicates_kv(infer_tp, tail):
    from areal.models.mcore.qwen4_exp_awex_layout import pack_qwen4_exp_gated_qkv

    heads, kv_heads, dim = 24, 2, 4
    queries = [
        [10000 + h * 100 + g * 10 + c for g in range(2) for c in range(dim)]
        for h in range(heads)
    ]
    keys = [[20000 + h * 100 + c for c in range(dim)] for h in range(kv_heads)]
    values = [[30000 + h * 100 + c for c in range(dim)] for h in range(kv_heads)]
    source = []
    for kv in range(kv_heads):
        for head in range(kv * 12, (kv + 1) * 12):
            source.extend(queries[head])
        source.extend(keys[kv])
        source.extend(values[kv])
    expected = []
    for rank in range(infer_tp):
        for head in range(rank * heads // infer_tp, (rank + 1) * heads // infer_tp):
            expected.extend(queries[head])
        owners = range(kv_heads) if infer_tp == 1 else [rank // (infer_tp // kv_heads)]
        for category in (keys, values):
            for owner in owners:
                expected.extend(category[owner])

    def tensor(rows):
        return torch.tensor(rows).reshape(-1, *([1] * len(tail))).expand(-1, *tail)

    actual = pack_qwen4_exp_gated_qkv(tensor(source), heads, kv_heads, dim, infer_tp)
    torch.testing.assert_close(actual, tensor(expected), rtol=0, atol=0)


def _metadata(raw):
    from awex.meta.meta_resolver import ParamMetaResolver

    class Resolver(ParamMetaResolver):
        def get_model_arch_name(self):
            return "Qwen4ExpForConditionalGeneration"

        def get_parameters_meta(self):
            return self._build_params_meta()

        def _get_params_raw_meta(self):
            return raw

        def _get_sharding_info(self, name, rank_info, param_meta):
            strategy = build_sharding_strategy()(
                engine_name="sglang" if rank_info.is_infer else "mcore",
                enable_dp_attention=False,
                enable_dp_lm_head=False,
                moe_dense_tp_size=rank_info.tp_size,
                tp_size=rank_info.tp_size,
                ep_size=1,
                ep_tp_size=1,
                rank_info=rank_info,
            )
            return strategy.get_sharding_strategy(name)

    return Resolver(SimpleNamespace(num_hidden_layers=48)).get_parameters_meta()


def _rank(tp, rank, pp, pp_rank, dp, dp_rank, inference):
    from awex.sharding.rank_info import RankInfo

    global_rank = dp_rank * pp * tp + pp_rank * tp + rank
    return RankInfo(
        tp_rank=rank,
        tp_size=tp,
        pp_rank=pp_rank,
        pp_size=pp,
        dp_rank=dp_rank,
        dp_size=dp,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=rank,
        attn_tp_size=tp,
        attn_dp_rank=dp_rank,
        world_size=tp * pp * dp,
        global_rank=global_rank,
        local_rank=global_rank % 8,
        engine_rank=0,
        is_infer=inference,
    )


def _raw(ranks, full_weights, tp):
    entries, tensors = [], {}
    for rank in ranks:
        params = []
        for name, full in full_weights.items():
            replicated = "hyper_connection" in name
            value = full if replicated else full.chunk(tp, dim=0)[rank.tp_rank]
            params.append(
                dict(
                    name=name,
                    shape=tuple(value.shape),
                    numel=value.numel(),
                    dtype=value.dtype,
                )
            )
            tensors[name, rank.global_rank] = value.clone()
        entries.append(
            dict(
                rank_info=rank,
                params_meta=params,
                model_arch_name="Qwen4ExpForConditionalGeneration",
            )
        )
    return entries, tensors


@pytest.mark.parametrize(
    "train_tp,infer_tp,dp,owner_pp",
    [(4, 4, 1, 1), (8, 4, 2, 3), (2, 4, 2, 1), (4, 8, 1, 1)],
)
def test_native_metadata_plan_reconstructs_every_destination_once(
    train_tp, infer_tp, dp, owner_pp
):
    from awex.transfer.transfer_plan import TransferPlanBuilder

    layout = Qwen4ExpGDNLayout(16, 48, 128, 128)
    source = torch.arange(16480 * 3, dtype=torch.float32).reshape(16480, 3)
    qkvz, ba = layout.pack_input(source, train_tp, infer_tp)
    conv = layout.pack_conv(
        torch.arange(10240 * 4, dtype=torch.float32).reshape(10240, 1, 4),
        train_tp,
        infer_tp,
    )
    weights = {
        "model.layers.24.linear_attn.in_proj_qkvz.weight": qkvz,
        "model.layers.24.linear_attn.in_proj_ba.weight": ba,
        "model.layers.24.linear_attn.conv1d.weight": conv,
        "model.layers.24.attn_hyper_connection.input_mix_weight_down.weight": torch.arange(
            15, dtype=torch.float32
        ).reshape(3, 5),
    }
    pp = owner_pp + 1
    train_ranks = [
        _rank(train_tp, rank, pp, owner_pp, dp, replica, False)
        for replica in range(dp)
        for rank in range(train_tp)
    ]
    infer_ranks = [_rank(infer_tp, rank, 1, 0, 1, 0, True) for rank in range(infer_tp)]
    train_raw, train_tensors = _raw(train_ranks, weights, train_tp)
    infer_raw, expected = _raw(infer_ranks, weights, infer_tp)
    train_meta, infer_meta = _metadata(train_raw), _metadata(infer_raw)
    for meta in train_meta + infer_meta:
        assert tuple(meta.global_shape) == tuple(weights[meta.name].shape)
    builder = TransferPlanBuilder(
        infer_world_size=infer_tp,
        train_world_size=train_tp * pp * dp,
        num_infer_engines=1,
        strict_param_key_match=True,
    )
    ops = builder.build_weights_mapping_operations(infer_meta, train_meta)
    destinations = {key: torch.empty_like(value) for key, value in expected.items()}
    written = {
        key: torch.zeros_like(value, dtype=torch.bool)
        for key, value in expected.items()
    }
    for op in ops:
        src_key = (op.send_shard_meta.name, op.send_rank - infer_tp)
        dst_key = (op.recv_shard_meta.name, op.recv_rank)
        assert not written[dst_key][op.inf_slices].any()
        destinations[dst_key][op.inf_slices].copy_(
            train_tensors[src_key][op.train_slices]
        )
        written[dst_key][op.inf_slices] = True
    for key in expected:
        assert written[key].all(), key
        torch.testing.assert_close(destinations[key], expected[key], rtol=0, atol=0)
