# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from types import (
    ModuleType,
    SimpleNamespace,
)
from typing import Any

import pytest
import torch
from awex.models.qwen4_exp import (
    _MIXER_WEIGHTS,
    build_mcore_converter,
    build_sharding_strategy,
)
from awex.models.qwen4_exp_contract import Qwen4ExpFrozenContract
from awex.models.qwen4_exp_layout import Qwen4ExpGDNLayout
from awex.transfer.nccl_bounded_stream import (
    BoundedMemoryNcclColocateStreamBatchTransport,
)
from torch import nn

from areal.engine.awex.colocate_reader import _DeviceBoundWeightsReader
from areal.models.mcore.qwen4_exp_awex_binding import McoreFrozenBinder
from areal.models.mcore.qwen4_exp_awex_memory import install_kv_residency_hooks
from areal.models.mcore.qwen4_exp_frozen_state import (
    snapshot_visual_parameters,
)


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


@pytest.mark.parametrize("mixer_suffix", [None, *sorted(_MIXER_WEIGHTS)])
def test_qwen_norm_and_final_mixer_do_not_inherit_qwen35_offset(writer, mixer_suffix):
    parameter = torch.tensor([0.5, 1.0, 1.5], dtype=torch.bfloat16)
    if mixer_suffix is None:
        actual = writer._convert_attention_param(
            "self_attention.out_norm.weight", parameter, "0"
        )
        expected_name = "linear_attn.norm.weight"
    else:
        actual = writer.convert_param(
            f"language_model.decoder.{mixer_suffix}", parameter
        )
        expected_name = f"model.{mixer_suffix}"
    assert actual[0][0] == expected_name
    torch.testing.assert_close(actual[0][1], parameter, rtol=0, atol=0)


@pytest.mark.parametrize("component,rows", [("qkvz", 16384), ("ba", 96)])
def test_actual_decoupled_gdn_entry_points(writer, component, rows):
    from awex.models.qwen4_exp_layout import Qwen4ExpGDNLayout

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


TABLE = "model.layers.1.ple.ple_embedding.ngram_embedding.weight"


VISUAL = "model.visual.patch_embed.proj.weight"


QSA = "model.layers.3.self_attn.indexer.index_qk_proj.weight"


@pytest.fixture
def contract():
    return Qwen4ExpFrozenContract(
        "a" * 64, frozenset({TABLE}), frozenset({VISUAL}), True, True
    )


def table():
    return nn.Parameter(torch.ones(4, 3, dtype=torch.bfloat16), requires_grad=False)


def frozen_checkpoint(tmp_path, *, global_id=0):
    import json

    from safetensors.torch import save_file

    from areal.models.mcore.qwen4_exp_awex_contract import FrozenCheckpoint

    prefix = (
        f"model.language_model.layers.{global_id}.ple.ple_embedding.ngram_embedding"
    )
    values = {
        prefix + ".shard_0.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        prefix + ".shard_1.weight": torch.ones(1, 2, dtype=torch.bfloat16),
        "model.visual.weight": torch.ones(3, 2, dtype=torch.bfloat16),
    }
    save_file(values, tmp_path / "weights.safetensors")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "text_config": {"split_ngram_parts": 2},
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {n: "weights.safetensors" for n in values},
            }
        )
    )
    return FrozenCheckpoint(tmp_path, True)


@pytest.mark.parametrize(
    "fault", ["none", "values", "coverage", "replica", "scale", "dtype"]
)
def test_automatic_frozen_proof_checks_contents_and_ownership(tmp_path, fault):
    import shutil

    from safetensors.torch import load_file, save_file

    from areal.models.mcore.qwen4_exp_awex_contract import (
        FrozenCheckpoint,
        check_frozen_proof,
        merge_frozen_proofs,
    )

    checkpoint = frozen_checkpoint(tmp_path)
    name = next(iter(checkpoint.tables))
    emb = nn.Embedding(3, 2, dtype=torch.bfloat16)
    emb.requires_grad_(False)
    emb.weight.data.fill_(1)
    emb.vocab_start_index, emb.vocab_end_index = 0, 3
    proof = checkpoint.verify_table(name, emb, "actor")
    proof["model.visual.weight"] = checkpoint.verify_source("model.visual.weight")
    record = {
        "contract": checkpoint.contract.to_dict(),
        "proof": proof,
        "ranges": {name: [0, 3]},
    }
    expected = merge_frozen_proofs(checkpoint, [record, record])
    copied = tmp_path / "copy"
    copied.mkdir()
    for file in ("config.json", "model.safetensors.index.json", "weights.safetensors"):
        shutil.copyfile(tmp_path / file, copied / file)
    other = FrozenCheckpoint(copied, True)
    assert other.contract == checkpoint.contract
    if fault == "coverage":
        record["ranges"][name] = [0, 2]
        with pytest.raises(ValueError, match="ownership"):
            merge_frozen_proofs(checkpoint, [record])
    elif fault == "replica":
        wrong = dict(proof)
        wrong["model.visual.weight"] = {"sha256": "changed"}
        with pytest.raises(ValueError, match="contents differ"):
            merge_frozen_proofs(checkpoint, [record, {**record, "proof": wrong}])
    else:
        emb.shard_indices = SimpleNamespace(
            org_vocab_start_index=0, org_vocab_end_index=3
        )
        emb.org_vocab_size = 3
        emb.weight_scale = torch.tensor([2 if fault == "scale" else 1])
        if fault == "values":
            values = load_file(copied / "weights.safetensors")
            values[other.tables[name][0]].zero_()
            save_file(values, copied / "weights.safetensors")
            emb.weight.data[:2].zero_()  # agrees locally, disagrees with actor
        elif fault == "dtype":
            emb.weight = nn.Parameter(emb.weight.float(), requires_grad=False)
        if fault == "none":
            check_frozen_proof(other.verify_table(name, emb, "inference"), expected)
        else:
            with pytest.raises(ValueError):
                check_frozen_proof(other.verify_table(name, emb, "inference"), expected)


def setup_binding(tmp_path, global_id=0, mapped=False, table=True):
    checkpoint = frozen_checkpoint(tmp_path, global_id=global_id)
    layer = nn.Module()
    layer.layer_number = global_id + 1
    layer.ple = nn.Module()
    layer.ple.ple_embedding = nn.Module()
    if table:
        layer.ple.ple_embedding.ngram_embedding = nn.Embedding(
            3, 2, dtype=torch.bfloat16
        )
        emb = layer.ple.ple_embedding.ngram_embedding
        emb.weight.requires_grad_(False)
        emb.weight.data.fill_(1)
        emb.vocab_start_index, emb.vocab_end_index = 0, 3
    chunk = nn.Module()
    chunk.decoder = nn.Module()
    chunk.decoder.layers = nn.ModuleList([layer])
    engine = SimpleNamespace(
        model=[chunk],
        config=SimpleNamespace(path=str(tmp_path)),
        pipeline_parallel_rank=0,
        bridge_cls="mcore-bridge",
        mcore_config=SimpleNamespace(language_model_only=True, freeze_ple_table=True),
        hf_config=SimpleNamespace(architectures=["Qwen4ExpForConditionalGeneration"]),
    )
    contract = checkpoint.contract
    converter_cls = build_mcore_converter()
    converter = converter_cls.__new__(converter_cls)
    converter.rank_info = SimpleNamespace(pp_rank=1)
    converter._pp_stage_layer_id_map = {(1, 0): {0: global_id}} if mapped else {}
    return engine, converter, McoreFrozenBinder(engine, contract)


def test_production_conversion_refreshes_binding_before_detach(tmp_path):
    from areal.engine.awex.colocate_writer import AwexWeightPublisher

    engine, unused, binder = setup_binding(tmp_path)
    cls = build_mcore_converter(binder)
    converter = cls.__new__(cls)
    converter.rank_info = SimpleNamespace(pp_rank=0, pp_size=1)
    converter._pp_stage_layer_id_map = {}
    converter.hf_config = engine.hf_config
    converter.tf_config = SimpleNamespace()
    adapter = AwexWeightPublisher(engine)
    adapter._qwen4_frozen_binder = binder
    adapter._weight_converter = converter
    adapter._rank_info = converter.rank_info
    assert adapter._convert_parameters() == {}
    embedding = engine.model[0].decoder.layers[0].ple.ple_embedding.ngram_embedding
    old = embedding.weight
    embedding.weight = nn.Parameter(old.detach().clone(), requires_grad=False)
    assert adapter._convert_parameters() == {}
    assert next(iter(converter._qwen4_original_parameters.values())) is embedding.weight
    embedding.weight = nn.Parameter(torch.ones_like(old))
    with pytest.raises(ValueError, match="trainable"):
        adapter._convert_parameters()
    assert getattr(converter, "_qwen4_frozen_contract", None) is None


def _vision_binding(tmp_path, *, pp_rank=0):
    engine, converter, old_binder = setup_binding(tmp_path)
    chunk = engine.model[0]
    chunk.pre_process = pp_rank == 0
    contract = replace(old_binder.contract, language_model_only=False, schema_version=2)
    engine.mcore_config.language_model_only = False
    converter.rank_info.pp_rank = pp_rank
    if pp_rank == 0:
        chunk.visual = nn.Module()
        chunk.visual.visual = nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
        chunk.visual.requires_grad_(False)
        chunk.visual.visual.weight.data.fill_(1)
    engine.config = SimpleNamespace(path=str(tmp_path))
    return engine, converter, McoreFrozenBinder(engine, contract)


@pytest.mark.parametrize("pp_rank", [0, 1, 3])
def test_vision_binding_and_converter_respect_pp_ownership(tmp_path, pp_rank):
    engine, converter, binder = _vision_binding(tmp_path, pp_rank=pp_rank)
    binder(converter)
    expected = frozenset({"model.visual.weight"}) if pp_rank == 0 else frozenset()
    assert converter._qwen4_local_visual_names == expected
    if pp_rank == 0:
        original = engine.model[0].visual.visual.weight
        assert converter._qwen4_original_parameters["model.visual.weight"] is original
        assert (
            converter.convert_param("module.visual.visual.weight", original.detach())
            == []
        )
        original.requires_grad_(True)
        with pytest.raises(ValueError, match="trainable"):
            converter.convert_param("module.visual.visual.weight", original.detach())
    else:
        with pytest.raises(ValueError, match="not owned"):
            converter.convert_param("visual.visual.weight", torch.ones(3, 2))


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "shape", "values", "trainable", "owner", "mode"]
)
def test_vision_binding_rejects_invalid_live_state(tmp_path, fault):
    engine, converter, binder = _vision_binding(tmp_path)
    chunk = engine.model[0]
    if fault == "missing":
        chunk.visual = None
    elif fault == "extra":
        chunk.visual.visual.register_parameter("unknown", table())
    elif fault == "shape":
        chunk.visual.visual.weight = nn.Parameter(
            torch.ones(2, 2, dtype=torch.bfloat16), requires_grad=False
        )
    elif fault == "values":
        chunk.visual.visual.weight.data.zero_()
    elif fault == "trainable":
        chunk.visual.visual.requires_grad_(True)
    elif fault == "owner":
        chunk.pre_process = False
    else:
        engine.mcore_config.language_model_only = True
    with pytest.raises(ValueError):
        binder(converter)


@pytest.fixture
def lifecycle(monkeypatch):
    events = []
    memory = SimpleNamespace(resident=True, enabled=True)
    monkeypatch.setattr(
        "areal.models.mcore.qwen4_exp_awex_memory.torch.get_device_module",
        lambda: SimpleNamespace(synchronize=lambda: events.append("sync")),
    )

    class Scheduler:
        def __init__(self):
            self._engine_paused = True
            self.running_batch = SimpleNamespace(is_empty=lambda: True)
            self.idle = True
            self.flush_success = True

        def flush_cache(self):
            if not memory.resident:
                raise RuntimeError("write to unmapped KV")
            events.append("clear")
            return self.idle and self.flush_success

    @dataclass(slots=True)
    class Manager:
        scheduler: Any
        tp_worker: Any
        memory_saver_adapter: Any
        is_fully_idle: Any
        flush_cache: Any
        fail_resume: bool = False

        def release_memory_occupation(self, request):
            if not request.tags or "kv_cache" in request.tags:
                assert self.is_fully_idle()
                events.append("unmap")
                memory.resident = False
                self.flush_cache()
            return "released"

        def resume_memory_occupation(self, request):
            if self.fail_resume:
                raise RuntimeError("mapping failed")
            if not request.tags or "kv_cache" in request.tags:
                events.append("map")
                memory.resident = True
            return "resumed"

    module = ModuleType("fake_weight_updater")
    module.SchedulerWeightUpdaterManager = Manager
    module.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    install_kv_residency_hooks(module, Scheduler)
    scheduler = Scheduler()
    model = type("Qwen4ExpForConditionalGeneration", (), {})()
    manager = Manager(
        scheduler,
        SimpleNamespace(model_runner=SimpleNamespace(model=model)),
        memory,
        lambda: scheduler.idle,
        scheduler.flush_cache,
    )
    return SimpleNamespace(
        scheduler=scheduler,
        manager=manager,
        memory=memory,
        events=events,
        module=module,
        scheduler_type=Scheduler,
    )


@pytest.mark.parametrize("tags", [["kv_cache"], ["weights", "kv_cache"], None, []])
def test_kv_multiple_cycles_reset_only_resident_memory(lifecycle, tags):
    case = lifecycle
    request = SimpleNamespace(tags=tags)
    bound_flush = case.manager.flush_cache
    for _ in range(2):
        assert case.manager.release_memory_occupation(request) == "released"
        assert not case.memory.resident
        assert case.manager.flush_cache is bound_flush
        assert case.scheduler.flush_cache() is True
        assert case.manager.resume_memory_occupation(request) == "resumed"
        assert case.memory.resident
    assert case.events == ["clear", "sync", "unmap", "map", "clear", "sync"] * 2


def test_busy_release_does_not_clear_or_unmap(lifecycle):
    lifecycle.scheduler.idle = False
    with pytest.raises(RuntimeError, match="idle or retract-paused"):
        lifecycle.manager.release_memory_occupation(SimpleNamespace(tags=["kv_cache"]))
    assert lifecycle.memory.resident
    assert lifecycle.events == []


def test_failed_flush_does_not_unmap(lifecycle):
    lifecycle.scheduler.flush_success = False
    with pytest.raises(RuntimeError, match="before KV release"):
        lifecycle.manager.release_memory_occupation(SimpleNamespace(tags=["kv_cache"]))
    assert lifecycle.memory.resident
    assert lifecycle.events == ["clear"]


def test_failed_resume_does_not_touch_unmapped_cache(lifecycle):
    case = lifecycle
    request = SimpleNamespace(tags=["kv_cache"])
    case.manager.release_memory_occupation(request)
    case.manager.fail_resume = True
    with pytest.raises(RuntimeError, match="mapping failed"):
        case.manager.resume_memory_occupation(request)
    assert case.scheduler._areal_qwen4_exp_kv_state.released
    assert case.events == ["clear", "sync", "unmap"]


class Qwen4ExpForConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(3, 2)
        self.model = nn.Linear(3, 2)


def test_static_hooks_preserve_visual_and_native_buffer_lifecycle():
    from types import SimpleNamespace

    from areal.models.mcore.qwen4_exp_frozen_state import install_static_state_hooks

    events = []
    updater = SimpleNamespace(
        _export_static_state=lambda model: {"native": "buffer-state"},
        _import_static_state=lambda model, state: events.append(state["native"]),
    )
    install_static_state_hooks(updater)
    exporter = updater._export_static_state
    install_static_state_hooks(updater)
    assert updater._export_static_state is exporter
    model = Qwen4ExpForConditionalGeneration()
    initial = snapshot_visual_parameters(model)
    for _ in range(2):
        state = updater._export_static_state(model)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(42)
        updater._import_static_state(model, state)
        for name, parameter in model.named_parameters():
            if name in initial:
                torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
    assert events == ["buffer-state", "buffer-state"]
    with pytest.raises(ValueError, match="not saved"):
        updater._import_static_state(model, {"native": "buffer-state"})


def test_reader_selects_bounded_awex_transport(monkeypatch):
    import awex.transfer.nccl_stream_batch as stream_batch

    monkeypatch.setattr(stream_batch.device_util, "create_stream", lambda: object())
    reader = object.__new__(_DeviceBoundWeightsReader)
    reader.transfer_rank = 7
    reader.infer_world_size = 64

    transport = reader.create_colocate_transport()

    assert isinstance(transport, BoundedMemoryNcclColocateStreamBatchTransport)
    assert transport.transfer_rank == 7
    assert transport.world_size == 64


def test_frozen_binding_revalidates_in_place_recovery(tmp_path):
    engine, converter, binder = setup_binding(tmp_path)
    binder.verify_loaded()
    parameter = (
        engine.model[0].decoder.layers[0].ple.ple_embedding.ngram_embedding.weight
    )
    with torch.no_grad():
        parameter.add_(1)
    binder.invalidate()
    with pytest.raises(ValueError, match="differs from checkpoint"):
        binder.verify_loaded()


@pytest.mark.parametrize(
    "kind",
    ["QKVParallelLinear", "ColumnParallelLinear", "RowParallelLinear", "Conv3dLayer"],
)
def test_frozen_visual_verification_uses_layer_tp(tmp_path, kind):
    from safetensors.torch import save_file

    from areal.models.mcore.qwen4_exp_awex_contract import visual_segments

    checkpoint = frozen_checkpoint(tmp_path)
    name = next(iter(checkpoint.visual_sources))
    source_name = checkpoint.visual_sources[name]
    source = torch.arange(48, dtype=torch.bfloat16).reshape(12, 4)
    # Each source is in a separate file so the PLE fixture remains intact.
    checkpoint.weight_map[source_name] = "vision.safetensors"
    save_file({source_name: source}, str(tmp_path / "vision.safetensors"))
    module = type(kind, (), {})()
    module.tp_rank, module.tp_size = 1, 2
    module.kv_tp_rank, module.kv_tp_size = 1, 2
    module.total_num_heads = module.total_num_kv_heads = 4
    module.head_size = module.v_head_size = 1
    module.output_partition_sizes = [6]
    if kind == "QKVParallelLinear":
        local = torch.cat([part.chunk(2)[1] for part in source.chunk(3)])
    elif kind == "ColumnParallelLinear":
        local = source.chunk(2)[1]
    elif kind == "RowParallelLinear":
        local = source.chunk(2, dim=1)[1]
    else:
        local = source
    segments, shape = visual_segments(module, name, tuple(source.shape))
    assert tuple(local.shape) == shape
    parameter = nn.Parameter(local.clone(), requires_grad=False)
    checkpoint.verify_source(name, parameter, segments)
    with torch.no_grad():
        parameter[0, 0].add_(1)
    with pytest.raises(ValueError, match="differs from checkpoint"):
        checkpoint.verify_source(name, parameter, segments)


def test_inference_frozen_binding_ignores_padding_and_rechecks_load(tmp_path):
    from areal.models.mcore.qwen4_exp_awex_binding import SglangFrozenBinder

    engine, _, actor = setup_binding(tmp_path)
    model = type("Qwen4ExpForConditionalGeneration", (nn.Module,), {})()
    model.model = nn.Module()
    model.model.layers = engine.model[0].decoder.layers
    model.visual = nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
    model.visual.weight = nn.Parameter(
        torch.ones(3, 2, dtype=torch.bfloat16), requires_grad=False
    )
    embedding = model.model.layers[0].ple.ple_embedding.ngram_embedding
    embedding.org_vocab_size = 3
    embedding.shard_indices = SimpleNamespace(
        org_vocab_start_index=2, org_vocab_end_index=3
    )
    embedding.weight = nn.Parameter(
        torch.tensor([[1, 1], [99, 99]], dtype=torch.bfloat16), requires_grad=False
    )
    embedding.weight_scale = torch.ones(1)
    checkpoint = actor.checkpoint
    expected = {
        name: checkpoint.verify_source(name) for name in checkpoint.source_names
    }
    binder = SglangFrozenBinder(
        lambda: model,
        actor.contract,
        SimpleNamespace(_areal_qwen4_exp_static_hooks=True),
        checkpoint,
        expected,
    )
    binder.verify_loaded()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    name = "model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    state[name][0, 0] = 2
    original = embedding.weight
    model.load_state_dict(state)
    assert embedding.weight is original
    with pytest.raises(ValueError, match="differs from checkpoint"):
        binder.verify_loaded()
