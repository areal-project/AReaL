# SPDX-License-Identifier: Apache-2.0

import importlib.util
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from areal.models.mcore.mcore_bridge_adapter import (
    MCoreBridgeAdapter,
    _configure_qwen4_exp_parameters,
    _validate_config_fields,
    qwen4_exp_optimizer_overrides,
)

LAYERS_PREFIX = "model.language_model.layers"
PLE_PREFIX = f"{LAYERS_PREFIX}.1.ple.ple_embedding."


@pytest.fixture
def ple_checkpoint():
    config = SimpleNamespace(
        hf_model_type="qwen4_exp",
        ngram_size=3,
        heads_per_ngram=1,
        ple_embed_dim=4,
        split_ngram_parts=3,
        make_ngram_vocab_size_divisible_by=4,
        ple_layer_ids=[2],
    )
    tensors = {
        PLE_PREFIX + "layer_multipliers": torch.tensor([11, 13, 17]),
        PLE_PREFIX + "ngram_heads_offsets": torch.tensor([0, 5]),
        PLE_PREFIX + "ngram_heads_vocab_sizes": torch.tensor([5, 7]),
        PLE_PREFIX + "ngram_embedding.weight_scale": torch.tensor(0.25),
    }
    for part in range(config.split_ngram_parts):
        tensors[f"{PLE_PREFIX}ngram_embedding.shard_{part}.weight"] = (
            torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
        )
    return config, tensors


def test_load_missing_ple_shard_fails_before_bridge_io(tmp_path, ple_checkpoint):
    pytest.importorskip("mcore_bridge")
    config, tensors = ple_checkpoint
    del tensors[PLE_PREFIX + "ngram_embedding.shard_0.weight"]
    save_file(tensors, tmp_path / "model.safetensors")
    adapter = MCoreBridgeAdapter.__new__(MCoreBridgeAdapter)
    adapter.config = config
    adapter.bridge = SimpleNamespace(
        hf_layers_prefix=LAYERS_PREFIX, load_weights=Mock()
    )

    with pytest.raises(ValueError, match="Missing required PLE"):
        adapter.load_weights([], str(tmp_path))

    adapter.bridge.load_weights.assert_not_called()


@pytest.fixture
def qwen_model_with_embeddings():
    model = torch.nn.Module()
    model.visual = torch.nn.Linear(2, 2)
    model.language_model = torch.nn.Module()
    model.language_model.embedding = torch.nn.Module()
    model.language_model.embedding.word_embeddings = torch.nn.Embedding(8, 2)
    model.language_model.decoder = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attention = torch.nn.Module()
    layer.self_attention.indexer = torch.nn.Linear(2, 2)
    layer.self_attention.linear_qkv = torch.nn.Linear(2, 2)
    layer.ple = torch.nn.Module()
    layer.ple.ple_embedding = torch.nn.Module()
    layer.ple.ple_embedding.cpu_offload = False
    layer.ple.ple_embedding.ngram_embedding = torch.nn.Embedding(8, 2)
    layer.ple.value_proj = torch.nn.Linear(2, 2)
    model.language_model.decoder.layers = torch.nn.ModuleList([layer])
    return model, layer


def test_qwen_embeddings_train_with_zero_decay_only_for_ngram_table(
    qwen_model_with_embeddings,
):
    model, layer = qwen_model_with_embeddings
    token_embedding = model.language_model.embedding.word_embeddings
    ple_table = layer.ple.ple_embedding.ngram_embedding
    token_embedding.requires_grad_(False)
    ple_table.requires_grad_(False)
    with torch.no_grad():
        token_embedding.weight.fill_(0.5)
        ple_table.weight.fill_(0.5)
        layer.ple.value_proj.weight.fill_(0.5)
        layer.ple.value_proj.bias.fill_(0.5)

    frozen = _configure_qwen4_exp_parameters(model, freeze_ple_table=False)

    assert set(frozen) == {
        "visual.weight",
        "visual.bias",
        "language_model.decoder.layers.0.self_attention.indexer.weight",
        "language_model.decoder.layers.0.self_attention.indexer.bias",
    }
    assert all(
        parameter.requires_grad for parameter in layer.ple.value_proj.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in layer.self_attention.linear_qkv.parameters()
    )
    assert token_embedding.weight.requires_grad
    assert ple_table.weight.requires_grad
    assert ple_table.weight.no_weight_decay
    # Exercise the table's marker with real CPU AdamW. The separate MCore test
    # validates ParamKey matching and the actual MCore parameter groups.
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (
                no_decay if getattr(parameter, "no_weight_decay", False) else decay
            ).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=0.01,
    )
    grouped_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(
        id(p) not in grouped_ids for n, p in model.named_parameters() if n in frozen
    )
    assert id(token_embedding.weight) in grouped_ids
    assert id(ple_table.weight) in grouped_ids
    assert any(parameter is ple_table.weight for parameter in no_decay)
    assert any(parameter is token_embedding.weight for parameter in decay)
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    token_ids = torch.tensor([1, 2, 1])
    ngram_ids = torch.tensor([3, 5, 3])
    embeddings = token_embedding(token_ids) + layer.ple.value_proj(ple_table(ngram_ids))
    embeddings.square().mean().backward()
    assert torch.count_nonzero(token_embedding.weight.grad[token_ids]).item() > 0
    assert torch.count_nonzero(ple_table.weight.grad[ngram_ids]).item() > 0
    adam_reference = torch.nn.Parameter(ple_table.weight.detach().clone())
    adam_reference.grad = ple_table.weight.grad.detach().clone()
    reference_optimizer = torch.optim.Adam([adam_reference], lr=0.01, weight_decay=0.0)
    reference_optimizer.step()
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert not torch.equal(
        layer.ple.value_proj.weight,
        before["language_model.decoder.layers.0.ple.value_proj.weight"],
    )
    assert not torch.equal(
        token_embedding.weight[token_ids],
        before["language_model.embedding.word_embeddings.weight"][token_ids],
    )
    assert not torch.equal(
        ple_table.weight[ngram_ids],
        before[
            "language_model.decoder.layers.0.ple.ple_embedding.ngram_embedding.weight"
        ][ngram_ids],
    )
    torch.testing.assert_close(ple_table.weight, adam_reference, rtol=0, atol=0)
    # Unused PLE rows receive neither lookup gradients nor decoupled decay.
    torch.testing.assert_close(
        ple_table.weight[0],
        before[
            "language_model.decoder.layers.0.ple.ple_embedding.ngram_embedding.weight"
        ][0],
        rtol=0,
        atol=0,
    )


def test_qwen_training_rejects_runtime_host_ple_even_when_env_disabled(
    qwen_model_with_embeddings, monkeypatch
):
    model, layer = qwen_model_with_embeddings
    monkeypatch.setenv("PLE_CPU_OFFLOAD", "0")
    layer.ple.ple_embedding.cpu_offload = True
    layer.ple.ple_embedding.host_table = torch.ones(8, 2, requires_grad=True)

    with pytest.raises(NotImplementedError, match="host table has no backward path"):
        _configure_qwen4_exp_parameters(model)


def test_qwen_freeze_ple_table_keeps_small_parameters_trainable(
    qwen_model_with_embeddings,
):
    model, layer = qwen_model_with_embeddings
    frozen = _configure_qwen4_exp_parameters(model, freeze_ple_table=True)

    assert not layer.ple.ple_embedding.ngram_embedding.weight.requires_grad
    assert not hasattr(
        layer.ple.ple_embedding.ngram_embedding.weight, "no_weight_decay"
    )
    assert model.language_model.embedding.word_embeddings.weight.requires_grad
    assert all(
        parameter.requires_grad for parameter in layer.ple.value_proj.parameters()
    )
    assert (
        "language_model.decoder.layers.0.ple.ple_embedding.ngram_embedding.weight"
        in frozen
    )


def test_qwen_training_rejects_unregistered_plain_tensor_table(
    qwen_model_with_embeddings,
):
    model, layer = qwen_model_with_embeddings
    table = layer.ple.ple_embedding.ngram_embedding
    del table.weight
    table.weight = torch.ones(8, 2, requires_grad=True)

    with pytest.raises(ValueError, match="registered trainable Parameter"):
        _configure_qwen4_exp_parameters(model)


def test_qwen_mcore_optimizer_groups_train_embeddings_with_ple_zero_decay(
    qwen_model_with_embeddings, tmp_path
):
    if importlib.util.find_spec("megatron") is None:
        pytest.skip("MCore optimizer grouping requires the pinned training runtime")
    from megatron.core.optimizer import OptimizerConfig, _get_param_groups

    model, layer = qwen_model_with_embeddings
    _configure_qwen4_exp_parameters(model, freeze_ple_table=False)
    config = OptimizerConfig(optimizer="adam", lr=0.01, min_lr=0.0, weight_decay=0.1)
    overrides = qwen4_exp_optimizer_overrides(config)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'optimizer-group-store'}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=20),
    )
    try:
        groups = _get_param_groups([model], config, overrides)
    finally:
        dist.destroy_process_group()
    grouped_parameters = {
        id(parameter): group for group in groups for parameter in group["params"]
    }
    token_embedding = model.language_model.embedding.word_embeddings
    ple_table = layer.ple.ple_embedding.ngram_embedding
    assert grouped_parameters[id(token_embedding.weight)]["wd_mult"] == 1.0
    assert grouped_parameters[id(ple_table.weight)]["wd_mult"] == 0.0
    assert grouped_parameters[id(layer.ple.value_proj.bias)]["wd_mult"] == 0.0
    assert id(layer.self_attention.indexer.weight) not in grouped_parameters
    assert id(model.visual.weight) not in grouped_parameters
    optimizer = torch.optim.AdamW(
        [
            {**group, "weight_decay": config.weight_decay * group["wd_mult"]}
            for group in groups
        ],
        lr=config.lr,
    )
    token_ids = torch.tensor([1, 2, 1])
    ngram_ids = torch.tensor([3, 5, 3])
    before_tokens = token_embedding.weight.detach().clone()
    before_ple = ple_table.weight.detach().clone()
    (
        token_embedding(token_ids).square().mean()
        + ple_table(ngram_ids).square().mean()
    ).backward()
    optimizer.step()
    assert not torch.equal(token_embedding.weight[token_ids], before_tokens[token_ids])
    assert not torch.equal(ple_table.weight[ngram_ids], before_ple[ngram_ids])
    torch.testing.assert_close(ple_table.weight[0], before_ple[0], atol=0, rtol=0)


def test_save_missing_table_does_not_claim_complete_checkpoint(
    tmp_path, ple_checkpoint
):
    pytest.importorskip("mcore_bridge")
    config, tensors = ple_checkpoint
    del tensors[PLE_PREFIX + "ngram_embedding.shard_1.weight"]
    save_file(tensors, tmp_path / "model.safetensors")
    adapter = MCoreBridgeAdapter.__new__(MCoreBridgeAdapter)
    adapter.config = config
    adapter.hf_config = SimpleNamespace(save_pretrained=Mock())
    adapter.bridge = SimpleNamespace(
        hf_layers_prefix=LAYERS_PREFIX, save_weights=Mock()
    )

    with pytest.raises(ValueError, match="Missing required PLE"):
        adapter.save_weights([], str(tmp_path))

    adapter.hf_config.save_pretrained.assert_not_called()


def test_export_omits_non_sender_none_values_and_preserves_valid_tensors():
    tensor = torch.ones(2)
    bridge = SimpleNamespace(
        export_weights=Mock(return_value=iter([("remote", None), ("local", tensor)]))
    )
    adapter = MCoreBridgeAdapter.__new__(MCoreBridgeAdapter)
    adapter.bridge = bridge
    models = [torch.nn.Linear(2, 2)]

    exported = list(adapter.export_hf_weights(models, cpu=True, show_progress=True))

    assert exported == [("local", tensor)]
    bridge.export_weights.assert_called_once_with(
        models, target_device="cpu", disable_tqdm=False
    )


def test_unknown_model_config_fields_raise_instead_of_disappearing():
    @dataclass
    class Config:
        hc_count: int = 4

    _validate_config_fields(Config, {"hc_count": 4})
    with pytest.raises(ValueError, match="ple_embed_dim"):
        _validate_config_fields(Config, {"hc_count": 4, "ple_embed_dim": 32})
