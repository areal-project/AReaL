# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

from tests.test_megatron_bridge_config import _load_function

from areal.api.cli_args import MegatronEngineConfig, to_structured_cfg
from areal.engine.megatron_utils.frozen_mtp import (
    FrozenMTPState,
    validate_mtp_checkpoint,
)
from areal.engine.megatron_utils.mtp import (
    _MTP_TRAIN_LABELS,
    _detach_mtp_output_weight,
    _IsolatedEmbedding,
    _wrap_forward_for_mtp_kwargs,
    mtp_backward_scale,
    prepare_mtp_mask,
    prepare_mtp_models_for_ddp,
)
from areal.engine.vllm_ext.npu_weight_reload import NpuReloadModel, weight_update_models


@dataclass
class _Config:
    megatron: MegatronEngineConfig = field(default_factory=MegatronEngineConfig)


@pytest.mark.parametrize("enabled", [False, True])
def test_mtp_config_training_flag_yaml(enabled):
    assert MegatronEngineConfig().enable_mtp_training is False
    cfg = OmegaConf.create(
        {
            "megatron": {
                "enable_mtp_training": enabled,
                "bridge_type": "megatron-bridge",
            }
        }
    )
    result = OmegaConf.to_object(to_structured_cfg(cfg, _Config))
    assert result.megatron.enable_mtp_training is enabled


@pytest.mark.parametrize("coefficient", [-1, float("inf"), float("nan")])
def test_mtp_invalid_coefficient_rejected(coefficient):
    with pytest.raises(ValueError, match="finite and non-negative"):
        MegatronEngineConfig(mtp_loss_scaling_factor=coefficient)


def test_mtp_other_bridge_rejected():
    with pytest.raises(ValueError, match="megatron-bridge"):
        MegatronEngineConfig(enable_mtp_training=True)


@pytest.mark.parametrize("fail", [False, True])
def test_hf_rank_zero_propagates_checkpoint_result(fail):
    """Every rank reaches the broadcast before a checkpoint error is raised."""
    group = object()
    broadcasts = []
    failure = [None]
    calls = []
    error = OSError("checkpoint unavailable")

    def operation():
        calls.append(rank)
        if fail:
            raise error

    def broadcast(payload, *, group, group_src):
        assert group is engine.cpu_group
        assert group_src == 0
        broadcasts.append(rank)
        if rank == 0:
            failure[:] = payload
        else:
            payload[:] = failure

    run = _load_function(
        "areal/engine/megatron_engine.py",
        "_run_hf_rank_zero",
        {
            "dist": SimpleNamespace(
                get_rank=lambda group: rank, broadcast_object_list=broadcast
            )
        },
    )
    engine = SimpleNamespace(cpu_group=group)
    for rank in (0, 1):
        if fail:
            with pytest.raises(
                OSError if rank == 0 else RuntimeError, match="checkpoint unavailable"
            ) as raised:
                run(engine, operation)
            if rank == 0:
                assert raised.value is error
        else:
            run(engine, operation)
    assert calls == [0]
    assert broadcasts == [0, 1]


def checkpoint(path, value, filename="model.safetensors"):
    path.mkdir()
    tensors = {
        "mtp.fc.weight": torch.full((2, 2), value),
        "model.weight": torch.ones(2),
    }
    save_file(tensors, path / filename)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: filename for k in tensors}})
    )
    (path / "config.json").write_text(json.dumps({"mtp_num_hidden_layers": 1}))


def read_all(path):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    result = {}
    for name, filename in index["weight_map"].items():
        with safe_open(path / filename, framework="pt") as f:
            result[name] = f.get_tensor(name)
    assert index["metadata"]["total_size"] == sum(
        t.numel() * t.element_size() for t in result.values()
    )
    return result


def test_frozen_snapshot_survives_source_deletion_and_load_b(tmp_path):
    a, b, out = (tmp_path / p for p in ("a", "b", "out"))
    checkpoint(a, 1.0)
    checkpoint(b, 7.0, "different-shard.safetensors")
    state = FrozenMTPState.load(str(a))
    state = FrozenMTPState.load(str(b))
    shutil.rmtree(a)
    shutil.rmtree(b)
    out.mkdir()
    save_file({"model.weight": torch.full((2,), 9.0)}, out / "model.safetensors")
    (out / "config.json").write_text('{"mtp_num_hidden_layers": 0}')
    state.export(str(out))
    tensors = read_all(out)
    torch.testing.assert_close(
        tensors["mtp.fc.weight"], torch.full((2, 2), 7.0), rtol=0, atol=0
    )
    torch.testing.assert_close(
        tensors["model.weight"], torch.full((2,), 9.0), rtol=0, atol=0
    )
    assert json.loads((out / "config.json").read_text())["mtp_num_hidden_layers"] == 1


def test_frozen_mixed_shard_retains_trained_backbone(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    checkpoint(src, 2.0)
    checkpoint(out, 0.0)
    FrozenMTPState.load(str(src)).export(str(out))
    tensors = read_all(out)
    assert tensors["mtp.fc.weight"].sum() == 8
    assert tensors["model.weight"].sum() == 2


def test_frozen_missing_declared_tensor_rejected(tmp_path):
    checkpoint(tmp_path / "src", 1.0)
    (tmp_path / "src/model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.missing": "model.safetensors"}})
    )
    with pytest.raises(Exception, match="File does not contain tensor"):
        FrozenMTPState.load(str(tmp_path / "src"))


def test_frozen_export_missing_backbone_rejected(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    checkpoint(src, 1.0)
    checkpoint(out, 1.0)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lost.backbone": "missing.safetensors"}})
    )
    with pytest.raises(ValueError, match="Incomplete HF export"):
        FrozenMTPState.load(str(src)).export(str(out))


def test_mtp_mask_excludes_packed_ends_padding_and_short_sequences():
    # Original lengths 3,1,2 aligned to 4 each. Last masks deliberately invalid.
    mask = torch.tensor([1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0])
    actual, count = prepare_mtp_mask(
        mask, torch.tensor([0, 4, 8, 12]), torch.tensor([0, 3, 4, 6])
    )
    assert actual.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    assert count == 1
    assert mask[2] == 1


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("cp,dp,microbatches", [(1, 1, 1), (2, 1, 3), (2, 2, 4)])
def test_mtp_gradient_scale_is_global_token_mean(per_token, cp, dp, microbatches):
    # Uneven microbatch/rank CE sums, including a rank with no main supervision.
    leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    numerator = (leaf * torch.tensor([2.0, 0.0, 5.0])).sum()
    scale = mtp_backward_scale(
        torch.tensor(11.0 * cp),
        torch.tensor(7.0),
        cp_size=cp,
        dp_cp_size=cp * dp,
        num_microbatches=microbatches,
        per_token=per_token,
    )
    loss = numerator * scale
    loss = loss / 11 if per_token else loss / microbatches / (dp * cp)
    loss.backward()
    torch.testing.assert_close(
        leaf.grad, torch.tensor([2.0, 0.0, 5.0]) / 7, rtol=1e-6, atol=1e-7
    )


def test_mtp_zero_target_scale_finite():
    scale = mtp_backward_scale(
        torch.tensor(3.0),
        torch.tensor(0.0),
        cp_size=1,
        dp_cp_size=1,
        num_microbatches=1,
        per_token=True,
    )
    assert torch.isfinite(scale)


@pytest.mark.parametrize("tied", [False, True])
def test_mtp_output_head_receives_no_auxiliary_gradient(tied):
    weight = torch.nn.Parameter(torch.randn(5, 3))
    hidden = torch.randn(2, 3, requires_grad=True)
    detached = _detach_mtp_output_weight(
        SimpleNamespace(weight=weight), weight if tied else None
    )
    (hidden @ detached.t()).sum().backward()
    assert weight.grad is None
    assert hidden.grad is not None


def test_mtp_context_cleared_after_forward_error():
    class Model:
        def forward(self, **kwargs):
            assert "mtp_kwargs" not in kwargs
            assert _MTP_TRAIN_LABELS.get() is not None
            raise ValueError("forward failed")

    _wrap_forward_for_mtp_kwargs(Model)
    with pytest.raises(ValueError, match="forward failed"):
        Model().forward(mtp_kwargs={"mtp_labels": torch.ones(2)})
    assert _MTP_TRAIN_LABELS.get() is None


def test_mtp_draft_receives_new_head_and_shared_weights():
    class DraftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = {}

        def load_weights(self, weights):
            self.weights.update(weights)

    model = DraftModel()
    target = torch.nn.Module()
    target.load_weights = lambda weights: set()
    runner = SimpleNamespace(
        model=target,
        model_config=object(),
        drafter=SimpleNamespace(model=model),
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="mtp", draft_model_config=object()
            )
        ),
    )
    weights = [
        (name, torch.tensor([2.0]))
        for name in (
            "mtp.fc.weight",
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
        )
    ]
    reload_model = NpuReloadModel(*(m for m, _ in weight_update_models(runner)))
    reload_model.load_weights(weights)
    assert set(model.weights) == {k for k, v in weights}


def test_mtp_mask_handles_added_batch_padding_segment():
    mask = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0])
    result, count = prepare_mtp_mask(
        mask, torch.tensor([0, 4, 8]), torch.tensor([0, 3])
    )
    assert torch.equal(result, mask)
    assert count == 1


def test_frozen_single_file_contains_mtp_for_hf_resolution(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    checkpoint(source, 3.0, "source-only.safetensors")
    output.mkdir()
    save_file({"model.weight": torch.ones(2)}, output / "model.safetensors")
    (output / "config.json").write_text("{}")
    state = FrozenMTPState.load(str(source))
    state.export(str(output))
    state.export(str(output))
    with safe_open(output / "model.safetensors", framework="pt") as f:
        assert "mtp.fc.weight" in f.keys()
        assert f.get_tensor("mtp.fc.weight").sum() == 12


def test_frozen_sidecar_collision_preserves_backbone(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    checkpoint(source, 3.0, "source-only.safetensors")
    output.mkdir()
    save_file({"model.weight": torch.ones(2)}, output / "model-mtp.safetensors")
    (output / "config.json").write_text("{}")
    state = FrozenMTPState.load(str(source))
    state.export(str(output))
    state.export(str(output))
    tensors = read_all(output)
    assert tensors["model.weight"].sum() == 2
    assert tensors["mtp.fc.weight"].sum() == 12
    assert list(output.glob("*.safetensors")) == [output / "model-mtp.safetensors"]


def test_mtp_embedding_keeps_zero_gradient_hook():
    embedding = torch.nn.Embedding(5, 3)
    called = []
    embedding.weight.register_hook(lambda grad: called.append(grad.clone()))
    values = embedding(torch.tensor([1, 2]))
    _IsolatedEmbedding.apply(values).sum().backward()
    assert len(called) == 1
    assert torch.count_nonzero(embedding.weight.grad) == 0


@pytest.mark.parametrize(
    "mtp,tied,post,output_kind,freeze",
    [
        (True, True, True, "separate", True),
        (False, True, True, "separate", False),
        (True, False, True, "separate", False),
        (True, True, False, "separate", False),
        (True, True, True, "shared", False),
        (True, True, True, "absent", False),
    ],
)
def test_mtp_ddp_preserves_active_and_checkpoint_weights(
    mtp, tied, post, output_kind, freeze
):
    """Only the unused tied PP output copy is excluded from gradient tracking."""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mtp_process = mtp
            self.share_embeddings_and_output_weights = tied
            self.post_process = post
            self.embedding = torch.nn.Embedding(5, 3)
            self.output_layer = torch.nn.Linear(3, 5, bias=False)
            if output_kind == "shared":
                self.output_layer.weight = self.embedding.weight
            elif output_kind == "absent":
                self.output_layer.register_parameter("weight", None)

        def shared_embedding_or_output_weight(self):
            return self.embedding.weight

    model = Model()
    models = [torch.nn.Sequential(model)]
    state_keys = set(model.state_dict())
    assert prepare_mtp_models_for_ddp(models) is models
    assert model.embedding.weight.requires_grad
    if model.output_layer.weight is not None:
        assert model.output_layer.weight.requires_grad is not freeze
    assert set(model.state_dict()) == state_keys
    # Bridge must still be able to copy the tied checkpoint into the output slot.
    state = {name: torch.ones_like(value) for name, value in model.state_dict().items()}
    model.load_state_dict(state, strict=True)
    for value in model.state_dict().values():
        torch.testing.assert_close(value, torch.ones_like(value), rtol=0, atol=0)


@pytest.mark.parametrize("loader", [FrozenMTPState.load, validate_mtp_checkpoint])
def test_mtp_load_rejects_absent_head(tmp_path, loader):
    checkpoint(tmp_path / "source", 1.0)
    save_file({"model.weight": torch.ones(2)}, tmp_path / "source/model.safetensors")
    (tmp_path / "source/model.safetensors.index.json").unlink()
    with pytest.raises(ValueError, match="No (frozen )?MTP weights"):
        loader(str(tmp_path / "source"))
