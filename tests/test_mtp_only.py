# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from areal.api.cli_args import MegatronEngineConfig
from areal.engine.megatron_utils.mtp_only import freeze_non_mtp_parameters


@pytest.fixture(autouse=True)
def supported_mtp_runtime(monkeypatch):
    from areal.utils import pkg_version

    original = pkg_version.get_version
    versions = {"megatron-core": "0.18.2", "megatron-bridge": "0.5.1"}
    monkeypatch.setattr(
        pkg_version,
        "get_version",
        lambda name: versions[name] if name in versions else original(name),
    )
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 91900)
    return versions


def _config(**kwargs) -> MegatronEngineConfig:
    options = dict(
        bridge_type="megatron-bridge",
        enable_mtp=True,
        enable_mtp_training=True,
        mtp_only=True,
    )
    options.update(kwargs)
    return MegatronEngineConfig(**options)


def test_mtp_only_default_keeps_joint_training() -> None:
    """Existing configurations do not acquire parameter freezing."""
    assert not MegatronEngineConfig().mtp_only
    assert _config().mtp_only


@pytest.mark.parametrize(
    "options, message",
    [
        ({"enable_mtp_training": False}, "requires enable_mtp_training"),
        ({"enable_mtp": False}, "requires enable_mtp"),
        ({"bridge_type": "mbridge"}, "requires bridge_type"),
        ({"mtp_loss_scaling_factor": 0}, "finite, positive"),
        ({"mtp_loss_scaling_factor": -1}, "finite, positive"),
        ({"mtp_loss_scaling_factor": float("nan")}, "finite, positive"),
        ({"mtp_loss_scaling_factor": float("inf")}, "finite, positive"),
        ({"use_custom_fsdp": True}, "FSDP wrappers"),
        ({"use_torch_fsdp2": True}, "FSDP wrappers"),
    ],
)
def test_mtp_only_invalid_config_raises(options, message) -> None:
    """Reject configurations that cannot produce isolated MTP updates."""
    with pytest.raises(ValueError, match=message):
        _config(**options)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(11, 4)
        self.backbone = nn.Linear(4, 4)
        self.mtp = nn.Linear(4, 4)
        self.output_layer = nn.Linear(4, 11, bias=False)


@pytest.mark.parametrize("wrapped", [False, True])
def test_freeze_non_mtp_optimizer_updates_only_mtp(wrapped: bool) -> None:
    """A real optimizer step preserves every backbone/shared parameter exactly."""
    torch.manual_seed(7)
    model = _Model()
    root = nn.ModuleDict({"language_model": model}) if wrapped else model
    models = [root]
    assert freeze_non_mtp_parameters(models) is models
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        [p for p in root.parameters() if p.requires_grad], lr=0.01, weight_decay=0.1
    )
    hidden = model.backbone(model.embedding(torch.tensor([1, 2, 3])))
    assert not hidden.requires_grad
    loss = F.cross_entropy(
        model.output_layer(model.mtp(hidden)), torch.tensor([3, 4, 5])
    )
    loss.backward()
    assert model.mtp.weight.grad.norm() > 0
    optimizer.step()
    for name, p in model.named_parameters():
        if name.startswith("mtp."):
            assert not torch.equal(p, before[name])
        else:
            assert p.grad is None
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("mtp_registered_first", [False, True])
def test_freeze_shared_alias_stays_frozen(mtp_registered_first: bool) -> None:
    """MTP aliases of shared output/embedding weights never become trainable."""
    shared = nn.Linear(4, 4, bias=False)
    mtp = nn.ModuleDict({"shared": shared, "projection": nn.Linear(4, 4)})
    entries = [("mtp", mtp), ("output_layer", shared)]
    model = nn.ModuleDict(entries if mtp_registered_first else entries[::-1])
    freeze_non_mtp_parameters([model])
    assert not shared.weight.requires_grad
    assert mtp["projection"].weight.requires_grad


def test_freeze_missing_mtp_raises_without_mutation() -> None:
    """A stripped checkpoint must not silently produce a no-op training run."""
    model = nn.ModuleDict({"not_mtp": nn.Linear(4, 4)})
    with pytest.raises(ValueError, match="no MTP-specific parameters"):
        freeze_non_mtp_parameters([model])
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_pipeline_stage_without_mtp_freezes_all_parameters() -> None:
    """An explicitly validated earlier PP stage needs no trainable weights."""
    model = nn.ModuleDict({"backbone": nn.Linear(4, 4)})
    models = [model]
    assert freeze_non_mtp_parameters(models, allow_missing_mtp=True) is models
    assert all(not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("has_parameters", [False, True])
def test_optimizer_step_empty_stage_preserves_step_and_scheduled_lr(
    has_parameters: bool,
) -> None:
    """Frozen PP ranks must participate in optimizer collectives every step."""
    from unittest.mock import Mock

    pytest.importorskip("megatron.core")
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    from areal.engine.megatron_engine import MegatronEngine

    optimizer = SimpleNamespace(
        param_groups=[{}] if has_parameters else [],
        step=Mock(return_value=(True, 0.25, None)),
    )
    scheduler = OptimizerParamScheduler(
        optimizer=optimizer,
        init_lr=0.0,
        max_lr=0.01,
        min_lr=0.001,
        lr_warmup_steps=2,
        lr_decay_steps=10,
        lr_decay_style="linear",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=10,
        wd_incr_style="constant",
    )
    engine = SimpleNamespace(
        optimizer=optimizer,
        lr_scheduler=scheduler,
        _dte_runtime_config=SimpleNamespace(enabled=False),
    )
    for step in range(5):
        if step:
            scheduler.step(1)
        metrics = MegatronEngine.optimizer_step(engine)
        assert metrics == {
            "update_successful": 1.0,
            "grad_norm": 0.25,
            "lr": scheduler.get_lr({}),
        }
        assert optimizer.step.call_count == step + 1


@pytest.mark.parametrize("first_stage", [False, True])
@pytest.mark.parametrize("deallocate", [False, True])
def test_frozen_pipeline_backward_preserves_activation_gradients(
    first_stage: bool, deallocate: bool
) -> None:
    """Use the real PP backward, including pseudo-deallocated output tensors."""
    schedules = pytest.importorskip("megatron.core.pipeline_parallel.schedules")
    layer = nn.Linear(4, 4).requires_grad_(False)
    inputs = torch.randn(3, 4, requires_grad=not first_stage)
    output = layer(inputs)
    output_grad = torch.randn_like(output)
    expected = output_grad @ layer.weight
    if deallocate:
        schedules.deallocate_output_tensor(output, True)
    input_grad = schedules.backward_step(
        None if first_stage else inputs,
        output,
        output_grad,
        SimpleNamespace(
            timers=None, grad_scale_func=None, deallocate_pipeline_outputs=deallocate
        ),
    )
    if first_stage:
        assert input_grad is None
    else:
        torch.testing.assert_close(input_grad, expected, rtol=1e-6, atol=1e-7)
    assert all(p.grad is None for p in layer.parameters())


def test_mcore_auxiliary_backward_with_frozen_main_matches_direct_loss() -> None:
    """Use MCore's real autograd carrier, including a frozen output projection."""
    mtp_module = pytest.importorskip("megatron.core.transformer.multi_token_prediction")
    scaler = mtp_module.MTPLossAutoScaler
    torch.manual_seed(11)
    model = _Model()
    freeze_non_mtp_parameters([model])
    reference = copy.deepcopy(model)
    ids, labels = torch.tensor([1, 2, 3]), torch.tensor([3, 4, 5])
    hidden = model.backbone(model.embedding(ids))
    mtp_loss = F.cross_entropy(model.output_layer(model.mtp(hidden)), labels)
    previous_scale = scaler.main_loss_backward_scale
    try:
        scaler.set_loss_scale(torch.tensor(1.0))
        carried_hidden = scaler.apply(hidden, 0.1 * mtp_loss)
        assert carried_hidden.requires_grad
        main_loss = F.cross_entropy(model.output_layer(carried_hidden), labels)
        main_loss.backward()
    finally:
        scaler.main_loss_backward_scale = previous_scale
    ref_hidden = reference.backbone(reference.embedding(ids))
    (
        0.1 * F.cross_entropy(reference.output_layer(reference.mtp(ref_hidden)), labels)
    ).backward()
    for (name, p), (_, ref_p) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        if name.startswith("mtp."):
            assert p.grad.norm() > 0
            torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-6, atol=1e-7)
        else:
            assert p.grad is None


@pytest.mark.parametrize("mtp_only", [False, True])
@pytest.mark.parametrize("expert_bias", [False, True])
@pytest.mark.parametrize("stage", ["single", "first", "last", "missing_last"])
@pytest.mark.parametrize("fused_loss", [False, True])
def test_registry_freezes_before_distributed_wrap(
    monkeypatch, mtp_only: bool, expert_bias: bool, stage: str, fused_loss: bool
) -> None:
    """Exercise registry wiring and preserve hooks installed by the provider."""
    pytest.importorskip("megatron.core")
    from areal.models.mcore import registry

    model = _Model()
    if stage in ("first", "missing_last"):
        del model.mtp
    events = []

    def existing_hook(models):
        events.append("existing")
        assert all(p.requires_grad for p in model.parameters())
        return models

    class Provider:
        mtp_num_layers = 1
        pipeline_model_parallel_layout = None
        cross_entropy_loss_fusion = not fused_loss
        moe_router_enable_expert_bias = expert_bias

        def __init__(self):
            self.hooks = [existing_hook]

        def finalize(self):
            assert self.cross_entropy_loss_fusion is fused_loss

        def register_pre_wrap_hook(self, hook):
            self.hooks.append(hook)

        def provide_distributed_model(self, **kwargs):
            models = [model]
            for hook in self.hooks:
                models = hook(models)
            events.append("wrap")
            assert model.backbone.weight.requires_grad is not mtp_only
            if hasattr(model, "mtp"):
                assert model.mtp.weight.requires_grad
            return models

    provider = Provider()
    bridge = SimpleNamespace(to_megatron_provider=lambda **kwargs: provider)
    for getter in (
        "get_tensor_model_parallel_world_size",
        "get_pipeline_model_parallel_world_size",
        "get_context_parallel_world_size",
        "get_expert_model_parallel_world_size",
        "get_expert_tensor_parallel_world_size",
    ):
        monkeypatch.setattr(registry.mpu, getter, lambda: 1)
    monkeypatch.setattr(
        registry.mpu,
        "get_pipeline_model_parallel_world_size",
        lambda: 1 if stage == "single" else 2,
    )
    monkeypatch.setattr(
        registry.mpu,
        "is_pipeline_last_stage",
        lambda **kwargs: stage != "first",
    )
    monkeypatch.setattr(
        registry.mpu,
        "get_pipeline_model_parallel_rank",
        lambda: 0 if stage in ("single", "first") else 1,
    )
    monkeypatch.setattr(registry, "_configure_actor_output_layers", lambda *args: None)
    arguments = dict(
        hf_config=SimpleNamespace(),
        tf_config=SimpleNamespace(params_dtype=torch.float32, fp16=False, bf16=False),
        mcore_config=_config(mtp_only=mtp_only, cross_entropy_loss_fusion=fused_loss),
        bridge=bridge,
        bridge_type="megatron-bridge",
    )
    if mtp_only and expert_bias:
        with pytest.raises(ValueError, match="expert-bias updates"):
            registry.make_mcore_model(**arguments)
        assert events == []
        return
    if mtp_only and stage == "missing_last":
        with pytest.raises(ValueError, match="no MTP-specific parameters"):
            registry.make_mcore_model(**arguments)
        assert events == ["existing"]
        return
    models = registry.make_mcore_model(**arguments)
    assert models == [model]
    assert events == ["existing", "wrap"]


@pytest.mark.parametrize("options", [{"use_lora": True}, {"is_critic": True}])
def test_registry_mtp_only_unsupported_mode_raises(monkeypatch, options) -> None:
    """Fail before constructing models for unsupported training modes."""
    pytest.importorskip("megatron.core")
    from areal.models.mcore import registry

    options = options.copy()
    pp = options.pop("pp", 1)
    monkeypatch.setattr(
        registry.mpu, "get_pipeline_model_parallel_world_size", lambda: pp
    )
    with pytest.raises(ValueError, match="mtp_only"):
        registry.make_mcore_model(
            hf_config=SimpleNamespace(),
            tf_config=SimpleNamespace(),
            mcore_config=_config(),
            bridge=object(),
            bridge_type="megatron-bridge",
            **options,
        )


@pytest.mark.parametrize("keyword_input", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_mtp_full_recompute_matches_direct_gradients(
    monkeypatch, keyword_input: bool, wrapped: bool
) -> None:
    """Run MCore's actual MTP forward and checkpoint autograd on CPU."""
    from contextlib import nullcontext

    mtp_module = pytest.importorskip("megatron.core.transformer.multi_token_prediction")
    from megatron.core.tensor_parallel import random as mcore_random

    from areal.engine.megatron_utils.megatron_bridge_patches import (
        _patch_mtp_checkpoint_padding_mask,
    )

    _patch_mtp_checkpoint_padding_mask()

    # Only CUDA RNG bookkeeping is replaced; checkpoint forward/backward is real.
    monkeypatch.setattr(mcore_random, "_get_all_rng_states", lambda: ())
    monkeypatch.setattr(mcore_random, "_set_all_rng_states", lambda *args: None)
    monkeypatch.setattr(mcore_random, "_fork_rng", nullcontext)

    class MTPLayer(mtp_module.MultiTokenPredictionLayer):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
                fp8=None,
                fp4=None,
                distribute_saved_activations=False,
            )
            self.projection = nn.Linear(8, 4)

        def _get_embeddings(
            self,
            input_ids,
            position_ids,
            embedding,
            hidden_states,
            packed_seq_params,
            **kwargs,
        ):
            layout = (input_ids, position_ids)
            if "padding_mask" in kwargs:
                layout += (kwargs["padding_mask"],)
            return (*layout, embedding(input_ids), hidden_states)

        def _proj_and_transformer_layer(
            self, hidden_states, decoder_input, *args, **kwargs
        ):
            return self.projection(torch.cat((hidden_states, decoder_input), dim=-1))

    torch.manual_seed(19)
    model = _Model()
    model.mtp = MTPLayer()
    root = nn.ModuleDict({"language_model": model}) if wrapped else model
    freeze_non_mtp_parameters([root])
    reference = copy.deepcopy(model)
    reference.mtp.config.recompute_granularity = None
    ids, labels = torch.tensor([1, 2, 3]), torch.tensor([3, 4, 5])
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    for current in (model, reference):
        hidden = current.backbone(current.embedding(ids))
        if keyword_input:
            output = current.mtp(
                input_ids=ids,
                position_ids=ids,
                hidden_states=hidden,
                attention_mask=None,
                embedding=current.embedding,
            )[0]
        else:
            output = current.mtp(ids, ids, hidden, None, embedding=current.embedding)[0]
        assert not hidden.requires_grad
        F.cross_entropy(current.output_layer(output), labels).backward()
    for (name, parameter), (_, ref_parameter) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        if name.startswith("mtp."):
            assert parameter.grad is not None
            assert parameter.grad.norm() > 0
            torch.testing.assert_close(
                parameter.grad, ref_parameter.grad, rtol=1e-6, atol=1e-7
            )
        else:
            assert parameter.grad is None
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name.startswith("mtp."):
            assert not torch.equal(parameter, before[name])
        else:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_mtp_input_hook_preserves_autograd_and_no_grad() -> None:
    """The hook is idempotent and preserves existing graphs and inference inputs."""

    class MTP(nn.Linear):
        def forward(self, input_ids, position_ids, hidden_states):
            self.seen_hidden = hidden_states
            return super().forward(hidden_states)

    model = _Model()
    model.mtp = MTP(4, 4)
    freeze_non_mtp_parameters([model])
    freeze_non_mtp_parameters([model])
    assert len(model.mtp._forward_pre_hooks) == 1
    leaf = torch.randn(3, 4, requires_grad=True)
    hidden = leaf * 2
    model.mtp(None, None, hidden).sum().backward()
    assert model.mtp.seen_hidden is hidden
    assert leaf.grad is not None
    frozen_hidden = torch.randn(3, 4)
    with torch.no_grad():
        output = model.mtp(None, None, frozen_hidden)
    assert model.mtp.seen_hidden is frozen_hidden
    assert not output.requires_grad


def test_mtp_checkpoint_compatibility_preserves_masks_or_rejects_them(monkeypatch):
    """The compatibility shim must never silently discard a real padding mask."""
    mtp = pytest.importorskip("megatron.core.transformer.multi_token_prediction")
    from areal.engine.megatron_utils.megatron_bridge_patches import (
        _patch_mtp_checkpoint_padding_mask,
    )

    def forward(self, hidden_states, padding_mask=None):
        return hidden_states

    def old_checkpoint(self, hidden_states):
        return hidden_states.square()

    monkeypatch.setattr(mtp.MultiTokenPredictionLayer, "forward", forward)
    monkeypatch.setattr(
        mtp.MultiTokenPredictionLayer, "_checkpointed_forward", old_checkpoint
    )
    _patch_mtp_checkpoint_padding_mask()
    patched = mtp.MultiTokenPredictionLayer._checkpointed_forward
    _patch_mtp_checkpoint_padding_mask()
    assert mtp.MultiTokenPredictionLayer._checkpointed_forward is patched
    hidden = torch.randn(3, 4, requires_grad=True)
    patched(None, hidden, padding_mask=None).sum().backward()
    torch.testing.assert_close(hidden.grad, 2 * hidden, rtol=0, atol=0)
    mask = torch.ones(3, 4, dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="MTP padding mask"):
        patched(None, hidden, padding_mask=mask)

    def fixed_checkpoint(self, hidden_states, padding_mask=None):
        assert padding_mask is mask
        return hidden_states

    monkeypatch.setattr(
        mtp.MultiTokenPredictionLayer, "_checkpointed_forward", fixed_checkpoint
    )
    _patch_mtp_checkpoint_padding_mask()
    assert mtp.MultiTokenPredictionLayer._checkpointed_forward is fixed_checkpoint
    assert fixed_checkpoint(None, hidden, padding_mask=mask) is hidden


@pytest.mark.parametrize(
    "package, version",
    [
        ("megatron-core", "0.18.1"),
        ("megatron-core", "0.18.2rc1"),
        ("megatron-bridge", "0.5.0"),
        ("megatron-bridge", "0.5.1rc1"),
    ],
)
def test_mtp_only_old_package_raises(supported_mtp_runtime, package, version):
    supported_mtp_runtime[package] = version
    with pytest.raises(ValueError, match=package):
        _config()


@pytest.mark.parametrize("package", ["megatron-core", "megatron-bridge"])
def test_mtp_only_missing_package_raises(monkeypatch, package):
    from importlib.metadata import PackageNotFoundError

    from areal.utils import pkg_version

    original = pkg_version.get_version

    def version(name):
        if name == package:
            raise PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(pkg_version, "get_version", version)
    with pytest.raises(ValueError, match=f"{package}.*not installed"):
        _config()


@pytest.mark.parametrize("version", [None, 91600, 91899])
def test_mtp_only_old_or_missing_loaded_cudnn_raises(monkeypatch, version):
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: version)
    with pytest.raises(ValueError, match="loaded cuDNN>=9.19.0"):
        _config()


def test_mtp_only_cudnn_load_failure_raises(monkeypatch):
    def unavailable():
        raise RuntimeError("incompatible shared libraries")

    monkeypatch.setattr(torch.backends.cudnn, "version", unavailable)
    with pytest.raises(ValueError, match="PyTorch could not load cuDNN"):
        _config()


@pytest.mark.parametrize("version", [91900, 92000])
def test_mtp_only_supported_runtime_passes(monkeypatch, supported_mtp_runtime, version):
    supported_mtp_runtime["megatron-core"] = "0.18.2+vendor.1"
    supported_mtp_runtime["megatron-bridge"] = "0.6.0"
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: version)
    assert _config().mtp_only


def test_joint_training_does_not_query_mtp_runtime(monkeypatch):
    from areal.utils import pkg_version

    def unexpected(*args):
        pytest.fail("Joint training must not query MTP-only runtime requirements")

    monkeypatch.setattr(pkg_version, "get_version", unexpected)
    monkeypatch.setattr(torch.backends.cudnn, "version", unexpected)
    assert not _config(mtp_only=False).mtp_only
