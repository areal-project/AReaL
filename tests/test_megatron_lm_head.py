# SPDX-License-Identifier: Apache-2.0

import gc
import weakref
from dataclasses import fields

import pytest
import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel import layers as mcore_layers
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module

from areal.api.cli_args import MegatronEngineConfig
from areal.models.mcore import registry as mcore_registry
from areal.models.mcore import vocab_parallel_head as lm_head_module
from areal.models.mcore.vocab_parallel_head import (
    AReaLVocabParallelLMHead,
    _ChunkedVocabParallelLMHead,
    _matches_storage,
    _pack_fp32_to_half_inplace,
    linear_with_fp32_output,
    replace_output_layer_with_areal_lm_head,
)
from areal.utils.functional.vocab_parallel import gather_logprobs_entropy
from areal.utils.functional.vocab_parallel_kernels import (
    reusable_vocab_parallel_logits,
)

CUDA_AVAILABLE = torch.cuda.is_available()
FUSED_WGRAD_AVAILABLE = getattr(mcore_layers, "_grad_accum_fusion_available", False)


class _ModelWithOutputLayer(torch.nn.Module):
    def __init__(self, output_layer: torch.nn.Module) -> None:
        super().__init__()
        self.output_layer = output_layer


class _BF16OutputModule(torch.nn.Module):
    def forward(self) -> torch.Tensor:
        return torch.ones(2, dtype=torch.bfloat16)


def _make_uninitialized_head() -> mcore_layers.ColumnParallelLinear:
    head = mcore_layers.ColumnParallelLinear.__new__(mcore_layers.ColumnParallelLinear)
    torch.nn.Module.__init__(head)
    head.weight = torch.nn.Parameter(torch.empty(8, 4))
    head.register_parameter("bias", None)
    return head


def test_areal_lm_head_is_enabled_by_default():
    config = MegatronEngineConfig()
    config_fields = {field.name: field for field in fields(MegatronEngineConfig)}

    assert config.use_areal_lm_head is True
    assert config.lm_head_loss_chunk_size == 0
    assert config.enable_fp32_lm_head is False
    assert "Deprecated" in config_fields["enable_fp32_lm_head"].metadata["help"]


def test_chunked_lm_head_config_validation():
    with pytest.raises(ValueError, match="must be non-negative"):
        MegatronEngineConfig(lm_head_loss_chunk_size=-1)
    with pytest.raises(ValueError, match="requires use_areal_lm_head"):
        MegatronEngineConfig(
            use_areal_lm_head=False,
            lm_head_loss_chunk_size=128,
        )
    with pytest.raises(ValueError, match="does not support enable_mtp"):
        MegatronEngineConfig(
            lm_head_loss_chunk_size=128,
            enable_mtp=True,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_actor_output_layer_replacement_uses_fused_fp32_output(enabled, monkeypatch):
    model = object()
    gpt_model = object()
    replacements: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        mcore_registry,
        "unwrap_to_gpt_model",
        lambda candidate: gpt_model if candidate is model else None,
    )
    monkeypatch.setattr(
        mcore_registry,
        "replace_output_layer_with_areal_lm_head",
        lambda candidate, *, fp32_output: replacements.append((candidate, fp32_output)),
    )

    mcore_registry._replace_actor_output_layers(
        [model],
        enabled=enabled,
    )

    assert replacements == ([(gpt_model, True)] if enabled else [])


def _native_megatron_areal_logits(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Original MCore BF16 linear followed by AReaL's FP32 output cast."""
    return mcore_layers.linear_with_grad_accumulation_and_async_allreduce(
        input_,
        weight,
        bias,
        False,
        False,
        False,
        None,
        0,
        None,
    ).float()


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    return (difference.norm() / expected.float().norm()).item()


def _cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return F.cosine_similarity(
        actual.float().flatten(),
        expected.float().flatten(),
        dim=0,
    ).item()


def test_replace_output_layer_preserves_module_parameter_and_state_key():
    """Promotion must not invalidate DDP hooks or checkpoint mappings."""
    head = _make_uninitialized_head()
    model = _ModelWithOutputLayer(head)
    module_id = id(head)
    parameter_id = id(head.weight)
    hook = head.weight.register_hook(lambda grad: grad)

    replace_output_layer_with_areal_lm_head(model, fp32_output=True)

    assert isinstance(model.output_layer, AReaLVocabParallelLMHead)
    assert id(model.output_layer) == module_id
    assert id(model.output_layer.weight) == parameter_id
    assert model.output_layer.fp32_output is True
    assert "output_layer.weight" in model.state_dict()
    assert hook.id in model.output_layer.weight._backward_hooks

    try:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
    except ImportError:
        return
    assert (
        AReaLVocabParallelLMHead.__name__ in AutoMapping._MODULE_TYPE_REGISTRY["column"]
    )


def test_replace_output_layer_preserves_bridge_subclass_forward():
    class BridgeOutputLayer(mcore_layers.ColumnParallelLinear):
        def forward(self):
            return "bridge-forward"

    head = _make_uninitialized_head()
    head.__class__ = BridgeOutputLayer
    model = _ModelWithOutputLayer(head)

    replace_output_layer_with_areal_lm_head(model, fp32_output=False)

    assert model.output_layer() == "bridge-forward"
    assert BridgeOutputLayer in type(model.output_layer).__mro__
    assert model.output_layer.fp32_output is False


def test_float16_module_default_output_remains_fp32(monkeypatch):
    """The BF16 LM Head mode must retain Megatron's default output cast."""
    from megatron.core.pipeline_parallel import utils as pp_utils
    from megatron.core.transformer import module as module_lib

    monkeypatch.setattr(
        module_lib.parallel_state,
        "get_pipeline_model_parallel_group",
        lambda: None,
    )
    monkeypatch.setattr(pp_utils, "is_pp_first_stage", lambda _group: True)
    monkeypatch.setattr(pp_utils, "is_pp_last_stage", lambda _group: True)
    monkeypatch.setattr(pp_utils, "is_vp_first_stage", lambda _stage, _size: True)
    monkeypatch.setattr(pp_utils, "is_vp_last_stage", lambda _stage, _size: True)

    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        bf16=True,
    )
    model = Float16Module(config, _BF16OutputModule())

    assert model().dtype == torch.float32
    assert model(fp32_output=False).dtype == torch.bfloat16


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for direct FP32 mm")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp32_output", [False, True])
def test_lm_head_output_and_gradients_match_reference(
    fp32_output: bool,
    dtype: torch.dtype,
):
    """Both output modes must match their GEMM and Megatron backward contracts."""
    torch.manual_seed(1234)
    device = torch.device("cuda")
    input_ = torch.randn(5, 2, 16, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(32, 16, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(5, 2, 32, dtype=torch.float32, device=device)

    head = _make_uninitialized_head()
    model = _ModelWithOutputLayer(head)
    replace_output_layer_with_areal_lm_head(
        model,
        fp32_output=fp32_output,
    )

    output = model.output_layer._forward_impl(
        input_,
        weight,
        None,
        False,
        False,
        False,
        None,
        0,
        None,
    )
    if fp32_output:
        expected_output = torch.mm(
            input_.detach().reshape(-1, 16),
            weight.detach().t(),
            out_dtype=torch.float32,
        ).view_as(output)
        backward_grad = grad_output.to(dtype)
    else:
        expected_output = torch.matmul(input_.detach(), weight.detach().t())
        backward_grad = grad_output.to(dtype)

    assert output.dtype == (torch.float32 if fp32_output else dtype)
    assert type(output.grad_fn).__name__ == (
        "_LinearWithFp32OutputBackward"
        if fp32_output
        else "_LinearWithNativeOutputBackward"
    )
    torch.testing.assert_close(output, expected_output, rtol=0.0, atol=0.0)
    if fp32_output:
        assert reusable_vocab_parallel_logits(output) is output

    output.backward(grad_output.to(output.dtype))
    expected_input_grad = backward_grad.matmul(weight.detach())
    expected_weight_grad = (
        backward_grad.reshape(-1, 32).t().matmul(input_.detach().reshape(-1, 16))
    )
    torch.testing.assert_close(input_.grad, expected_input_grad, rtol=0.0, atol=0.0)
    torch.testing.assert_close(weight.grad, expected_weight_grad, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for direct FP32 mm")
@pytest.mark.parametrize("gradient_accumulation_fusion", [False, True])
@pytest.mark.parametrize("fp32_output", [False, True])
def test_frozen_lm_head_skips_wgrad_and_matches_dgrad(
    gradient_accumulation_fusion: bool,
    fp32_output: bool,
):
    torch.manual_seed(4321)
    input_ = torch.randn(
        5, 2, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    weight = torch.randn(32, 16, dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn(5, 2, 32, dtype=torch.float32, device="cuda")
    head = _make_uninitialized_head()
    model = _ModelWithOutputLayer(head)
    replace_output_layer_with_areal_lm_head(model, fp32_output=fp32_output)

    output = model.output_layer._forward_impl(
        input_,
        weight,
        None,
        gradient_accumulation_fusion,
        False,
        False,
        None,
        None,
        None,
    )
    expected_output = (
        torch.mm(
            input_.detach().reshape(-1, 16),
            weight.t(),
            out_dtype=torch.float32,
        ).view_as(output)
        if fp32_output
        else torch.matmul(input_.detach(), weight.t())
    )
    assert type(output.grad_fn).__name__ == (
        "_LinearWithFrozenFp32OutputBackward"
        if fp32_output
        else "_LinearWithFrozenNativeOutputBackward"
    )
    torch.testing.assert_close(output, expected_output, rtol=0.0, atol=0.0)

    output.backward(grad_output.to(output.dtype))

    expected_input_grad = grad_output.to(torch.bfloat16).matmul(weight)
    torch.testing.assert_close(input_.grad, expected_input_grad, rtol=0.0, atol=0.0)
    assert weight.grad is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for in-place packing")
@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 17), (7, 33), (17, 1537)])
def test_pack_fp32_to_half_inplace_preserves_values_and_storage(
    target_dtype: torch.dtype,
    shape: tuple[int, int],
):
    torch.manual_seed(20260720)
    source = torch.randn(shape, dtype=torch.float32, device="cuda")
    expected = source.to(dtype=target_dtype)
    storage_ptr = source.data_ptr()
    storage_size = source.untyped_storage().nbytes()
    source_version = source._version

    packed = _pack_fp32_to_half_inplace(
        source,
        target_dtype=target_dtype,
        seed_elements=shape[-1],
    )

    assert packed.dtype is target_dtype
    assert packed.shape == source.shape
    assert packed.data_ptr() == storage_ptr
    assert packed.untyped_storage().nbytes() == storage_size
    assert source._version > source_version
    torch.testing.assert_close(packed, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for storage reuse")
def test_storage_identity_rejects_recycled_cuda_address():
    torch.cuda.empty_cache()
    shape = (1024, 1024)
    source = torch.empty(shape, dtype=torch.float32, device="cuda")
    source_ptr = source.data_ptr()
    storage_ref = weakref.ref(source.untyped_storage())
    del source
    gc.collect()
    assert storage_ref() is None

    replacement = torch.empty(shape, dtype=torch.float32, device="cuda")

    assert replacement.data_ptr() == source_ptr
    assert not _matches_storage(replacement, storage_ref)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for direct FP32 mm")
@pytest.mark.parametrize("use_bias", [False, True])
def test_direct_fp32_backward_exactly_matches_native_cast_contract(
    use_bias: bool,
    monkeypatch,
):
    """Both paths must pass the same BF16 dlogits into MCore backward."""
    monkeypatch.setattr(
        lm_head_module,
        "_pack_fp32_to_half_inplace",
        lambda *_args, **_kwargs: pytest.fail(
            "external grad_output must use the non-destructive fallback"
        ),
    )
    torch.manual_seed(20260720)
    base_input = torch.randn(7, 2, 32, dtype=torch.bfloat16, device="cuda")
    base_weight = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    base_bias = torch.randn(64, dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn(7, 2, 64, dtype=torch.float32, device="cuda")

    native_input = base_input.clone().requires_grad_()
    native_weight = base_weight.clone().requires_grad_()
    native_bias = base_bias.clone().requires_grad_() if use_bias else None
    direct_input = base_input.clone().requires_grad_()
    direct_weight = base_weight.clone().requires_grad_()
    direct_bias = base_bias.clone().requires_grad_() if use_bias else None

    native_logits = _native_megatron_areal_logits(
        native_input,
        native_weight,
        native_bias,
    )
    direct_logits = linear_with_fp32_output(
        direct_input,
        direct_weight,
        direct_bias,
        False,
        False,
        False,
        None,
        0,
        None,
    )
    native_logits.backward(grad_output)
    direct_logits.backward(grad_output)

    torch.testing.assert_close(direct_input.grad, native_input.grad, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        direct_weight.grad, native_weight.grad, rtol=0.0, atol=0.0
    )
    if use_bias:
        torch.testing.assert_close(
            direct_bias.grad, native_bias.grad, rtol=0.0, atol=0.0
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for direct FP32 mm")
def test_direct_fp32_end_to_end_precision_matches_native_megatron_areal():
    """Compare logits, CE, and gradients with the original production path."""
    torch.manual_seed(20260720)
    sequence_length, batch_size, hidden_size, vocab_size = 17, 3, 128, 512
    base_input = torch.randn(
        sequence_length,
        batch_size,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    base_weight = 0.02 * torch.randn(
        vocab_size,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    labels = torch.randint(
        vocab_size,
        (sequence_length, batch_size),
        device="cuda",
    )

    native_input = base_input.clone().requires_grad_()
    native_weight = base_weight.clone().requires_grad_()
    direct_input = base_input.clone().requires_grad_()
    direct_weight = base_weight.clone().requires_grad_()

    native_logits = _native_megatron_areal_logits(native_input, native_weight)
    direct_logits = linear_with_fp32_output(
        direct_input,
        direct_weight,
        None,
        False,
        False,
        False,
        None,
        0,
        None,
    )
    fp32_reference_logits = base_input.float().matmul(base_weight.float().t())

    assert native_logits.dtype == torch.float32
    assert direct_logits.dtype == torch.float32
    assert _relative_l2(direct_logits, native_logits) <= 2.0e-3
    assert _cosine_similarity(direct_logits, native_logits) >= 0.99999
    assert _relative_l2(direct_logits, fp32_reference_logits) <= _relative_l2(
        native_logits, fp32_reference_logits
    )

    flat_labels = labels.flatten()
    native_token_loss = F.cross_entropy(
        native_logits.flatten(0, -2), flat_labels, reduction="none"
    )
    direct_token_loss = F.cross_entropy(
        direct_logits.flatten(0, -2), flat_labels, reduction="none"
    )
    torch.testing.assert_close(
        direct_token_loss,
        native_token_loss,
        rtol=2.0e-3,
        atol=1.0e-2,
    )

    native_loss = native_token_loss.mean()
    direct_loss = direct_token_loss.mean()
    torch.testing.assert_close(direct_loss, native_loss, rtol=1.0e-3, atol=2.0e-3)
    native_loss.backward()
    direct_loss.backward()

    assert _relative_l2(direct_input.grad, native_input.grad) <= 5.0e-3
    assert _cosine_similarity(direct_input.grad, native_input.grad) >= 0.99995
    assert _relative_l2(direct_weight.grad, native_weight.grad) <= 1.0e-2
    assert _cosine_similarity(direct_weight.grad, native_weight.grad) >= 0.9999


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for chunked LM Head")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_chunked_lm_head_matches_full_fused_path(
    dtype: torch.dtype,
    temperature: float,
):
    torch.manual_seed(20260721)
    sequence_length, hidden_size, vocab_size = 13, 32, 96
    logit_scale = 0.5
    base_input = torch.randn(
        sequence_length, 1, hidden_size, dtype=dtype, device="cuda"
    )
    base_weight = torch.randn(vocab_size, hidden_size, dtype=dtype, device="cuda")
    labels = torch.randint(vocab_size, (sequence_length, 1), device="cuda")
    logprob_weights = torch.randn(sequence_length, 1, device="cuda")

    chunked_input = base_input.clone().requires_grad_()
    chunked_weight = base_weight.clone().requires_grad_()
    outputs = _ChunkedVocabParallelLMHead.apply(
        chunked_input,
        chunked_weight,
        None,
        labels,
        temperature,
        5,
        False,
        False,
        False,
        None,
        logit_scale,
    )

    reference_input = base_input.clone().requires_grad_()
    reference_weight = base_weight.clone().requires_grad_()
    reference_logits = linear_with_fp32_output(
        reference_input,
        reference_weight,
        None,
        False,
        False,
        False,
        tp_group=None,
    )
    reference_logits.mul_(logit_scale)
    expected_stats = (
        reference_logits.detach().min(dim=-1).values.reshape(-1),
        reference_logits.detach().max(dim=-1).values.reshape(-1),
        reference_logits.detach().mean(dim=-1).reshape(-1),
        torch.linalg.vector_norm(reference_logits.detach(), dim=-1).reshape(-1),
    )
    reference_logprobs, reference_entropy = gather_logprobs_entropy(
        reference_logits,
        labels,
        temperature=temperature,
        reuse_logits=True,
        chunk_size=5,
    )

    torch.testing.assert_close(
        outputs[0], reference_logprobs.reshape(-1), rtol=1e-5, atol=2e-5
    )
    torch.testing.assert_close(
        outputs[1], reference_entropy.reshape(-1), rtol=1e-5, atol=2e-5
    )
    for actual, expected in zip(outputs[2:], expected_stats, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
        assert not actual.requires_grad
    assert not outputs[1].requires_grad
    assert all(
        tensor.shape != (sequence_length, vocab_size)
        for tensor in outputs[0].grad_fn.saved_tensors
    )

    (outputs[0] * logprob_weights.reshape(-1)).sum().backward()
    (reference_logprobs * logprob_weights).sum().backward()
    torch.testing.assert_close(
        chunked_input.grad, reference_input.grad, rtol=1e-5, atol=2e-5
    )
    assert _relative_l2(chunked_weight.grad, reference_weight.grad) <= 5e-3
    assert _cosine_similarity(chunked_weight.grad, reference_weight.grad) >= 0.99999


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for chunked LM Head")
@pytest.mark.skipif(
    not FUSED_WGRAD_AVAILABLE,
    reason="Apex fused weight-gradient extension is required",
)
def test_chunked_lm_head_accumulates_fp32_main_grad():
    torch.manual_seed(20260721)
    input_ = torch.randn(
        11, 1, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    weight = torch.nn.Parameter(
        torch.randn(48, 16, dtype=torch.bfloat16, device="cuda")
    )
    weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
    weight.grad_added_to_main_grad = False
    weight.zero_out_wgrad = True
    labels = torch.randint(48, (11, 1), device="cuda")

    outputs = _ChunkedVocabParallelLMHead.apply(
        input_,
        weight,
        None,
        labels,
        1.0,
        4,
        True,
        False,
        False,
        None,
        1.0,
    )
    outputs[0].sum().backward()

    assert input_.grad is not None
    assert weight.main_grad.norm() > 0
    assert weight.grad_added_to_main_grad
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight.grad))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for direct FP32 mm")
def test_direct_fp32_lm_head_reuses_logits_through_loss_backward(monkeypatch):
    """The direct GEMM owner must remain reusable by the fused loss autograd node."""
    pack_calls: list[tuple[int, int]] = []
    original_pack = lm_head_module._pack_fp32_to_half_inplace

    def tracking_pack(
        tensor: torch.Tensor,
        target_dtype: torch.dtype,
        seed_elements: int,
    ) -> torch.Tensor:
        packed = original_pack(tensor, target_dtype, seed_elements)
        pack_calls.append((tensor.data_ptr(), packed.data_ptr()))
        return packed

    monkeypatch.setattr(
        lm_head_module,
        "_pack_fp32_to_half_inplace",
        tracking_pack,
    )
    torch.manual_seed(5678)
    input_ = torch.randn(
        11, 1, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    weight = torch.randn(
        32, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    labels = torch.randint(0, 32, (11, 1), device="cuda")
    loss_weights = torch.randn(11, 1, device="cuda")

    logits = linear_with_fp32_output(
        input_, weight, None, False, False, False, tp_group=None
    )
    reference_logits = logits.detach().clone().requires_grad_(True)
    reference_logprobs = (
        torch.log_softmax(reference_logits, dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )
    (reference_logprobs * loss_weights).sum().backward()

    owner = reusable_vocab_parallel_logits(logits)
    assert owner is logits
    storage_ptr = owner.data_ptr()
    logprobs, entropy = gather_logprobs_entropy(
        logits,
        labels,
        reuse_logits=True,
        chunk_size=4,
    )
    torch.testing.assert_close(logprobs, reference_logprobs, rtol=1e-5, atol=2e-5)
    assert not entropy.requires_grad
    assert owner.data_ptr() == storage_ptr

    (logprobs * loss_weights).sum().backward()
    assert pack_calls == [(storage_ptr, storage_ptr)]
    expected_dlogits = reference_logits.grad.to(torch.bfloat16)
    expected_input_grad = expected_dlogits.matmul(weight.detach())
    expected_weight_grad = (
        expected_dlogits.reshape(-1, 32).t().matmul(input_.detach().reshape(-1, 16))
    )
    torch.testing.assert_close(input_.grad, expected_input_grad, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(weight.grad, expected_weight_grad, rtol=1e-5, atol=2e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for direct FP32 mm")
def test_direct_fp32_lm_head_does_not_pack_reused_allocator_address(monkeypatch):
    """A recycled CUDA address is not proof that dlogits owns the logits storage."""

    class _ReleaseLogits(torch.autograd.Function):
        grad_ptr: int | None = None

        @staticmethod
        def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
            ctx.shape = tensor.shape
            ctx.device = tensor.device
            return tensor.sum()

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
            del grad_output
            grad = torch.full(ctx.shape, 2.0, dtype=torch.float32, device=ctx.device)
            _ReleaseLogits.grad_ptr = grad.data_ptr()
            return (grad,)

    monkeypatch.setattr(
        lm_head_module,
        "_pack_fp32_to_half_inplace",
        lambda *_args, **_kwargs: pytest.fail(
            "allocator address reuse must use the non-destructive fallback"
        ),
    )
    torch.manual_seed(20260720)
    input_ = torch.randn(
        17, 1, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    weight = torch.randn(
        32, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    logits = linear_with_fp32_output(
        input_, weight, None, False, False, False, tp_group=None
    )
    storage_ref = weakref.ref(logits.untyped_storage())
    loss = _ReleaseLogits.apply(logits)
    del logits
    gc.collect()
    assert storage_ref() is None

    loss.backward()

    assert _ReleaseLogits.grad_ptr is not None
    expected_dlogits = torch.full((17, 1, 32), 2.0, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(
        input_.grad, expected_dlogits.matmul(weight.detach()), rtol=0.0, atol=0.0
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for direct FP32 mm")
def test_fused_loss_preserves_public_logits_hook_and_retain_grad(monkeypatch):
    """The destructive loss must keep the public LM Head autograd edge."""
    pack_calls = 0
    original_pack = lm_head_module._pack_fp32_to_half_inplace

    def tracking_pack(*args, **kwargs):
        nonlocal pack_calls
        pack_calls += 1
        return original_pack(*args, **kwargs)

    monkeypatch.setattr(lm_head_module, "_pack_fp32_to_half_inplace", tracking_pack)
    torch.manual_seed(20260720)
    input_ = torch.randn(
        13, 1, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    weight = torch.randn(
        32, 16, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    labels = torch.randint(0, 32, (13, 1), device="cuda")
    loss_weights = torch.randn(13, 1, device="cuda")
    logits = linear_with_fp32_output(
        input_, weight, None, False, False, False, tp_group=None
    )
    reference_logits = logits.detach().clone().requires_grad_(True)
    reference_logprobs = (
        torch.log_softmax(reference_logits, dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )
    (reference_logprobs * loss_weights).sum().backward()

    hook_grads: list[torch.Tensor] = []
    hook_scale = 0.5
    logits.retain_grad()
    logits.register_hook(
        lambda grad: (hook_grads.append(grad.detach().clone()), grad * hook_scale)[1]
    )
    logprobs, _ = gather_logprobs_entropy(
        logits,
        labels,
        reuse_logits=True,
        chunk_size=4,
    )
    (logprobs * loss_weights).sum().backward()

    assert len(hook_grads) == 1
    torch.testing.assert_close(
        hook_grads[0], reference_logits.grad, rtol=1e-5, atol=2e-5
    )
    torch.testing.assert_close(
        logits.grad, reference_logits.grad * hook_scale, rtol=1e-5, atol=2e-5
    )
    expected_dlogits = (reference_logits.grad * hook_scale).to(torch.bfloat16)
    torch.testing.assert_close(
        input_.grad,
        expected_dlogits.matmul(weight.detach()),
        rtol=1e-5,
        atol=2e-5,
    )
    assert pack_calls == 0
