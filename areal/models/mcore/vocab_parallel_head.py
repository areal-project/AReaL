# SPDX-License-Identifier: Apache-2.0

import math
import weakref
from typing import NamedTuple

import torch
import torch.distributed as dist
from megatron.core.tensor_parallel import layers as mcore_layers


def _matches_storage(tensor: torch.Tensor, storage_ref: weakref.ReferenceType) -> bool:
    expected_storage = storage_ref()
    return (
        expected_storage is not None
        and tensor.untyped_storage()._cdata == expected_storage._cdata
    )


def _pack_fp32_to_half_inplace(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    seed_elements: int,
) -> torch.Tensor:
    """Pack FP32 values into the first half of their storage as BF16/FP16."""
    if (
        not tensor.is_cuda
        or tensor.dtype is not torch.float32
        or target_dtype not in (torch.bfloat16, torch.float16)
        or not tensor.is_contiguous()
        or tensor.storage_offset() != 0
    ):
        raise ValueError("in-place packing requires contiguous CUDA FP32 at offset 0")
    if seed_elements <= 0:
        raise ValueError(f"seed_elements must be positive, got {seed_elements}")

    numel = tensor.numel()
    # dtype views share the source version counter. This matters because packing
    # destructively changes the FP32 tensor's storage during backward.
    packed = tensor.view(-1).view(target_dtype)[:numel]
    if numel == 0:
        return packed.view(tensor.shape)

    source = tensor.view(-1)
    prefix_end = min(seed_elements, numel)
    scratch = source[:prefix_end].to(dtype=target_dtype)
    packed[:prefix_end].copy_(scratch)
    del scratch

    # FP32 source [s, e) occupies bytes [4s, 4e), while its packed half output
    # occupies [2s, 2e). Choosing e <= 2s makes the ranges disjoint. Doubling
    # each processed prefix therefore packs the rest in logarithmically many
    # cast-copy launches without another full-sized tensor.
    start = prefix_end
    while start < numel:
        end = min(2 * start, numel)
        packed[start:end].copy_(source[start:end])
        start = end

    return packed.view(tensor.shape)


def _prepare_linear_forward(
    ctx,
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[torch.Tensor] | None,
    wgrad_deferral_limit: int | None,
    tp_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
        main_grad = weight.main_grad
    else:
        main_grad = None

    ctx.save_for_backward(input_, weight)
    ctx.main_grad = main_grad
    ctx.use_bias = bias is not None
    ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
    ctx.allreduce_dgrad = allreduce_dgrad
    ctx.sequence_parallel = sequence_parallel
    ctx.wgrad_deferral_limit = wgrad_deferral_limit
    ctx.grad_output_buffer = grad_output_buffer
    ctx.tp_group = tp_group

    if not sequence_parallel:
        return input_

    dim_size = list(input_.size())
    dim_size[0] *= tp_group.size()
    all_gather_buffer = mcore_layers.get_global_memory_buffer().get_tensor(
        dim_size, input_.dtype, "mpu"
    )
    mcore_layers.dist_all_gather_func(all_gather_buffer, input_, group=tp_group)
    return all_gather_buffer


class _LinearWithNativeOutput(
    mcore_layers.LinearWithGradAccumulationAndAsyncCommunication
):
    """AReaL TP linear with the native input/weight output dtype."""

    @staticmethod
    @mcore_layers.custom_fwd
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        gradient_accumulation_fusion: bool,
        allreduce_dgrad: bool,
        sequence_parallel: bool,
        grad_output_buffer: list[torch.Tensor] | None,
        wgrad_deferral_limit: int | None,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        total_input = _prepare_linear_forward(
            ctx,
            input_,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
            tp_group,
        )
        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output


class _LinearWithFrozenNativeOutput(mcore_layers.LinearWithFrozenWeight):
    """AReaL TP linear that skips wgrad for a frozen weight."""

    @staticmethod
    @mcore_layers.custom_fwd
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        allreduce_dgrad: bool,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight)
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        output = torch.matmul(input_, weight.t())
        if bias is not None:
            output = output + bias
        return output


class _LinearWithFp32Output(
    mcore_layers.LinearWithGradAccumulationAndAsyncCommunication
):
    """Megatron TP linear that writes BF16/FP16 GEMM results directly as FP32."""

    @staticmethod
    @mcore_layers.custom_fwd
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        gradient_accumulation_fusion: bool,
        allreduce_dgrad: bool,
        sequence_parallel: bool,
        grad_output_buffer: list[torch.Tensor] | None,
        wgrad_deferral_limit: int | None,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        total_input = _prepare_linear_forward(
            ctx,
            input_,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
            tp_group,
        )

        input_2d = total_input.reshape(-1, total_input.size(-1))
        ctx.output_shape = (*total_input.shape[:-1], weight.size(0))
        output = torch.empty(
            ctx.output_shape,
            dtype=torch.float32,
            device=input_.device,
        )
        torch.mm(
            input_2d,
            weight.t(),
            out=output.view(-1, weight.size(0)),
            out_dtype=torch.float32,
        )
        ctx.output_storage_ref = weakref.ref(output.untyped_storage())
        ctx.output_numel = output.numel()
        if bias is not None:
            output.add_(bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # The original BF16-output path receives this cast from ToCopyBackward.
        # Preserve that backward contract while removing the forward BF16 tensor.
        _, weight = ctx.saved_tensors
        can_pack_inplace = (
            grad_output.is_cuda
            and grad_output.dtype is torch.float32
            and grad_output.is_contiguous()
            and grad_output.storage_offset() == 0
            and _matches_storage(grad_output, ctx.output_storage_ref)
            and grad_output.numel() == ctx.output_numel
            and weight.dtype in (torch.bfloat16, torch.float16)
        )
        if can_pack_inplace:
            grad_output = _pack_fp32_to_half_inplace(
                grad_output,
                target_dtype=weight.dtype,
                seed_elements=grad_output.size(-1),
            ).view(ctx.output_shape)
        else:
            grad_output = grad_output.view(ctx.output_shape).to(dtype=weight.dtype)
        return mcore_layers.LinearWithGradAccumulationAndAsyncCommunication.backward(
            ctx, grad_output
        )


class _LinearWithFrozenFp32Output(mcore_layers.LinearWithFrozenWeight):
    """AReaL frozen-weight TP linear with direct FP32 GEMM output."""

    @staticmethod
    @mcore_layers.custom_fwd
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        allreduce_dgrad: bool,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight)
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        ctx.output_shape = (*input_.shape[:-1], weight.size(0))
        output = torch.empty(
            ctx.output_shape,
            dtype=torch.float32,
            device=input_.device,
        )
        torch.mm(
            input_.reshape(-1, input_.size(-1)),
            weight.t(),
            out=output.view(-1, weight.size(0)),
            out_dtype=torch.float32,
        )
        ctx.output_storage_ref = weakref.ref(output.untyped_storage())
        ctx.output_numel = output.numel()
        if bias is not None:
            output.add_(bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        (weight,) = ctx.saved_tensors
        can_pack_inplace = (
            grad_output.is_cuda
            and grad_output.dtype is torch.float32
            and grad_output.is_contiguous()
            and grad_output.storage_offset() == 0
            and _matches_storage(grad_output, ctx.output_storage_ref)
            and grad_output.numel() == ctx.output_numel
            and weight.dtype in (torch.bfloat16, torch.float16)
        )
        if can_pack_inplace:
            grad_output = _pack_fp32_to_half_inplace(
                grad_output,
                target_dtype=weight.dtype,
                seed_elements=grad_output.size(-1),
            ).view(ctx.output_shape)
        else:
            grad_output = grad_output.view(ctx.output_shape).to(dtype=weight.dtype)

        grad_input = grad_output.matmul(weight)
        if ctx.allreduce_dgrad:
            torch.distributed.all_reduce(grad_input, group=ctx.tp_group)
        return grad_input, None, None, None, None


def linear_with_fp32_output(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[torch.Tensor] | None = None,
    wgrad_deferral_limit: int | None = 0,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Run Megatron's TP linear with a direct FP32 CUDA GEMM output."""
    output = _LinearWithFp32Output.apply(
        input_,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
    )
    return output


def _linear_with_frozen_areal_output(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[torch.Tensor] | None,
    wgrad_deferral_limit: int | None,
    tp_group: torch.distributed.ProcessGroup | None,
    *,
    fp32_output: bool,
) -> torch.Tensor:
    del gradient_accumulation_fusion
    if grad_output_buffer is not None:
        raise AssertionError(
            "grad_output_buffer is only supported for trainable LM Head weights"
        )
    if wgrad_deferral_limit is not None:
        raise AssertionError(
            "wgrad_deferral_limit is only supported for trainable LM Head weights"
        )

    tp_group = mcore_layers.get_tensor_model_parallel_group_if_none(tp_group)
    if sequence_parallel:
        input_ = mcore_layers.gather_from_sequence_parallel_region(
            input_,
            tensor_parallel_output_grad=True,
            group=tp_group,
        )

    can_use_direct_fp32 = (
        fp32_output
        and input_.is_cuda
        and input_.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == input_.dtype
    )
    function = (
        _LinearWithFrozenFp32Output
        if can_use_direct_fp32
        else _LinearWithFrozenNativeOutput
    )
    output = function.apply(input_, weight, bias, allreduce_dgrad, tp_group)
    return output.float() if fp32_output and not can_use_direct_fp32 else output


def linear_with_areal_output(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[torch.Tensor] | None = None,
    wgrad_deferral_limit: int | None = 0,
    tp_group: torch.distributed.ProcessGroup | None = None,
    *,
    fp32_output: bool,
) -> torch.Tensor:
    """Run the AReaL TP linear with native or direct FP32 output."""
    if not weight.requires_grad:
        return _linear_with_frozen_areal_output(
            input_,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
            tp_group,
            fp32_output=fp32_output,
        )

    can_use_direct_fp32 = (
        fp32_output
        and input_.is_cuda
        and input_.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == input_.dtype
    )
    if can_use_direct_fp32:
        return linear_with_fp32_output(
            input_,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
            tp_group,
        )

    output = _LinearWithNativeOutput.apply(
        input_,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
    )
    return output.float() if fp32_output else output


class ChunkedLMHeadOutput(NamedTuple):
    """Per-token outputs produced without materializing full-sequence logits."""

    logprobs: torch.Tensor
    entropy: torch.Tensor
    vocab_min_logits: torch.Tensor
    vocab_max_logits: torch.Tensor
    vocab_mean_logits: torch.Tensor
    vocab_norm_logits: torch.Tensor


def _tp_world_size(tp_group: dist.ProcessGroup | None) -> int:
    if tp_group is None or not dist.is_initialized():
        return 1
    return dist.get_world_size(tp_group)


def _all_reduce_if_needed(
    tensor: torch.Tensor,
    op: dist.ReduceOp.RedOpType,
    tp_group: dist.ProcessGroup | None,
) -> None:
    if _tp_world_size(tp_group) > 1:
        dist.all_reduce(tensor, op=op, group=tp_group)


def _gather_sequence_parallel_input(
    input_: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    if tp_group is None:
        raise ValueError("sequence-parallel LM Head requires a TP process group")
    shape = list(input_.shape)
    shape[0] *= dist.get_world_size(tp_group)
    output = mcore_layers.get_global_memory_buffer().get_tensor(
        shape, input_.dtype, "mpu"
    )
    mcore_layers.dist_all_gather_func(output, input_, group=tp_group)
    return output


def _linear_chunk_fp32(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    logit_scale: float,
) -> torch.Tensor:
    output = torch.empty(
        (input_.size(0), weight.size(0)),
        dtype=torch.float32,
        device=input_.device,
    )
    if input_.dtype in (torch.bfloat16, torch.float16) and weight.dtype == input_.dtype:
        torch.mm(input_, weight.t(), out=output, out_dtype=torch.float32)
    else:
        torch.mm(input_, weight.t(), out=output)
    if bias is not None:
        output.add_(bias)
    if logit_scale != 1.0:
        output.mul_(logit_scale)
    return output


class _ChunkedVocabParallelLMHead(torch.autograd.Function):
    """Recompute chunk logits in backward to bound vocab activation memory."""

    @staticmethod
    @mcore_layers.custom_fwd
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        labels: torch.Tensor,
        temperature: float,
        chunk_size: int,
        gradient_accumulation_fusion: bool,
        allreduce_dgrad: bool,
        sequence_parallel: bool,
        tp_group: dist.ProcessGroup | None,
        logit_scale: float,
    ) -> tuple[torch.Tensor, ...]:
        from areal.utils.functional.vocab_parallel_kernels import (
            fused_exp_sum_inplace,
            vocab_tile_count,
        )

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not math.isfinite(logit_scale) or logit_scale <= 0:
            raise ValueError(f"logit_scale must be positive, got {logit_scale}")
        if input_.dtype != weight.dtype:
            raise ValueError(
                "chunked LM Head requires matching hidden and weight dtypes, got "
                f"{input_.dtype} and {weight.dtype}"
            )

        total_input = (
            _gather_sequence_parallel_input(input_, tp_group)
            if sequence_parallel
            else input_
        )
        input_2d = total_input.reshape(-1, total_input.size(-1))
        labels_1d = labels.reshape(-1)
        if input_2d.size(0) != labels_1d.numel():
            raise ValueError(
                "LM Head hidden leading dimensions must match labels: "
                f"got {tuple(total_input.shape)} and {tuple(labels.shape)}"
            )

        partition_vocab_size = weight.size(0)
        tp_rank = dist.get_rank(tp_group) if _tp_world_size(tp_group) > 1 else 0
        local_targets = labels_1d - tp_rank * partition_vocab_size
        local_targets.masked_fill_(
            (local_targets < 0) | (local_targets >= partition_vocab_size), -1
        )

        num_tokens = input_2d.size(0)
        output_tensors = [
            torch.empty(num_tokens, dtype=torch.float32, device=input_.device)
            for _ in range(6)
        ]
        logprobs, entropy, vocab_min, vocab_max, vocab_mean, vocab_norm = output_tensors
        row_maxes = torch.empty_like(logprobs)
        sum_exps = torch.empty_like(logprobs)
        workspace_rows = min(chunk_size, num_tokens)
        partial_reduction = torch.empty(
            (2, workspace_rows, vocab_tile_count(partition_vocab_size)),
            dtype=torch.float32,
            device=input_.device,
        )
        reduced_values = torch.empty(
            (3, workspace_rows), dtype=torch.float32, device=input_.device
        )
        inv_temperature = 1.0 / temperature

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            rows = end - start
            logits = _linear_chunk_fp32(input_2d[start:end], weight, bias, logit_scale)
            chunk_targets = local_targets[start:end]

            vocab_min[start:end] = logits.min(dim=-1).values
            vocab_max[start:end] = logits.max(dim=-1).values
            vocab_mean[start:end] = logits.mean(dim=-1)
            vocab_norm[start:end] = torch.linalg.vector_norm(logits, dim=-1)

            logits_max = logits.max(dim=-1).values
            _all_reduce_if_needed(logits_max, dist.ReduceOp.MAX, tp_group)
            row_maxes[start:end] = logits_max

            row_indices = torch.arange(rows, device=input_.device)
            predicted_logits = logits[
                row_indices, chunk_targets.clamp_min(0)
            ].masked_fill_(chunk_targets < 0, 0.0)
            predicted_logits.sub_(logits_max).mul_(inv_temperature)
            predicted_logits.masked_fill_(chunk_targets < 0, 0.0)

            partial_sums = partial_reduction[0, :rows]
            partial_weighted_sums = partial_reduction[1, :rows]
            fused_exp_sum_inplace(
                logits,
                logits_max,
                partial_sums,
                partial_weighted_sums,
                inv_temperature,
            )
            reduced = reduced_values[:, :rows]
            reduced[0].copy_(predicted_logits)
            reduced[1].copy_(partial_sums.sum(dim=-1))
            reduced[2].copy_(partial_weighted_sums.sum(dim=-1))
            if rows < workspace_rows:
                reduced_values[:, rows:].zero_()
            _all_reduce_if_needed(reduced_values, dist.ReduceOp.SUM, tp_group)

            sum_exp = reduced[1]
            sum_exps[start:end] = sum_exp
            log_sum_exp = sum_exp.log()
            logprobs[start:end] = reduced[0] - log_sum_exp
            entropy[start:end] = log_sum_exp - reduced[2] / sum_exp

        saved_bias = (
            bias
            if bias is not None
            else torch.empty(0, dtype=weight.dtype, device=weight.device)
        )
        ctx.save_for_backward(
            input_, weight, saved_bias, local_targets, row_maxes, sum_exps
        )
        ctx.has_bias = bias is not None
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.use_main_grad = (
            gradient_accumulation_fusion
            and weight.requires_grad
            and hasattr(weight, "main_grad")
            and not hasattr(weight, "__fsdp_param__")
        )
        ctx.main_grad = weight.main_grad if ctx.use_main_grad else None
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.tp_group = tp_group
        ctx.logit_scale = logit_scale
        ctx.input_shape = input_.shape
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(
            entropy, vocab_min, vocab_max, vocab_mean, vocab_norm
        )
        return tuple(output_tensors)

    @staticmethod
    @mcore_layers.custom_bwd
    def backward(
        ctx,
        grad_logprobs: torch.Tensor | None,
        grad_entropy: torch.Tensor | None,
        grad_vocab_min: torch.Tensor | None,
        grad_vocab_max: torch.Tensor | None,
        grad_vocab_mean: torch.Tensor | None,
        grad_vocab_norm: torch.Tensor | None,
    ) -> tuple:
        from areal.utils.functional.vocab_parallel_kernels import (
            fused_logits_backward_inplace,
        )

        del (
            grad_entropy,
            grad_vocab_min,
            grad_vocab_max,
            grad_vocab_mean,
            grad_vocab_norm,
        )
        input_, weight, saved_bias, local_targets, row_maxes, sum_exps = (
            ctx.saved_tensors
        )
        bias = saved_bias if ctx.has_bias else None
        total_input = (
            _gather_sequence_parallel_input(input_, ctx.tp_group)
            if ctx.sequence_parallel
            else input_
        )
        input_2d = total_input.reshape(-1, total_input.size(-1))
        grad_input_2d = torch.empty_like(input_2d)
        grad_logprobs_1d = (
            grad_logprobs.reshape(-1).contiguous()
            if grad_logprobs is not None
            else None
        )

        needs_weight_grad = ctx.needs_input_grad[1]
        grad_weight = (
            torch.zeros_like(weight)
            if needs_weight_grad and not ctx.use_main_grad
            else None
        )
        grad_bias = (
            torch.zeros_like(bias)
            if bias is not None and ctx.needs_input_grad[2]
            else None
        )

        num_tokens = input_2d.size(0)
        for start in range(0, num_tokens, ctx.chunk_size):
            end = min(start + ctx.chunk_size, num_tokens)
            input_chunk = input_2d[start:end]
            logits = _linear_chunk_fp32(input_chunk, weight, bias, ctx.logit_scale)
            fused_logits_backward_inplace(
                logits,
                row_maxes[start:end],
                sum_exps[start:end],
                local_targets[start:end],
                (grad_logprobs_1d[start:end] if grad_logprobs_1d is not None else None),
                1.0 / ctx.temperature,
                ctx.logit_scale,
            )
            dlogits = (
                _pack_fp32_to_half_inplace(
                    logits,
                    target_dtype=weight.dtype,
                    seed_elements=weight.size(0),
                )
                if weight.dtype in (torch.bfloat16, torch.float16)
                else logits
            )
            torch.mm(dlogits, weight, out=grad_input_2d[start:end])

            if needs_weight_grad:
                prepared_dlogits, prepared_input = (
                    mcore_layers.prepare_input_tensors_for_wgrad_compute(
                        dlogits, input_chunk
                    )
                )
                if ctx.use_main_grad:
                    weight.main_grad = ctx.main_grad
                    if weight.main_grad.dtype == torch.float32:
                        mcore_layers.fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                            prepared_input, prepared_dlogits, weight.main_grad
                        )
                    elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                        mcore_layers.fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                            prepared_input, prepared_dlogits, weight.main_grad
                        )
                    else:
                        raise RuntimeError(
                            f"Unsupported main_grad dtype {weight.main_grad.dtype}"
                        )
                else:
                    torch.addmm(
                        grad_weight,
                        prepared_dlogits.t(),
                        prepared_input,
                        out=grad_weight,
                    )
            if grad_bias is not None:
                grad_bias.add_(dlogits.sum(dim=0))

        if ctx.sequence_parallel:
            grad_input = torch.empty(
                ctx.input_shape,
                dtype=input_.dtype,
                device=input_.device,
                requires_grad=False,
            )
            mcore_layers.dist_reduce_scatter_func(
                grad_input, grad_input_2d.view_as(total_input), group=ctx.tp_group
            )
        else:
            grad_input = grad_input_2d.view_as(input_)
            if ctx.allreduce_dgrad:
                dist.all_reduce(grad_input, group=ctx.tp_group)

        if ctx.use_main_grad:
            if hasattr(weight, "grad_added_to_main_grad"):
                grad_weight = mcore_layers.get_dummy_wgrad(
                    list(weight.main_grad.shape),
                    input_.dtype,
                    zero=getattr(weight, "zero_out_wgrad", False),
                )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_lm_head_logprobs_entropy(
    output_layer: mcore_layers.ColumnParallelLinear,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
    logit_scale: float = 1.0,
) -> ChunkedLMHeadOutput:
    """Run a Megatron-compatible LM Head and logprob backward by sequence chunk."""
    if not isinstance(output_layer, _AReaLVocabParallelLMHeadMixin):
        raise TypeError("chunked LM Head loss requires the AReaL LM Head")
    if not getattr(output_layer, "chunked_loss_supported", True):
        raise NotImplementedError(
            "chunked LM Head loss does not support this output-layer forward override"
        )
    if output_layer.config.defer_embedding_wgrad_compute:
        raise NotImplementedError(
            "chunked LM Head loss is incompatible with defer_embedding_wgrad_compute"
        )
    if output_layer.gather_output:
        raise NotImplementedError(
            "chunked LM Head loss requires vocab-parallel (ungathered) output"
        )

    bias = output_layer.bias if not output_layer.skip_bias_add else None
    values = _ChunkedVocabParallelLMHead.apply(
        hidden_states,
        weight,
        bias,
        labels,
        temperature,
        chunk_size,
        output_layer.gradient_accumulation_fusion,
        output_layer.allreduce_dgrad,
        output_layer.sequence_parallel,
        output_layer.tp_group,
        logit_scale,
    )
    return ChunkedLMHeadOutput(*values)


class _AReaLVocabParallelLMHeadMixin:
    fp32_output: bool = False

    def _forward_impl(self, input, weight, *args, **kwargs):
        return linear_with_areal_output(
            input,
            weight,
            *args,
            **kwargs,
            fp32_output=self.fp32_output,
        )


class AReaLVocabParallelLMHead(
    _AReaLVocabParallelLMHeadMixin,
    mcore_layers.ColumnParallelLinear,
):
    """A ColumnParallelLinear LM head with an optional direct FP32 output."""


def replace_output_layer_with_areal_lm_head(
    model: torch.nn.Module,
    *,
    fp32_output: bool,
) -> None:
    """Promote an existing MCore LM head without replacing its parameters or hooks."""
    if not hasattr(model, "output_layer"):
        return

    output_layer = model.output_layer
    if not isinstance(output_layer, mcore_layers.ColumnParallelLinear):
        raise TypeError(
            "AReaL LM Head requires Megatron ColumnParallelLinear, got "
            f"{type(output_layer)}"
        )

    # Bridge model builders may add output-layer mixins such as Gemma2 logit
    # soft-capping. Add AReaL's implementation ahead of the existing class so
    # its forward behavior, parameters, and installed hooks all remain intact.
    if not isinstance(output_layer, _AReaLVocabParallelLMHeadMixin):
        original_class = output_layer.__class__
        output_layer.chunked_loss_supported = (
            original_class.forward is mcore_layers.ColumnParallelLinear.forward
        )
        if original_class is mcore_layers.ColumnParallelLinear:
            areal_class = AReaLVocabParallelLMHead
        else:
            areal_class = type(
                f"AReaL{original_class.__name__}",
                (_AReaLVocabParallelLMHeadMixin, original_class),
                {},
            )
        output_layer.__class__ = areal_class
    output_layer.fp32_output = fp32_output

    try:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
    except ImportError:
        return
    AutoMapping.register_module_type(output_layer.__class__.__name__, "column")
