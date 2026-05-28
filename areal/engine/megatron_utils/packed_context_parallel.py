# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_F
from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams

from areal.utils.data import is_multi_modal_key


def preprocess_packed_seqs_context_parallel(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences.
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1 gets second and second last chunks, and so on),
    this is for load balancing with causal masking. See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = input_lens.max().item()
    batch_size = input_lens.shape[0]

    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    align_to_multiple_of = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    # assume input_ids and cu_seqlens are already padded to align_to_multiple_of
    if any(length % align_to_multiple_of for length in input_lens) != 0:
        raise ValueError(
            f"Some of the input sequence length ({input_lens}) is not a multiple of align_to_multiple_of {align_to_multiple_of} "
            "for context/sequence parallel in Megatron."
        )

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_seqlen,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_kv=max_seqlen,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
    )

    if cp_size <= 1:
        return input_ids.unsqueeze(0), packed_seq_params

    shape = (input_lens.sum().item() // cp_size,)
    splitted = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
    for i in range(batch_size):
        seqlen = input_lens[i] // cp_size
        half_seqlen = seqlen // 2
        start_idx = cu_seqlens[i] // cp_size
        # split to 2 chunks
        d = input_ids[cu_seqlens[i] : cu_seqlens[i + 1]]
        splitted[start_idx : start_idx + half_seqlen] = d[
            half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
        ]

        remain_start = input_lens[i] - half_seqlen * (cp_rank + 1)
        remain_end = input_lens[i] - half_seqlen * cp_rank
        remain_end = min(remain_end, d.shape[0])
        remain_len = remain_end - remain_start
        splitted[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
            remain_start:remain_end
        ]
    return splitted.unsqueeze(0), packed_seq_params


def split_packed_seqs_for_context_parallel(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Split a 1D packed tensor using the same interleaved pattern as
    preprocess_packed_seqs_context_parallel."""
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    if cp_size <= 1:
        return tensor

    input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch_size = input_lens.shape[0]
    output_len = input_lens.sum().item() // cp_size

    splitted = torch.zeros(output_len, dtype=tensor.dtype, device=tensor.device)
    for i in range(batch_size):
        seqlen = input_lens[i] // cp_size
        half_seqlen = seqlen // 2
        start_idx = cu_seqlens[i] // cp_size

        d = tensor[cu_seqlens[i] : cu_seqlens[i + 1]]
        splitted[start_idx : start_idx + half_seqlen] = d[
            half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
        ]

        remain_start = input_lens[i] - half_seqlen * (cp_rank + 1)
        remain_end = input_lens[i] - half_seqlen * cp_rank
        remain_end = min(remain_end, d.shape[0])
        remain_len = remain_end - remain_start
        splitted[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
            remain_start:remain_end
        ]
    return splitted


def _build_cp_reassemble_indices(
    padded_cu_seqlens: torch.Tensor,
    cp_size: int,
) -> torch.Tensor:
    """Build the index mapping from concatenated CP chunks to original order.

    Returns a 1D LongTensor of length ``output_len`` where ``indices[dst] = src``
    means the token at position ``dst`` in the full sequence comes from position
    ``src`` in the flattened ``torch.cat(gathered_list)`` tensor.
    """
    input_lens = padded_cu_seqlens[1:] - padded_cu_seqlens[:-1]
    batch_size = input_lens.shape[0]
    output_len = int(padded_cu_seqlens[-1].item())
    local_len = output_len // cp_size
    device = padded_cu_seqlens.device

    indices = torch.empty(output_len, dtype=torch.long, device=device)

    for i in range(batch_size):
        seq_len = int(input_lens[i].item())
        chunk_size = seq_len // cp_size
        half_chunk = chunk_size // 2
        local_start = int(padded_cu_seqlens[i].item()) // cp_size
        full_start = int(padded_cu_seqlens[i].item())

        k = torch.arange(half_chunk, device=device)
        for j in range(cp_size):
            src_offset = j * local_len + local_start
            # first half → positions [j*H, (j+1)*H) in full sequence
            indices[full_start + j * half_chunk + k] = src_offset + k
            # second half → mirror positions [L-(j+1)*H, L-j*H)
            indices[full_start + seq_len - (j + 1) * half_chunk + k] = (
                src_offset + half_chunk + k
            )

    return indices


def reassemble_cp_packed_logprobs(
    local_tensor: torch.Tensor,
    padded_cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """All-gather CP-local 1D tensors and reassemble in original sequence order.

    This is the differentiable inverse of ``split_packed_seqs_for_context_parallel``.
    It uses ``torch.distributed.nn.functional.all_gather`` (backward = reduce_scatter)
    followed by advanced indexing (differentiable permutation) so that gradients
    flow correctly back to each CP rank's local logprobs.

    Args:
        local_tensor: 1D tensor of shape ``(total_packed_len // cp_size,)`` holding
            this rank's CP-local values (e.g. logprobs, entropy, vocab stats).
        padded_cu_seqlens: Cumulative sequence lengths in the *padded* (pre-split)
            layout, of shape ``(batch_size + 1,)``.

    Returns:
        Full-sequence 1D tensor of shape ``(total_packed_len,)`` with values placed
        back in original token order. Gradients flow back through the all-gather.
    """
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size <= 1:
        return local_tensor

    cp_group = mpu.get_context_parallel_group()

    # Differentiable all-gather: backward is reduce_scatter(sum).
    gathered_list = dist_F.all_gather(local_tensor, group=cp_group)

    # Concatenate all gathered chunks into a single flat tensor.
    # cat is differentiable (backward splits gradients back to each chunk).
    gathered_flat = torch.cat(gathered_list, dim=0)

    # Build index mapping and apply via advanced indexing (differentiable).
    # indices[dst] = src means output[dst] = gathered_flat[src].
    indices = _build_cp_reassemble_indices(padded_cu_seqlens, cp_size)
    return gathered_flat[indices]


def postprocess_packed_seqs_context_parallel(
    output: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    post_process: bool,
    gather_output: bool = True,
) -> torch.Tensor:
    """
    Postprocess packed sequences
    """
    cp_size = mpu.get_context_parallel_world_size()
    if not post_process:
        return output
    if cp_size <= 1 or cu_seqlens is None:
        return output.squeeze(0)

    if not gather_output:
        return output.squeeze(0)

    # shape = [batch_size, seq_len] + list(output.shape[2:])
    # [1, packed, dim] -> [batch_size, seq_len, dim]
    batch_size = cu_seqlens.shape[0] - 1
    output_len = int(cu_seqlens[-1].item())
    # output shape: [total_packed_seq_len] + list(output.shape[2:]
    output_new = torch.empty(
        (output_len, *output.shape[2:]), device=output.device, dtype=output.dtype
    )
    # all gather output across context parallel group
    # need to gather across cp group and concatenate in sequence dimension
    output_list = [torch.empty_like(output) for _ in range(cp_size)]
    dist.all_gather(
        output_list, output.detach(), group=mpu.get_context_parallel_group()
    )
    output_list[mpu.get_context_parallel_rank()] = output

    for i in range(batch_size):
        seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
        splitted_seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]) // cp_size
        half_splitted_seq_len = splitted_seq_len // 2

        tmp = torch.empty(
            (seq_len, *output.shape[2:]), device=output.device, dtype=output.dtype
        )
        for j in range(cp_size):
            o = output_list[j].squeeze(0)
            # split to 2 chunks
            start = cu_seqlens[i] // cp_size
            o0, o1 = (
                o[start : start + half_splitted_seq_len],
                o[start + half_splitted_seq_len : start + splitted_seq_len],
            )
            tmp[j * half_splitted_seq_len : (j + 1) * half_splitted_seq_len] = o0
            splitted_start = seq_len - (j + 1) * half_splitted_seq_len
            splitted_end = seq_len - j * half_splitted_seq_len
            tmp[splitted_start:splitted_end] = o1

        output_new[cu_seqlens[i] : cu_seqlens[i + 1]] = tmp[:seq_len]
    return output_new


_VLM_FORWARD_KEYS = ("pixel_values", "image_grid_thw", "video_grid_thw")


def _is_multi_modal_payload_key(key: str) -> bool:
    return key in _VLM_FORWARD_KEYS or is_multi_modal_key(key)


def _drop_multi_modal_payload(data: dict[str, Any]) -> None:
    for key in list(data.keys()):
        if _is_multi_modal_payload_key(key):
            data.pop(key, None)


def extract_vision_from_multi_modal(
    mb: dict[str, Any], padded_mb: dict[str, Any]
) -> None:
    """Extract pixel_values / image_grid_thw / video_grid_thw from multi_modal_input.

    Mirrors FSDPEngine's `_prepare_multimodal_forward_inputs` (#1272): vision
    tensors are placed only on ``padded_mb`` (forward side); ``mb`` is the
    loss/bookkeeping side and does not need them. The original
    ``multi_modal_input`` list-of-dicts is popped from both to avoid carrying
    raw per-sample tensors alongside the concatenated batched form.
    """
    multi_modal_input = mb.pop("multi_modal_input", None)
    if multi_modal_input is None:
        multi_modal_input = padded_mb.pop("multi_modal_input", None)
    else:
        padded_mb.pop("multi_modal_input", None)

    if multi_modal_input is not None:
        for key in _VLM_FORWARD_KEYS:
            items = [item[key] for item in multi_modal_input if key in item]
            if items:
                padded_mb[key] = torch.cat(items, dim=0)

    _drop_multi_modal_payload(mb)


def packed_context_parallel_forward(
    model: torch.nn.Module,
    input_: dict[str, Any],
    gather_cp_output: bool = True,
    is_vision_model: bool = False,
):
    input_ids = input_["input_ids"]
    position_ids = input_.get("position_ids", None)
    cu_seqlens = input_.get("cu_seqlens", None)
    # `attention_mask`: dense torch.Tensor (flex attention with Megatron) or None.
    # `tree_triton_data`: read from a separate key; takes priority over
    # attention_mask when forwarded as the final attention mask argument.
    attention_mask = input_.get("attention_mask", None)
    tree_triton_data = input_.get("tree_triton_data", None)
    packed_seq_params = None

    is_vision = is_vision_model and any(key in input_ for key in _VLM_FORWARD_KEYS)

    # Track whether we reconstructed 2D batch form for vision
    vision_repack_info = None

    if cu_seqlens is not None:
        if not is_vision:
            if attention_mask is not None or tree_triton_data is not None:
                raise ValueError(
                    "Attention mask should be None when using packed sequences."
                )
            input_ids, packed_seq_params = preprocess_packed_seqs_context_parallel(
                input_ids, cu_seqlens
            )
            input_ids = input_ids.contiguous()
        else:
            # VLM models expect batch-form [B, S] input_ids for mRoPE position
            # computation and vision token embedding replacement. Reconstruct
            # padded 2D tensors from packed 1D using cu_seqlens via boolean
            # masking — avoids per-sample Python loop and GPU-CPU sync.
            batch_size = cu_seqlens.shape[0] - 1
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = int(seq_lens.max().item())
            # int64 for input_ids: mbridge's get_rope_index uses input_ids.dtype
            # for position_ids, and some kernels (_index_put_impl_) require int64.
            # Upcast to torch.long so the scatter `input_ids_2d[mask] = input_ids`
            # below has matching source/dest dtypes (data pipeline may emit int32).
            if input_ids.dtype != torch.long:
                input_ids = input_ids.to(torch.long)
            attention_mask = (
                torch.arange(max_seqlen, device=input_ids.device)[None, :]
                < seq_lens[:, None]
            )
            input_ids_2d = torch.zeros(
                batch_size, max_seqlen, dtype=torch.long, device=input_ids.device
            )
            input_ids_2d[attention_mask] = input_ids
            input_ids = input_ids_2d
            vision_repack_info = (cu_seqlens, seq_lens, max_seqlen)

    # Pass tree_triton_data as attention_mask if present (for Triton tree attention)
    # Otherwise use the attention_mask from input (could be dense tensor for flex attention)
    # For VLM: pass None — the model's get_rope_index uses the 2D attention_mask
    # internally for correct mRoPE positions. Each batch slot holds one sequence
    # with trailing padding, so causal attention yields correct outputs at
    # non-padding positions; padding outputs are discarded during repack.
    if is_vision:
        final_attention_mask = None
    else:
        final_attention_mask = (
            tree_triton_data if tree_triton_data is not None else attention_mask
        )

    # VLM: pass vision inputs through to model forward. The VLM model computes
    # mRoPE position_ids internally, so position_ids remains None for VLM.
    vlm_kwargs: dict[str, Any] = {}
    if is_vision:
        for key in _VLM_FORWARD_KEYS:
            if key in input_:
                vlm_kwargs[key] = input_[key]

    try:
        output = model(
            input_ids=input_ids,
            attention_mask=final_attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            **vlm_kwargs,
        )
    except Exception as e:
        raise RuntimeError(
            f"Error occurred in packed context parallel forward pass on model {model} "
            f"with input_ids shape {input_ids.shape} and packed_seq_params {packed_seq_params}."
        ) from e

    model_vp_stage = getattr(model, "vp_stage", None)
    is_pipeline_last_stage = mpu.is_pipeline_last_stage(
        ignore_virtual=False, vp_stage=model_vp_stage
    )

    # Repack vision output to packed [total_len, ...] for the last PP stage only.
    # Intermediate stages must return their output unchanged so the pipeline
    # send/recv shapes match what the next stage expects (megatron-core's
    # `_communicate_shapes` negotiates based on this return value).
    #
    # On the last PP stage, megatron-core GPTModel returns logits already
    # transposed to [B, S, V] (gpt_model.py: `return logits.transpose(0, 1).contiguous()`),
    # so a boolean mask of valid positions selects the packed sequence.
    if vision_repack_info is not None and is_pipeline_last_stage:
        _, repack_seq_lens, repack_max_seqlen = vision_repack_info
        mask = (
            torch.arange(repack_max_seqlen, device=output.device)[None, :]
            < repack_seq_lens[:, None]
        )
        output = output[mask]
    output = postprocess_packed_seqs_context_parallel(
        output, cu_seqlens, is_pipeline_last_stage, gather_output=gather_cp_output
    )
    return output
