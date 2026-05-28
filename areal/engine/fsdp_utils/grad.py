# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from areal.infra.platforms import current_platform

__all__ = [
    "fsdp2_clip_grad_norm",
]

try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
        multi_tensor_scale,
    )

    l2_norm_impl = multi_tensor_l2norm
    multi_tensor_scale_impl = multi_tensor_scale
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier

        l2_norm_impl = amp_C.multi_tensor_l2norm
        multi_tensor_scale_impl = amp_C.multi_tensor_scale
    except ImportError:
        import warnings

        warnings.warn(
            "Transformer Engine and Apex are not installed. "
            "Falling back to local implementations of multi_tensor_applier, "
            "multi_tensor_l2norm, and multi_tensor_scale"
        )

        from .multi_tensor_apply import (
            local_multi_tensor_applier,
            local_multi_tensor_l2_norm,
            local_multi_tensor_scale,
        )

        multi_tensor_applier = local_multi_tensor_applier
        l2_norm_impl = local_multi_tensor_l2_norm
        multi_tensor_scale_impl = local_multi_tensor_scale


def to_local_if_dtensor(tensor: Tensor | DTensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def device_mesh_has_dim(mesh: DeviceMesh, dim_name: str) -> bool:
    return mesh.mesh_dim_names is not None and dim_name in mesh.mesh_dim_names


def is_param_not_tensor_parallel_duplicate(param, tensor_parallel_rank: int):
    if tensor_parallel_rank == 0:
        return True

    if not isinstance(param, DTensor) or not device_mesh_has_dim(
        param.device_mesh, "tp"
    ):
        return False

    mesh = param.device_mesh
    if mesh.mesh_dim_names:
        placement = param.placements[mesh.mesh_dim_names.index("tp")]
        return not placement.is_replicate()

    return True


def get_main_grads_for_grad_norm(params, tensor_parallel_rank: int) -> list[Tensor]:
    return [
        param.grad
        for param in params
        if param.grad is not None
        and is_param_not_tensor_parallel_duplicate(param, tensor_parallel_rank)
    ]


# Adapted from Megatron-LM
def get_grad_norm_fp32(
    grads_for_norm: list[Tensor] | Tensor,
    data_parallel_group: ProcessGroup | None = None,
    model_parallel_group: ProcessGroup | None = None,
    norm_type: float = 2.0,
    offload_params: bool = False,
) -> float:
    if isinstance(grads_for_norm, Tensor):
        grads_for_norm = [grads_for_norm]

    grads_for_norm = [to_local_if_dtensor(grad).detach() for grad in grads_for_norm]

    norm_type = float(norm_type)
    total_norm = 0.0

    device = current_platform.current_device()

    if not grads_for_norm:
        # Still participate in all_reduce with zero contribution so that
        # ranks with grads don't hang waiting for this rank (e.g. LoRA frozen ranks).
        total_norm_cuda = torch.tensor(0.0, dtype=torch.float, device=device)
        reduce_op = dist.ReduceOp.MAX if norm_type == torch.inf else dist.ReduceOp.SUM
        if data_parallel_group:
            dist.all_reduce(total_norm_cuda, op=reduce_op, group=data_parallel_group)
        if model_parallel_group is not None:
            dist.all_reduce(total_norm_cuda, op=reduce_op, group=model_parallel_group)
        total_norm = float(total_norm_cuda.item())
        if norm_type != torch.inf and total_norm > 0:
            total_norm = total_norm ** (1.0 / norm_type)
        return total_norm

    if norm_type == torch.inf:
        norms = [grad.abs().max() for grad in grads_for_norm]
        total_norm = torch.max(torch.stack(norms)) if norms else 0.0
        total_norm_cuda = torch.tensor(
            float(total_norm), dtype=torch.float, device=device
        )
        if data_parallel_group:
            dist.all_reduce(
                total_norm_cuda,
                op=dist.ReduceOp.MAX,
                group=data_parallel_group,
            )
        if model_parallel_group is not None:
            dist.all_reduce(
                total_norm_cuda,
                op=dist.ReduceOp.MAX,
                group=model_parallel_group,
            )
        total_norm = float(total_norm_cuda.item())
    else:
        if norm_type == 2.0 and not offload_params:
            # Use multi_tensor_applier for better performance when grads are on GPU
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=device)
            grad_norm, _ = multi_tensor_applier(
                l2_norm_impl,
                dummy_overflow_buf,
                [grads_for_norm],
                False,
            )
            total_norm_cuda = grad_norm**norm_type
        elif not offload_params:
            grad_norms = [torch.norm(grad, norm_type) for grad in grads_for_norm]
            total_norm_cuda = torch.stack(grad_norms).pow(norm_type).sum()
        else:
            total_norm = 0.0
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type).item()
                total_norm += grad_norm**norm_type
            total_norm_cuda = torch.tensor(
                float(total_norm), dtype=torch.float, device=device
            )

        if data_parallel_group:
            dist.all_reduce(
                total_norm_cuda,
                op=dist.ReduceOp.SUM,
                group=data_parallel_group,
            )
        if model_parallel_group is not None:
            dist.all_reduce(
                total_norm_cuda,
                op=dist.ReduceOp.SUM,
                group=model_parallel_group,
            )
        total_norm = float(total_norm_cuda.item()) ** (1.0 / norm_type)

    return total_norm


# Adapted from Megatron-LM
def clip_grad_by_total_norm_fp32(
    parameters: list[nn.Parameter],
    max_norm: int | float,
    total_norm: float,
    offload_params: bool = False,
) -> None:
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff >= 1.0:
        return

    # dtype -> grad
    grads = defaultdict(list)
    for param in parameters:
        if param.grad is not None:
            grad = to_local_if_dtensor(param.grad).detach()
            grads[grad.dtype].append(grad)

    if len(grads) == 0:
        return

    from .multi_tensor_apply import (
        local_multi_tensor_applier,
        local_multi_tensor_scale,
    )

    if not offload_params:
        # GPU path
        for dtype, _grads in grads.items():
            dummy_overflow_buf = torch.tensor(
                [0], dtype=torch.int, device=current_platform.device_type
            )
            # For naive FSDP, lm_head has bf16 grad while others have fp32 grad
            if dtype == torch.float32:
                multi_tensor_applier(
                    multi_tensor_scale_impl,
                    dummy_overflow_buf,
                    [_grads, _grads],
                    clip_coeff,
                )
            else:
                local_multi_tensor_applier(
                    local_multi_tensor_scale,
                    dummy_overflow_buf,
                    [_grads, _grads],
                    clip_coeff,
                )
    else:
        # CPU path
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cpu")
        for _grads in grads.values():
            local_multi_tensor_applier(
                local_multi_tensor_scale,
                dummy_overflow_buf,
                [_grads, _grads],
                clip_coeff,
            )


def fsdp2_clip_grad_norm(
    parameters: list[nn.Parameter],
    max_norm: float,
    fsdp_group: ProcessGroup,
    tp_group: ProcessGroup | None = None,
    pp_group: ProcessGroup | None = None,
    norm_type: float = 2.0,
    offload_params: bool = False,
) -> float:
    if norm_type <= 0 and norm_type != float("inf"):
        raise ValueError(
            f"Invalid norm_type {norm_type}. Must be a positive float or inf."
        )

    tensor_parallel_rank = dist.get_rank(tp_group) if tp_group is not None else 0

    grads_for_norm = get_main_grads_for_grad_norm(parameters, tensor_parallel_rank)

    grad_norm = get_grad_norm_fp32(
        grads_for_norm,
        fsdp_group,
        tp_group,
        norm_type=norm_type,
        offload_params=offload_params,
    )

    # Reduce gradient norm across PP stages
    if pp_group is not None:
        device = current_platform.current_device()
        grad_norm_tensor = torch.tensor(grad_norm, dtype=torch.float, device=device)
        if norm_type == float("inf"):
            # For inf norm, use MAX reduction
            dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.MAX, group=pp_group)
        else:
            # For L-p norm: sum the p-th powers, then take p-th root
            grad_norm_tensor **= norm_type
            dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.SUM, group=pp_group)
            grad_norm_tensor **= 1.0 / norm_type
        grad_norm = float(grad_norm_tensor.item())

    if parameters:
        clip_grad_by_total_norm_fp32(parameters, max_norm, grad_norm, offload_params)

    return grad_norm
