# SPDX-License-Identifier: Apache-2.0

"""Compatibility shims for torch_npu API drift across versions.

Applied on import; a no-op when torch_npu is absent (CUDA/CPU hosts). Import
this BEFORE ``mindspeed.megatron_adaptor`` so the shims are in place before
MindSpeed re-binds the affected symbols.
"""

import os

GROUPED_P2P_ENV = "AREAL_NPU_GROUPED_P2P"

# torch-npu 2.10.0.post2 added a device guard to
# ``ProcessGroupHCCL::startCoalescing()``:
#
#     npuGuard.set_index(getDeviceForRank(groupRanks()[getRank()]).index());
#
# ``groupRanks()`` falls back to a function-local ``static`` identity vector for
# groups without an explicit ``global_ranks_in_group``. That vector is sized by
# the first process group to call it, so a later root group whose rank exceeds
# that size reads out of bounds -- an invalid device or a hang. Weight-update
# groups are created as root groups, so any process holding one is exposed,
# regardless of SoC. Only the coalescing path is affected; the fused
# ``ProcessGroupHCCL::batch_isend_irecv()`` call is not.
#
# Upstream fix: https://gitcode.com/Ascend/pytorch/pull/42579
# The pinned 2.10.0.post4 wheel (git 5dd8ef3f) still contains this code.
GROUPED_P2P_BROKEN_SINCE = "2.10.0.post2"
GROUPED_P2P_FIXED_IN = None  # set once a wheel carrying the fix is validated


def grouped_p2p_broken(torch_npu_version: str) -> bool:
    """Whether this torch-npu carries the ``startCoalescing()`` regression."""
    from packaging.version import InvalidVersion, Version

    try:
        version = Version(torch_npu_version)
    except InvalidVersion:
        return False
    if version < Version(GROUPED_P2P_BROKEN_SINCE):
        return False
    return GROUPED_P2P_FIXED_IN is None or version < Version(GROUPED_P2P_FIXED_IN)


def _apply_grouped_p2p_gate() -> None:
    """Route NPU batched P2P away from torch-npu's broken coalescing path.

    On affected torch-npu versions (see ``GROUPED_P2P_BROKEN_SINCE``), dispatch
    to the fused per-group HCCL call instead of ``_coalescing_manager()``. This
    is the path every device used before 2.10.0.post2 and A3 ``Ascend910_9392``
    used until post2 widened the grouped-P2P device gate.

    Set ``AREAL_NPU_GROUPED_P2P=1`` to keep the coalescing path (e.g. to verify
    an upstream fix) or ``=0`` to force the fused call on any torch-npu version.
    """
    try:
        import torch
        import torch.distributed as dist
        import torch_npu
    except ImportError:
        return

    from areal.utils import logging

    logger = logging.getLogger("TorchNPUCompat")

    version = getattr(torch_npu, "__version__", "")
    override = os.getenv(GROUPED_P2P_ENV, "auto").lower()
    if override in ("1", "true", "on"):
        logger.info(f"{GROUPED_P2P_ENV}={override}: keeping grouped NPU P2P.")
        return
    forced = override in ("0", "false", "off")
    if not forced and not grouped_p2p_broken(version):
        return

    original = dist.batch_isend_irecv
    if getattr(original, "_areal_grouped_p2p_gate", False):
        return
    if getattr(original, "__name__", "") != "_batch_isend_irecv":
        # torch_npu did not install its override; nothing to gate.
        return

    def legacy_batch_isend_irecv(p2p_op_list):
        """The fused branch of torch-npu's ``_batch_isend_irecv``."""
        from torch.distributed.distributed_c10d import (
            _get_default_group,
            get_group_rank,
        )

        group = p2p_op_list[0].group
        is_multi_pg = group is not None
        if group is None:
            group = _get_default_group()
        backend = group._get_backend(p2p_op_list[0].tensor.device)

        op_type, tensors, remote_rank_list = [], [], []
        for p2p_op in p2p_op_list:
            if p2p_op.tensor.device.type != "npu":
                raise RuntimeError(
                    "No backend type associated with device type "
                    f"{p2p_op.tensor.device.type}"
                )
            op_type.append(p2p_op.op.__name__)
            tensors.append(p2p_op.tensor)
            remote_rank_list.append(
                get_group_rank(group, p2p_op.peer) if is_multi_pg else p2p_op.peer
            )
        return [backend.batch_isend_irecv(op_type, tensors, remote_rank_list)]

    def batch_isend_irecv(p2p_op_list):
        if p2p_op_list and p2p_op_list[0].tensor.device.type == "npu":
            return legacy_batch_isend_irecv(p2p_op_list)
        return original(p2p_op_list)

    batch_isend_irecv._areal_grouped_p2p_gate = True
    torch.distributed.batch_isend_irecv = batch_isend_irecv
    torch.distributed.distributed_c10d.batch_isend_irecv = batch_isend_irecv
    reason = f"{GROUPED_P2P_ENV}={override}" if forced else f"torch_npu {version}"
    logger.info(
        f"Grouped NPU P2P disabled ({reason}); using the fused HCCL call. "
        f"Override with {GROUPED_P2P_ENV}=1."
    )


def _apply() -> None:
    _apply_grouped_p2p_gate()


_apply()
