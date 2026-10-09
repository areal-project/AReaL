# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist

from areal.utils.data import TRANSPORT_DUMMY_KEY, MicroBatchList


def validate_transport_padding(
    mb_list: MicroBatchList,
    *,
    has_internal_objectives: bool,
    cpu_group: dist.ProcessGroup,
) -> None:
    """Reject unsupported padding on every training rank before model execution.

    Zero external loss does not neutralize MCore's MoE/MTP auxiliary gradients,
    router state, or num_microbatches normalization. The group must include all
    DP/TP/CP/PP/EP participants, including stages without local auxiliary heads.
    """
    flags = torch.tensor(
        [
            any(mb.get(TRANSPORT_DUMMY_KEY) is True for mb in mb_list.mbs),
            has_internal_objectives,
        ],
        dtype=torch.int32,
        device="cpu",
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=cpu_group)
    if flags[0] and flags[1]:
        raise ValueError(
            "Megatron transport padding is not supported with MoE or MTP training: "
            "dummy microbatches can affect auxiliary gradients, router state, and "
            "num_microbatches normalization. Use batches and microbatch settings "
            "that require no transport padding, or disable MoE/MTP. "
            "All training ranks stopped before model execution."
        )
