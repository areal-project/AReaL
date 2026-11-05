# SPDX-License-Identifier: Apache-2.0

"""Compare native packed MTP CP loss/gradients with an unsharded reference."""

import os
from types import SimpleNamespace

from areal.utils.torch_npu_compat import import_mindspeed_adaptor

import_mindspeed_adaptor()

import torch  # noqa: E402, I001
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from megatron.core import parallel_state as mpu  # noqa: E402
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler  # noqa: E402

from areal.engine.megatron_utils.mtp import (  # noqa: E402
    _MTP_BATCH,
    _MTP_TRAIN_LABELS,
    _MTP_TRAIN_LOSS_MASK,
    _mtp_loss,
    mtp_backward_scale,
)
from areal.engine.megatron_utils.packed_context_parallel import (  # noqa: E402
    split_packed_seqs_for_context_parallel,
)


def main():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    cp = dist.get_world_size()
    mpu.initialize_model_parallel(context_parallel_size=cp)
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    cu = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=8,
        max_seqlen_kv=8,
    )
    ids = torch.arange(16, device=device)
    # Rank zero loses all main supervision in the first sequence, but receives
    # its MTP target from rank one when CP=2.
    mask = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0], device=device)
    features = torch.arange(48, device=device, dtype=torch.float32).reshape(16, 3) / 48
    initial = torch.arange(60, device=device, dtype=torch.float32).reshape(3, 20) / 60
    reference = initial.clone().requires_grad_(True)
    targets = torch.tensor(
        [2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12, 13, 14, 15, 0, 0], device=device
    )
    shifted_mask = torch.tensor(
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0], device=device
    )
    ref_loss = (
        F.cross_entropy(features @ reference, targets, reduction="none") * shifted_mask
    ).sum() / shifted_mask.sum()
    ref_loss.backward()
    local_ids = split_packed_seqs_for_context_parallel(ids, cu).unsqueeze(0)
    local_mask = split_packed_seqs_for_context_parallel(mask, cu).unsqueeze(0)
    indices = split_packed_seqs_for_context_parallel(ids, cu)
    for per_token in (False, True):
        parameter = initial.clone().requires_grad_(True)
        mtp_hidden = (features[indices] @ parameter).unsqueeze(1)
        backbone = torch.zeros_like(mtp_hidden, requires_grad=True)
        output_weight = torch.nn.Parameter(torch.eye(20, device=device))

        def projection(x, weight, runtime_gather_output):
            return x @ weight.t(), None

        projection.weight = output_weight

        def cross_entropy(labels, logits):
            return F.cross_entropy(
                logits.squeeze(1), labels.squeeze(0), reduction="none"
            ).unsqueeze(0)

        batch = {
            "scale": mtp_backward_scale(
                mask.sum() * cp,
                shifted_mask.sum(),
                cp_size=cp,
                dp_cp_size=cp,
                num_microbatches=3,
                per_token=per_token,
            ),
            "metrics": torch.zeros(2, device=device),
        }
        tokens = [
            (_MTP_BATCH, _MTP_BATCH.set(batch)),
            (_MTP_TRAIN_LABELS, _MTP_TRAIN_LABELS.set(local_ids)),
            (_MTP_TRAIN_LOSS_MASK, _MTP_TRAIN_LOSS_MASK.set(local_mask)),
        ]
        try:
            MTPLossAutoScaler.set_loss_scale(
                torch.tensor(1.0 if per_token else 1 / 3, device=device)
            )
            output = _mtp_loss(
                torch.cat((backbone, mtp_hidden)),
                None,
                None,
                projection,
                None,
                None,
                True,
                cross_entropy,
                SimpleNamespace(
                    mtp_loss_scaling_factor=1.0, calculate_per_token_loss=per_token
                ),
                cp_group=mpu.get_context_parallel_group(),
                packed_seq_params=packed,
            )
            (output.sum() * 0).backward()
        finally:
            for var, token in reversed(tokens):
                var.reset(token)
        assert output_weight.grad is None
        assert torch.count_nonzero(backbone.grad) == 0
        dist.all_reduce(parameter.grad, group=mpu.get_context_parallel_group())
        parameter.grad.div_(mask.sum() if per_token else cp)
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-5, atol=1e-6)
        dist.all_reduce(batch["metrics"], group=mpu.get_context_parallel_group())
        torch.testing.assert_close(
            batch["metrics"][0] / batch["metrics"][1], ref_loss, rtol=1e-5, atol=1e-6
        )
    print(f"MTP_CP_{cp}_LOSS_GRADIENT_OK", flush=True)
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
