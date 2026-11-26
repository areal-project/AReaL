# SPDX-License-Identifier: Apache-2.0
"""Two-NPU checks for PointerHead context-parallel reassembly and replica sync.

Launched by ``tests/test_qwen35_decision_npu.py``. Cases:

* CP=2: all-gather 2-D PointerHead projections, unpad, packed decision loss
* TP=2: broadcast / load ``head.pt`` so last-stage replicas match
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import GPTModel
from torch import nn

from areal.dataset.decision import ROLE_DECIDE, ROLE_OPT
from areal.engine.megatron_utils.packed_context_parallel import (
    reassemble_cp_packed_logprobs,
)
from areal.infra.platforms import current_platform
from areal.models.mcore.qwen35_decision import (
    DecisionHeadMeta,
    _agree_flag,
    load_decision_pointer_head,
    replace_output_layer_with_pointer_head,
    save_decision_pointer_head,
)
from areal.trainer.sft.decision import compute_packed_decision_loss, question_loss
from areal.utils.data import unpad_logits
from areal.utils.stats_tracker import DistributedStatsTracker


class _Stage(GPTModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.post_process = True
        self.vocab_size = 16
        self.output_layer = nn.Linear(8, 16, bias=False)


def _device() -> torch.device:
    return torch.device(current_platform.device_type, dist.get_rank())


def _split_packed(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Interleaved CP split that keeps trailing feature dims."""
    extra = tensor.shape[1:]
    input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    output_len = int(input_lens.sum().item()) // cp_size
    splitted = tensor.new_zeros((output_len, *extra))
    for i in range(input_lens.shape[0]):
        seqlen = int(input_lens[i].item()) // cp_size
        half = seqlen // 2
        start = int(cu_seqlens[i].item()) // cp_size
        row = tensor[cu_seqlens[i] : cu_seqlens[i + 1]]
        splitted[start : start + half] = row[half * cp_rank : half * (cp_rank + 1)]
        remain_start = seqlen * cp_size - half * (cp_rank + 1)
        remain_end = min(seqlen * cp_size - half * cp_rank, row.shape[0])
        remain = remain_end - remain_start
        splitted[start + half : start + half + remain] = row[remain_start:remain_end]
    return splitted


def _place_scores(
    proj: torch.Tensor,
    opt_idx: list[int],
    decide_idx: int,
    scores: torch.Tensor,
) -> None:
    dp = proj.shape[-1] // 2
    scale = 1.0 / math.sqrt(dp)
    proj[decide_idx, 0] = 1.0
    proj[opt_idx, dp] = scores.to(dtype=proj.dtype) / scale


def run_cp_reassemble_and_loss() -> None:
    """Decide / option tokens sit on different CP ranks after the interleaved split."""
    rank = dist.get_rank()
    device = _device()
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size != 2:
        raise RuntimeError(f"expected CP=2, got {cp_size}")

    # Two aligned rows. Seq0=16, seq1=8. Dummy tail of 8 is later unpadded.
    # Seq0 opts 3,7 / decide 15; seq1 opts 2,5 / decide 7.
    old_cu = torch.tensor([0, 16, 24], dtype=torch.int32, device=device)
    padded_cu = torch.tensor([0, 16, 24, 32], dtype=torch.int32, device=device)
    padding_length = 8
    dp = 4
    proj = torch.zeros(32, 2 * dp, dtype=torch.float32, device=device)
    z0 = torch.tensor([0.4, -0.7], dtype=torch.float32, device=device)
    z1 = torch.tensor([1.2, 0.3], dtype=torch.float32, device=device)
    _place_scores(proj, [3, 7], 15, z0)
    _place_scores(proj, [18, 21], 23, z1)

    roles = torch.zeros(24, dtype=torch.long, device=device)
    roles[[3, 7, 18, 21]] = ROLE_OPT
    roles[[15, 23]] = ROLE_DECIDE
    labels = torch.full((24,), -1, dtype=torch.long, device=device)
    labels[15] = 1
    labels[23] = 0
    qtypes = torch.full((24,), -1, dtype=torch.long, device=device)
    qtypes[[15, 23]] = 0
    soft = torch.full((24,), -1.0, dtype=torch.float32, device=device)
    orig = {
        "cu_seqlens": old_cu,
        "token_roles": roles,
        "choice_label": labels,
        "question_type": qtypes,
        "soft_target": soft,
    }

    local = _split_packed(proj, padded_cu, rank, cp_size).detach().requires_grad_(True)
    reassembled = reassemble_cp_packed_logprobs(local, padded_cu)
    unpadded = unpad_logits(reassembled, padding_length, padded_cu, old_cu)
    if tuple(unpadded.shape) != (24, 2 * dp):
        raise RuntimeError(f"rank {rank}: unpadded shape {tuple(unpadded.shape)}")

    tracker = DistributedStatsTracker()
    with patch("areal.trainer.sft.decision.stats_tracker", tracker):
        loss = compute_packed_decision_loss(unpadded, orig)
    expected = torch.stack(
        [
            question_loss(z0, 1, 0, None, 0.0),
            question_loss(z1, 0, 0, None, 0.0),
        ]
    ).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-5)

    loss.backward()
    if local.grad is None or not torch.isfinite(local.grad).all():
        raise RuntimeError(f"rank {rank}: CP decision loss gradient is invalid")
    # Each local token maps to exactly one global token. The dummy tail (24:32)
    # is dropped by unpad, so those CP shards must receive zero gradient.
    dummy_local = _split_packed(
        torch.arange(32, device=device), padded_cu, rank, cp_size
    )
    dummy_mask = dummy_local >= 24
    if local.grad[dummy_mask].abs().sum() != 0:
        raise RuntimeError(f"rank {rank}: dummy CP shard received a gradient")

    dist.barrier()
    if rank == 0:
        print("PASS decision CP reassemble + packed loss", flush=True)


def run_head_broadcast_and_checkpoint(weights_path: str) -> None:
    rank = dist.get_rank()
    device = _device()
    torch.manual_seed(10 + rank)
    stage = _Stage()
    stage.output_layer = stage.output_layer.to(device)
    head = replace_output_layer_with_pointer_head(
        stage,
        SimpleNamespace(hidden_size=8, sequence_parallel=True),
        head_dim=4,
    )
    if head is None:
        raise RuntimeError(f"rank {rank}: PointerHead was not attached")
    head.temperature = 1.25 + rank

    missing = os.path.join(weights_path, "missing")
    os.makedirs(missing, exist_ok=True)
    load_decision_pointer_head(stage, missing)
    gathered = [torch.empty_like(head.q.weight) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, head.q.weight.detach())
    for replica in gathered[1:]:
        torch.testing.assert_close(replica, gathered[0])
    temps = [torch.tensor([head.temperature], device=device) for _ in range(2)]
    dist.all_gather(temps, torch.tensor([head.temperature], device=device))
    if temps[0].item() != temps[1].item():
        raise RuntimeError(f"temperature replicas diverged: {temps}")

    if mpu.get_tensor_model_parallel_rank() == 0:
        head.q.weight.data.fill_(0.37)
        head.k.bias.data.fill_(-0.2)
        head.temperature = 2.5
    path = save_decision_pointer_head(
        stage,
        weights_path,
        meta=DecisionHeadMeta(
            base="Qwen/Qwen3.5-4B",
            weights="full",
            lora=0,
            temperature=2.5,
        ),
    )
    if rank == 0 and path is None:
        raise RuntimeError("checkpoint writer did not write head.pt")
    dist.barrier()

    with torch.no_grad():
        head.q.weight.fill_(float(rank + 1))
        head.k.bias.fill_(float(rank + 3))
        head.temperature = 9.0
    meta = load_decision_pointer_head(stage, weights_path)
    if meta is None or meta.weights != "full":
        raise RuntimeError(f"rank {rank}: failed to reload head.pt")
    torch.testing.assert_close(
        head.q.weight, torch.full_like(head.q.weight, 0.37), rtol=0, atol=0
    )
    torch.testing.assert_close(
        head.k.bias, torch.full_like(head.k.bias, -0.2), rtol=0, atol=0
    )
    if head.temperature != 2.5:
        raise RuntimeError(f"rank {rank}: temperature {head.temperature}")

    dist.barrier()
    if rank == 0:
        print("PASS decision head broadcast + head.pt", flush=True)


def run_agree_flag() -> None:
    device = _device()
    local = dist.get_rank() == 0
    agreed = _agree_flag(local, device)
    if agreed:
        raise RuntimeError("disagreeing ranks must take the missing-checkpoint branch")
    if not _agree_flag(True, device):
        raise RuntimeError("unanimous True must stay True")
    dist.barrier()
    if dist.get_rank() == 0:
        print("PASS decision agree-flag", flush=True)


def _init_mp(*, tensor_parallel: int = 1, context_parallel: int = 1) -> None:
    if mpu.model_parallel_is_initialized():
        mpu.destroy_model_parallel()
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tensor_parallel,
        context_parallel_size=context_parallel,
    )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    current_platform.set_device(local_rank)
    dist.init_process_group(backend=current_platform.communication_backend)
    weights_path = os.environ.get("DECISION_TEST_TMP")
    if not weights_path:
        raise RuntimeError("DECISION_TEST_TMP must point at a shared temp directory")
    try:
        _init_mp(context_parallel=2)
        run_cp_reassemble_and_loss()
        _init_mp(tensor_parallel=2)
        run_head_broadcast_and_checkpoint(weights_path)
        run_agree_flag()
    finally:
        if mpu.model_parallel_is_initialized():
            mpu.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
