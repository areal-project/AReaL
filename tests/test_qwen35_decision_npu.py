# SPDX-License-Identifier: Apache-2.0
"""NPU tests for Qwen3.5 PointerHead decision training.

CPU tests cover the formulas. These exercise the device path that actually
trains: packed roles through AReaL pad/unpad, GDN host ``cu_seqlens``,
context-parallel reassembly of 2-D projections, and last-stage head sync.
"""

from __future__ import annotations

import math
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from areal.api.cli_args import MicroBatchSpec
from areal.dataset.decision import (
    QUESTION_TYPE_IDS,
    ROLE_DECIDE,
    ROLE_OPT,
    SPECIAL,
    encode_question_rows,
)
from areal.engine.megatron_utils.mindspeed_gdn_patch import (
    cu_seqlens_host_list,
    remember_cu_seqlens,
)
from areal.engine.megatron_utils.packed_context_parallel import (
    _build_thd_packed_seq_params,
    reassemble_cp_packed_logprobs,
)
from areal.models.mcore.qwen35_decision import PointerHead
from areal.trainer.sft.decision import (
    _decision_loss_weight,
    compute_packed_decision_loss,
    question_loss,
)
from areal.utils.data import (
    MicroBatchList,
    _move_tensor_container_remembering_cu_seqlens,
    amend_position_ids,
    collate_samples_to_list,
    concat_padded_tensors,
    pack_tensor_dict,
    pad_packed_tensor_dict,
    unpad_logits,
)
from areal.utils.stats_tracker import DistributedStatsTracker

pytestmark = pytest.mark.npu

_CHOICE = QUESTION_TYPE_IDS["choice"]
_SCORE = QUESTION_TYPE_IDS["score"]


class _Tok:
    def __init__(self):
        self._ids = {token: i + 1 for i, token in enumerate(SPECIAL)}

    def convert_tokens_to_ids(self, token):
        return self._ids[token]

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[100 + ord(ch) for ch in text])


def _require_npu():
    from areal.infra.platforms import current_platform

    if current_platform.device_type != "npu" or current_platform.device_count() < 1:
        pytest.skip("NPU is not available")
    current_platform.set_device(0)
    return torch.device("npu", 0)


def _proj_from_logits(z: torch.Tensor, dp: int = 8) -> torch.Tensor:
    k = z.numel()
    scale = 1.0 / math.sqrt(dp)
    q_all = torch.zeros(k + 1, dp, dtype=torch.float32, device=z.device)
    k_all = torch.zeros(k + 1, dp, dtype=torch.float32, device=z.device)
    q_all[-1, 0] = 1.0
    k_all[:-1, 0] = z.float() / scale
    return torch.cat([q_all, k_all], dim=-1)


def _question_tensors(
    z: torch.Tensor,
    label: int,
    qtype: int,
    soft: list[float] | None = None,
    dp: int = 8,
) -> dict[str, torch.Tensor]:
    k = z.numel()
    roles = torch.full((k + 1,), ROLE_OPT, dtype=torch.long, device=z.device)
    roles[-1] = ROLE_DECIDE
    choice_label = torch.full((k + 1,), -1, dtype=torch.long, device=z.device)
    choice_label[-1] = label
    question_type = torch.full((k + 1,), -1, dtype=torch.long, device=z.device)
    question_type[-1] = qtype
    if soft is None:
        soft_target = torch.full((k + 1,), -1.0, dtype=torch.float32, device=z.device)
    else:
        soft_target = torch.tensor(
            list(soft) + [-1.0], dtype=torch.float32, device=z.device
        )
    return {
        "proj": _proj_from_logits(z, dp=dp),
        "token_roles": roles,
        "choice_label": choice_label,
        "question_type": question_type,
        "soft_target": soft_target,
    }


def _pack(questions: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, dict]:
    proj = torch.cat([q["proj"] for q in questions], dim=0)
    lengths = [q["token_roles"].numel() for q in questions]
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=proj.device)
    cu[1:] = torch.cumsum(
        torch.tensor(lengths, dtype=torch.int32, device=proj.device), dim=0
    )
    return proj, {
        "cu_seqlens": cu,
        "token_roles": torch.cat([q["token_roles"] for q in questions]),
        "choice_label": torch.cat([q["choice_label"] for q in questions]),
        "question_type": torch.cat([q["question_type"] for q in questions]),
        "soft_target": torch.cat([q["soft_target"] for q in questions]),
    }


def _dummy_row(n: int, device: torch.device, dp: int = 8) -> dict[str, torch.Tensor]:
    return {
        "proj": torch.zeros(n, 2 * dp, dtype=torch.float32, device=device),
        "token_roles": torch.zeros(n, dtype=torch.long, device=device),
        "choice_label": torch.full((n,), -1, dtype=torch.long, device=device),
        "question_type": torch.full((n,), -1, dtype=torch.long, device=device),
        "soft_target": torch.full((n,), -1.0, dtype=torch.float32, device=device),
    }


def _split_packed(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
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
        remain_start = int(input_lens[i].item()) - half * (cp_rank + 1)
        remain_end = min(int(input_lens[i].item()) - half * cp_rank, row.shape[0])
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
    proj[opt_idx, dp] = scores.to(dtype=proj.dtype, device=proj.device) / scale


def _record():
    return {
        "state": "ticket 12",
        "questions": [
            {
                "instr": "team?",
                "options": ["billing", "shipping"],
                "label": 1,
                "qtype": "choice",
            },
            {
                "instr": "urgent?",
                "options": ["low", "high"],
                "label": 0,
                "qtype": "score",
                "target": [0.25, 0.75],
            },
        ],
    }


def test_npu_question_loss_variants_backward():
    device = _require_npu()
    z = torch.tensor([0.4, -0.7, 1.2], device=device, requires_grad=True)
    hard = question_loss(z, 1, _CHOICE, None, 0.0)
    torch.testing.assert_close(
        hard, F.cross_entropy(z[None], torch.tensor([1], device=device))
    )
    t = torch.tensor([1 / 3, 1 / 3, 1 / 3], device=device)
    soft = question_loss(z.detach(), 0, _CHOICE, t, 0.0, brier_w=0.5)
    torch.testing.assert_close(soft, -(t * F.log_softmax(z.detach(), -1)).sum())
    score = question_loss(
        torch.tensor([0.2, 0.1, 0.7], device=device).log(), 1, _SCORE, None, 0.5
    )
    assert score.device.type == "npu"
    hard.backward()
    assert torch.isfinite(z.grad).all()


def test_npu_packed_loss_skips_dummy_and_matches_mean():
    device = _require_npu()
    z0 = torch.tensor([0.4, -0.7, 1.2], device=device)
    z1 = torch.tensor([-0.3, 0.9], device=device)
    proj, input_ = _pack(
        [
            _question_tensors(z0, 2, _CHOICE),
            _dummy_row(5, device),
            _question_tensors(z1, 0, _SCORE),
        ]
    )
    proj = proj.detach().requires_grad_(True)
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.sft.decision.stats_tracker", tracker):
        loss = compute_packed_decision_loss(proj, input_, ord_w=0.5)
    expected = torch.stack(
        [
            question_loss(z0, 2, _CHOICE, None, 0.5),
            question_loss(z1, 0, _SCORE, None, 0.5),
        ]
    ).mean()
    torch.testing.assert_close(loss, expected)
    assert _decision_loss_weight(input_) == 2
    loss.backward()
    assert torch.isfinite(proj.grad).all()
    stats = tracker.export(reset=True)
    assert stats["n_questions"] == 2.0


def test_npu_pointer_head_scores_packed_loss():
    device = _require_npu()
    torch.manual_seed(0)
    head = PointerHead(8, dp=4).to(device)
    hidden = torch.randn(6, 8, dtype=torch.bfloat16, device=device, requires_grad=True)
    projected, extra = head(hidden)
    assert extra is None
    assert projected.shape == (6, 8)
    assert projected.dtype == torch.float32
    roles = torch.tensor([0, ROLE_OPT, ROLE_OPT, 0, ROLE_DECIDE, 0], device=device)
    labels = torch.full((6,), -1, dtype=torch.long, device=device)
    labels[4] = 1
    qtypes = torch.full((6,), -1, dtype=torch.long, device=device)
    qtypes[4] = _CHOICE
    input_ = {
        "cu_seqlens": torch.tensor([0, 6], dtype=torch.int32, device=device),
        "token_roles": roles,
        "choice_label": labels,
        "question_type": qtypes,
        "soft_target": torch.full((6,), -1.0, device=device),
    }
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.sft.decision.stats_tracker", tracker):
        loss = compute_packed_decision_loss(projected, input_)
    q_all, k_all = projected.detach().chunk(2, dim=-1)
    z = (k_all[[1, 2]] @ q_all[4]) * head.scale
    torch.testing.assert_close(
        loss, F.cross_entropy(z[None], torch.tensor([1], device=device))
    )
    loss.backward()
    assert head.q.weight.grad is not None
    assert hidden.grad is not None
    assert torch.isfinite(head.q.weight.grad).all()


def test_npu_encode_pack_align_unpad_matches_unpadded_head():
    """TP/CP alignment pads must not move </opt> or <decide> after unpad."""
    device = _require_npu()
    torch.manual_seed(1)
    rows = encode_question_rows(_Tok(), _record(), as_tensors=True)
    batch = amend_position_ids(concat_padded_tensors(collate_samples_to_list(rows)))
    packed = pack_tensor_dict(batch)
    padded, pad_len, old_cu, _ = pad_packed_tensor_dict(
        packed, pad_to_length=256, seq_align_to=8
    )
    assert pad_len > 0
    assert (padded["token_roles"] == ROLE_DECIDE).sum() == (
        packed["token_roles"] == ROLE_DECIDE
    ).sum()
    assert (padded["token_roles"] == ROLE_OPT).sum() == (
        packed["token_roles"] == ROLE_OPT
    ).sum()
    assert padded["token_roles"].numel() > packed["token_roles"].numel()

    head = PointerHead(8, dp=4).to(device)
    hidden = torch.randn(
        packed["input_ids"].numel(), 8, dtype=torch.bfloat16, device=device
    )
    hidden_pad = torch.zeros(
        padded["input_ids"].numel(), 8, dtype=torch.bfloat16, device=device
    )
    new_cu = padded["cu_seqlens"]
    for i in range(old_cu.numel() - 1):
        old_s, old_e = int(old_cu[i]), int(old_cu[i + 1])
        new_s = int(new_cu[i])
        hidden_pad[new_s : new_s + (old_e - old_s)] = hidden[old_s:old_e]

    ref, _ = head(hidden)
    proj_pad, _ = head(hidden_pad)
    proj = unpad_logits(
        proj_pad.to(device),
        pad_len,
        new_cu.to(device),
        old_cu.to(device),
    )
    torch.testing.assert_close(proj, ref)

    orig = {
        key: packed[key].to(device)
        for key in (
            "cu_seqlens",
            "token_roles",
            "choice_label",
            "question_type",
            "soft_target",
        )
    }
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.sft.decision.stats_tracker", tracker):
        actual = compute_packed_decision_loss(proj, orig)
        expected = compute_packed_decision_loss(ref, orig)
    torch.testing.assert_close(actual, expected)
    assert _decision_loss_weight(orig) == 2


def test_npu_cu_seqlens_host_cache_survives_device_move():
    device = _require_npu()
    data = {
        "cu_seqlens": torch.tensor([0, 5, 11], dtype=torch.int32),
        "max_seqlen": 6,
        "input_ids": torch.arange(11, dtype=torch.int32),
        "token_roles": torch.tensor(
            [
                0,
                ROLE_OPT,
                0,
                ROLE_OPT,
                ROLE_DECIDE,
                0,
                ROLE_OPT,
                0,
                ROLE_OPT,
                0,
                ROLE_DECIDE,
            ]
        ),
    }
    padded, _, _, _ = pad_packed_tensor_dict(data, pad_to_length=256, seq_align_to=8)
    host = cu_seqlens_host_list(padded["cu_seqlens"])
    assert host[0] == 0
    assert host[-1] == 256

    mb = MicroBatchList(
        data={"cu_seqlens": padded["cu_seqlens"]},
        mb_spec=MicroBatchSpec(),
        mbs=[{"cu_seqlens": padded["cu_seqlens"]}],
        group_lens=[11],
        padded_mbs=[{"cu_seqlens": padded["cu_seqlens"]}],
        padding_lengths=[256 - int(padded["cu_seqlens"][-2].item())],
        padded_to_lengths=[256],
    )
    moved = mb.to(device)
    moved_cu = moved.padded_mbs[0]["cu_seqlens"]
    assert moved_cu.device.type == "npu"
    moved_cu.detach = lambda: (_ for _ in ()).throw(AssertionError("device sync"))  # type: ignore[method-assign]
    assert cu_seqlens_host_list(moved_cu) == host

    converted = moved_cu.to(dtype=torch.int64)
    remember_cu_seqlens(converted, host=host)
    converted.detach = lambda: (_ for _ in ()).throw(AssertionError("device sync"))  # type: ignore[method-assign]
    assert cu_seqlens_host_list(converted) == host


def test_npu_thd_packed_seq_params_require_tp_cp_alignment():
    device = _require_npu()
    aligned = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    remember_cu_seqlens(aligned, host=[0, 8, 16])
    with (
        patch(
            "areal.engine.megatron_utils.packed_context_parallel.mpu.get_tensor_model_parallel_world_size",
            return_value=2,
        ),
        patch(
            "areal.engine.megatron_utils.packed_context_parallel.mpu.get_context_parallel_world_size",
            return_value=2,
        ),
    ):
        params = _build_thd_packed_seq_params(aligned)
        assert int(params.max_seqlen_q) == 8
        with pytest.raises(ValueError, match="divisible"):
            _build_thd_packed_seq_params(
                torch.tensor([0, 5, 11], dtype=torch.int32, device=device)
            )


def test_npu_2d_cp_reassemble_unpad_and_decision_loss():
    """PointerHead output is [tokens, 2*dp]; CP must permute dim 0 only."""
    device = _require_npu()
    cp_size = 2
    old_cu = torch.tensor([0, 16, 24], dtype=torch.int32, device=device)
    padded_cu = torch.tensor([0, 16, 24, 32], dtype=torch.int32, device=device)
    dp = 4
    proj = torch.zeros(32, 2 * dp, dtype=torch.float32, device=device)
    z0 = torch.tensor([0.5, -1.0], device=device)
    z1 = torch.tensor([0.25, 0.75], device=device)
    _place_scores(proj, [3, 7], 15, z0)
    _place_scores(proj, [18, 21], 23, z1)
    splits = [
        _split_packed(proj, padded_cu, rank, cp_size).detach().requires_grad_(True)
        for rank in range(cp_size)
    ]

    with (
        patch(
            "areal.engine.megatron_utils.packed_context_parallel.mpu.get_context_parallel_world_size",
            return_value=cp_size,
        ),
        patch(
            "areal.engine.megatron_utils.packed_context_parallel.mpu.get_context_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "areal.engine.megatron_utils.packed_context_parallel.dist_F.all_gather",
            side_effect=lambda tensor, group=None: [
                tensor if r == 0 else splits[r] for r in range(cp_size)
            ],
        ),
    ):
        reassembled = reassemble_cp_packed_logprobs(splits[0], padded_cu)

    torch.testing.assert_close(reassembled, proj)
    unpadded = unpad_logits(reassembled, 8, padded_cu, old_cu)
    roles = torch.zeros(24, dtype=torch.long, device=device)
    roles[[3, 7, 18, 21]] = ROLE_OPT
    roles[[15, 23]] = ROLE_DECIDE
    labels = torch.full((24,), -1, dtype=torch.long, device=device)
    labels[15] = 1
    labels[23] = 0
    qtypes = torch.full((24,), -1, dtype=torch.long, device=device)
    qtypes[[15, 23]] = 0
    orig = {
        "cu_seqlens": old_cu,
        "token_roles": roles,
        "choice_label": labels,
        "question_type": qtypes,
        "soft_target": torch.full((24,), -1.0, device=device),
    }
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.sft.decision.stats_tracker", tracker):
        loss = compute_packed_decision_loss(unpadded, orig)
    expected = torch.stack(
        [
            question_loss(z0, 1, _CHOICE, None, 0.0),
            question_loss(z1, 0, _CHOICE, None, 0.0),
        ]
    ).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert splits[0].grad is not None
    assert torch.isfinite(splits[0].grad).all()


def test_npu_engine_reassemble_unpad_and_scaled_decision_loss():
    device = _require_npu()
    from areal.engine.megatron_engine import MegatronEngine

    old_cu = torch.tensor([0, 4], dtype=torch.int32, device=device)
    padded_cu = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    z = torch.tensor([1.5, -0.5], device=device)
    padded = torch.zeros(16, 8, dtype=torch.float32, device=device)
    _place_scores(padded, [0, 1], 3, z)
    roles = torch.tensor([ROLE_OPT, ROLE_OPT, 0, ROLE_DECIDE], device=device)
    orig = {
        "cu_seqlens": old_cu,
        "token_roles": roles,
        "choice_label": torch.tensor([-1, -1, -1, 0], device=device),
        "question_type": torch.tensor([-1, -1, -1, _CHOICE], device=device),
        "soft_target": torch.full((4,), -1.0, device=device),
        "_cp_padded_cu_seqlens": padded_cu,
        "_cp_padding_length": 8,
        "_cp_old_cu_seqlens": old_cu,
    }
    engine = object.__new__(MegatronEngine)
    engine.config = SimpleNamespace(is_decision_model=True, is_critic=False)
    engine.enable_tree_training = False

    tracker = DistributedStatsTracker()
    with (
        patch(
            "areal.engine.megatron_engine.reassemble_cp_packed_logprobs",
            side_effect=lambda output, cu: output,
        ),
        patch("areal.trainer.sft.decision.stats_tracker", tracker),
    ):
        output, inputs = MegatronEngine._reassemble_and_unpad_output(
            engine, padded, orig
        )
        loss = MegatronEngine._compute_logprobs_and_loss(
            engine,
            padded,
            orig,
            lambda proj, batch: compute_packed_decision_loss(proj, batch),
            _decision_loss_weight,
            torch.tensor(2.0, device=device),
            loss_multiplier=3.0,
        )

    assert output.shape == (4, 8)
    assert "_cp_padded_cu_seqlens" not in inputs
    expected = question_loss(z, 0, _CHOICE, None, 0.0) * (1.0 / 2.0) * 3.0
    torch.testing.assert_close(loss, expected)


def test_npu_pad_alignment_does_not_create_readout_tokens():
    device = _require_npu()
    roles = torch.tensor([0, ROLE_OPT, ROLE_OPT, ROLE_DECIDE], dtype=torch.long)
    packed = {
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "max_seqlen": 4,
        "input_ids": torch.arange(4, dtype=torch.int32),
        "token_roles": roles,
        "choice_label": torch.tensor([-1, -1, -1, 1], dtype=torch.long),
        "question_type": torch.tensor([-1, -1, -1, _CHOICE], dtype=torch.long),
        "soft_target": torch.tensor([-1.0, 0.2, 0.8, -1.0]),
    }
    padded, pad_len, old_cu, _ = pad_packed_tensor_dict(
        packed, pad_to_length=256, seq_align_to=8
    )
    moved = _move_tensor_container_remembering_cu_seqlens(padded, device)
    assert moved["token_roles"].device.type == "npu"
    assert int((moved["token_roles"] == ROLE_DECIDE).sum().item()) == 1
    assert int((moved["token_roles"] == ROLE_OPT).sum().item()) == 2
    restored = unpad_logits(
        moved["token_roles"].float(),
        pad_len,
        moved["cu_seqlens"],
        old_cu.to(device),
    ).long()
    torch.testing.assert_close(restored.cpu(), roles, rtol=0, atol=0)
    assert pad_len > 0


@pytest.mark.multi_npu
def test_npu_decision_parallel_torchrun(tmp_path):
    from areal.infra.platforms import current_platform
    from areal.utils.network import find_free_ports

    if current_platform.device_type != "npu" or current_platform.device_count() < 2:
        pytest.skip("decision parallel tests require two NPUs")

    port = find_free_ports(1)[0]
    script = (
        pathlib.Path(__file__).parent / "torchrun" / "run_qwen35_decision_parallel.py"
    )
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        str(script),
    ]
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
    env["DECISION_TEST_TMP"] = str(tmp_path)
    try:
        subprocess.run(
            command,
            check=True,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stdout,
            text=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"decision parallel torchrun failed: {exc.returncode}")
    except subprocess.TimeoutExpired:
        pytest.fail("decision parallel torchrun timed out")
