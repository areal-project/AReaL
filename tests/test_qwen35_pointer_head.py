"""Tests for the Qwen3.5 Megatron PointerHead and head.pt helpers.

Unmarked tests run on CPU. Tests marked ``npu`` execute the head on an Ascend
device; the sequence-parallel case launches two NPUs with torchrun.
"""

import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("megatron.core")

from megatron.core.models.gpt.gpt_model import GPTModel

from areal.models.mcore import qwen35_decision as decision
from areal.models.mcore.qwen35_decision import (
    DecisionHeadMeta,
    PointerHead,
    load_decision_pointer_head,
    replace_output_layer_with_pointer_head,
    save_decision_pointer_head,
)


class _Stage(GPTModel):
    """GPTModel subclass that skips Megatron's distributed constructor."""

    def __init__(self, *, post_process: bool, with_output: bool) -> None:
        nn.Module.__init__(self)
        self.post_process = post_process
        self.vocab_size = 16
        if with_output:
            layer = nn.Linear(8, 16, bias=False)
            layer.weight.data = layer.weight.data.to(dtype=torch.bfloat16)
            self.output_layer = layer


class _Wrap(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.module = inner


def _config(*, sequence_parallel: bool) -> SimpleNamespace:
    return SimpleNamespace(hidden_size=8, sequence_parallel=sequence_parallel)


def test_forward_returns_fp32_projections():
    head = PointerHead(8, dp=4)
    hidden = torch.randn(5, 3, 8, dtype=torch.bfloat16)
    projected, extra = head(hidden, weight=torch.zeros(1), runtime_gather_output=True)
    manual = torch.cat([head.q(hidden.float()), head.k(hidden.float())], dim=-1)
    assert extra is None
    assert projected.shape == (5, 3, 8)
    assert projected.dtype == torch.float32
    assert torch.equal(projected, manual)
    assert head.q.weight.dtype == torch.float32


def test_sequence_parallel_marks_q_and_k_and_gathers(monkeypatch):
    seen = {"n": 0}

    def fake_gather(projected, tensor_parallel_output_grad=True, **kwargs):
        del kwargs
        seen["n"] += 1
        seen["grad"] = tensor_parallel_output_grad
        seen["shape"] = tuple(projected.shape)
        return projected + 1

    monkeypatch.setattr(
        decision.tensor_parallel,
        "gather_from_sequence_parallel_region",
        fake_gather,
    )
    head = PointerHead(8, dp=4, config=_config(sequence_parallel=True))
    for linear in (head.q, head.k):
        assert linear.weight.sequence_parallel is True
        assert linear.bias.sequence_parallel is True
    hidden = torch.randn(2, 1, 8)
    projected, _ = head(hidden)
    assert seen["n"] == 1
    assert seen["grad"] is False
    assert seen["shape"] == (2, 1, 8)
    assert torch.equal(
        projected,
        torch.cat([head.q(hidden), head.k(hidden)], dim=-1) + 1,
    )

    plain = PointerHead(8, dp=4, config=_config(sequence_parallel=False))
    assert plain.q.weight.sequence_parallel is False
    plain(torch.randn(2, 1, 8))
    assert seen["n"] == 1


def test_fp32_parameters_survive_parent_bfloat16():
    head = PointerHead(4, dp=2, config=_config(sequence_parallel=True))
    before = head.q.weight.detach().clone()
    parent = nn.Module()
    parent.trunk = nn.Linear(4, 4)
    parent.head = head
    parent.bfloat16()
    assert parent.trunk.weight.dtype == torch.bfloat16
    assert head.q.weight.dtype == torch.float32
    assert head.k.bias.dtype == torch.float32
    assert torch.equal(head.q.weight, before)
    assert head.q.weight.sequence_parallel is True
    assert head.k.bias.sequence_parallel is True


def test_scores_match_dot_product_and_temperature():
    torch.manual_seed(0)
    head = PointerHead(6, dp=3)
    q_decide = torch.randn(3)
    k_opts = torch.randn(5, 3)
    head.train()
    head.temperature = 2.0
    raw = (k_opts * q_decide).sum(-1) * head.scale
    assert torch.equal(head.score_single(q_decide, k_opts), raw)

    queries = torch.randn(2, 3)
    owner = torch.tensor([0, 1, 1, 0])
    many_raw = (k_opts[:4] * queries[owner]).sum(-1) * head.scale
    assert torch.equal(head.score_many(queries, k_opts[:4], owner), many_raw)

    head.eval()
    assert torch.allclose(head.score_single(q_decide, k_opts), raw / 2)


def test_replace_output_layer_only_on_last_stage():
    last = _Stage(post_process=True, with_output=True)
    head = replace_output_layer_with_pointer_head(
        last, _config(sequence_parallel=True), head_dim=4
    )
    assert isinstance(head, PointerHead)
    assert last.output_layer is head
    assert last.vocab_size == 8
    assert head.q.weight.dtype == torch.float32
    assert head.sequence_parallel is True

    wrapped = _Wrap(_Stage(post_process=True, with_output=True))
    wrapped_head = replace_output_layer_with_pointer_head(
        wrapped, _config(sequence_parallel=False), head_dim=4
    )
    assert wrapped.module.output_layer is wrapped_head

    early = _Stage(post_process=False, with_output=True)
    original = early.output_layer
    assert (
        replace_output_layer_with_pointer_head(
            early, _config(sequence_parallel=False), head_dim=4
        )
        is None
    )
    assert early.output_layer is original
    assert early.vocab_size == 16

    missing = _Stage(post_process=True, with_output=False)
    assert (
        replace_output_layer_with_pointer_head(
            missing, _config(sequence_parallel=False), head_dim=4
        )
        is None
    )
    assert missing.vocab_size == 16


def test_head_pt_round_trip(tmp_path):
    torch.manual_seed(2)
    stage = _Stage(post_process=True, with_output=True)
    head = replace_output_layer_with_pointer_head(
        stage, _config(sequence_parallel=False), head_dim=4
    )
    assert head is not None
    head.temperature = 1.0
    meta = DecisionHeadMeta(
        base="Qwen/Qwen3.5-4B",
        base_revision="abc",
        lora=16,
        weights="lora",
        weights_dtype="bf16",
        temperature=2.25,
        holdout=["dev"],
        extra={"suite_sha256": "deadbeef", "args": {"lr": 1e-4}},
    )
    path = save_decision_pointer_head(stage, str(tmp_path), meta=meta)
    assert path == os.path.join(tmp_path, "head.pt")

    fresh = _Stage(post_process=True, with_output=True)
    fresh_head = replace_output_layer_with_pointer_head(
        fresh, _config(sequence_parallel=False), head_dim=4
    )
    assert fresh_head is not None
    with torch.no_grad():
        fresh_head.q.weight.fill_(0)
        fresh_head.k.bias.fill_(0)
    loaded = load_decision_pointer_head(fresh, str(tmp_path))
    assert loaded is not None
    assert loaded.base == "Qwen/Qwen3.5-4B"
    assert loaded.base_revision == "abc"
    assert loaded.lora == 16
    assert loaded.weights == "lora"
    assert loaded.weights_dtype == "bf16"
    assert loaded.head_dim == 4
    assert loaded.holdout == ["dev"]
    assert loaded.extra["suite_sha256"] == "deadbeef"
    assert loaded.temperature == 2.25
    assert fresh_head.temperature == 2.25
    for key, value in head.state_dict().items():
        assert torch.equal(fresh_head.state_dict()[key], value.float())
        assert loaded.head[key].dtype == torch.float32

    payload = torch.load(path, map_location="cpu", weights_only=False)
    again = DecisionHeadMeta.from_dict(payload)
    assert again.base == meta.base
    assert again.lora == 16
    assert again.weights == "lora"
    assert again.head_dim == 4
    assert again.temperature == 2.25
    assert again.extra["suite_sha256"] == "deadbeef"
    assert again.to_dict()["args"] == {"lr": 1e-4}
    assert torch.equal(again.head["q.weight"], head.q.weight.detach().cpu())


def test_pointer_head_pt_raw_state_dict(tmp_path):
    head = PointerHead(8, dp=4)
    raw = os.path.join(tmp_path, "pointer_head.pt")
    torch.save(head.state_dict(), raw)
    stage = _Stage(post_process=True, with_output=True)
    loaded_head = replace_output_layer_with_pointer_head(
        stage, _config(sequence_parallel=False), head_dim=4
    )
    assert loaded_head is not None
    with torch.no_grad():
        loaded_head.q.weight.normal_()
    loaded_head.temperature = 1.7
    meta = load_decision_pointer_head(stage, str(tmp_path))
    assert meta is not None
    assert meta.head_dim == 4
    assert loaded_head.temperature == 1.7
    for key, value in head.state_dict().items():
        assert torch.equal(loaded_head.state_dict()[key], value)


def test_missing_checkpoint_keeps_initialization(tmp_path):
    stage = _Stage(post_process=True, with_output=True)
    head = replace_output_layer_with_pointer_head(
        stage, _config(sequence_parallel=False), head_dim=4
    )
    assert head is not None
    before = head.q.weight.detach().clone()
    assert load_decision_pointer_head(stage, str(tmp_path)) is None
    assert torch.equal(head.q.weight, before)


def test_replica_groups_add_data_parallel_for_fresh_init(monkeypatch):
    tp, cp, dp = object(), object(), object()
    monkeypatch.setattr(decision, "_mpu_ready", lambda: True)
    monkeypatch.setattr(decision.mpu, "get_tensor_model_parallel_group", lambda: tp)
    monkeypatch.setattr(decision.mpu, "get_context_parallel_group", lambda: cp)
    monkeypatch.setattr(
        decision.mpu,
        "get_data_parallel_group",
        lambda with_context_parallel=False: dp,
    )
    monkeypatch.setattr(decision, "_group_world_size", lambda group: 2)
    assert decision._replica_groups(include_data_parallel=False) == [tp, cp]
    assert decision._replica_groups(include_data_parallel=True) == [tp, cp, dp]


def test_broadcast_sends_parameters_and_temperature(monkeypatch):
    calls = []
    monkeypatch.setattr(decision, "_distributed", lambda: True)

    def get_global_rank(group, rank):
        del rank
        return 4 if group == "tp" else 9

    def broadcast(tensor, src, group):
        calls.append((src, group, tuple(tensor.shape), float(tensor.flatten()[0])))

    monkeypatch.setattr(decision.dist, "get_global_rank", get_global_rank)
    monkeypatch.setattr(decision.dist, "broadcast", broadcast)
    head = PointerHead(4, dp=2)
    head.temperature = 3.0
    decision._broadcast_head(head, ["tp"])
    # q.weight, q.bias, k.weight, k.bias, temperature
    assert len(calls) == 5
    assert {src for src, group, _, _ in calls} == {4}
    assert calls[-1][2] == (1,)
    assert calls[-1][3] == 3.0


def _require_npu():
    from areal.infra.platforms import current_platform

    if current_platform.device_type != "npu" or current_platform.device_count() < 1:
        pytest.skip("NPU is not available")
    current_platform.set_device(0)
    return torch.device("npu", 0)


@pytest.mark.npu
def test_npu_forward_backward_and_scores():
    device = _require_npu()
    torch.manual_seed(0)
    head = PointerHead(8, dp=4).to(device)
    hidden = torch.randn(5, 3, 8, dtype=torch.bfloat16, device=device)
    projected, extra = head(hidden)
    manual = torch.cat([head.q(hidden.float()), head.k(hidden.float())], dim=-1)
    assert extra is None
    assert projected.shape == (5, 3, 8)
    assert projected.dtype == torch.float32
    assert projected.device.type == "npu"
    torch.testing.assert_close(projected, manual)

    projected.sum().backward()
    assert head.q.weight.grad is not None
    assert head.q.weight.grad.device.type == "npu"
    assert head.q.weight.grad.dtype == torch.float32
    assert torch.isfinite(head.q.weight.grad).all()

    head.zero_grad()
    q_decide = torch.randn(4, device=device)
    k_opts = torch.randn(5, 4, device=device)
    head.temperature = 2.0
    head.train()
    raw = (k_opts * q_decide).sum(-1) * head.scale
    torch.testing.assert_close(head.score_single(q_decide, k_opts), raw)
    head.eval()
    torch.testing.assert_close(head.score_single(q_decide, k_opts), raw / 2)


@pytest.mark.npu
def test_npu_fp32_parameters_survive_move_and_bfloat16():
    device = _require_npu()
    head = PointerHead(4, dp=2, config=_config(sequence_parallel=True))
    before = head.q.weight.detach().clone()
    parent = nn.Module()
    parent.trunk = nn.Linear(4, 4)
    parent.head = head
    parent.to(device)
    assert head.q.weight.device.type == "npu"
    assert head.q.weight.dtype == torch.float32
    parent.bfloat16()
    assert parent.trunk.weight.dtype == torch.bfloat16
    assert parent.trunk.weight.device.type == "npu"
    assert head.q.weight.dtype == torch.float32
    assert head.k.bias.dtype == torch.float32
    assert head.q.weight.device.type == "npu"
    assert torch.equal(head.q.weight.detach().cpu(), before)
    assert head.q.weight.sequence_parallel is True
    # The gather itself needs a tensor-parallel group; the two-NPU test covers it.
    head.sequence_parallel = False

    hidden = torch.randn(2, 1, 4, dtype=torch.bfloat16, device=device)
    projected, _ = head(hidden)
    assert projected.dtype == torch.float32
    assert projected.shape == (2, 1, 4)


@pytest.mark.npu
def test_npu_replace_and_head_pt_round_trip(tmp_path):
    device = _require_npu()
    stage = _Stage(post_process=True, with_output=True)
    stage.output_layer = stage.output_layer.to(device)
    head = replace_output_layer_with_pointer_head(
        stage, _config(sequence_parallel=False), head_dim=4
    )
    assert head is not None
    assert head.q.weight.device.type == "npu"
    assert head.q.weight.dtype == torch.float32
    with torch.no_grad():
        head.q.weight.normal_()
        head.k.bias.normal_()
    path = save_decision_pointer_head(
        stage,
        str(tmp_path),
        meta=DecisionHeadMeta(
            base="Qwen/Qwen3.5-4B",
            weights="full",
            lora=0,
            temperature=1.5,
        ),
    )
    assert path is not None

    fresh = _Stage(post_process=True, with_output=True)
    fresh.output_layer = fresh.output_layer.to(device)
    loaded = replace_output_layer_with_pointer_head(
        fresh, _config(sequence_parallel=False), head_dim=4
    )
    assert loaded is not None
    with torch.no_grad():
        loaded.q.weight.zero_()
    meta = load_decision_pointer_head(fresh, str(tmp_path))
    assert meta is not None
    assert meta.weights == "full"
    assert loaded.temperature == 1.5
    assert loaded.q.weight.device.type == "npu"
    torch.testing.assert_close(loaded.q.weight, head.q.weight)
    torch.testing.assert_close(loaded.k.bias, head.k.bias)


@pytest.mark.npu
@pytest.mark.multi_npu
def test_npu_sequence_parallel_gather():
    from areal.infra.platforms import current_platform
    from areal.utils.network import find_free_ports

    if current_platform.device_type != "npu" or current_platform.device_count() < 2:
        pytest.skip("sequence-parallel gather requires two NPUs")

    port = find_free_ports(1)[0]
    script = (
        pathlib.Path(__file__).parent / "torchrun" / "run_qwen35_pointer_head_sp.py"
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
        pytest.fail(f"PointerHead sequence-parallel torchrun failed: {exc.returncode}")
    except subprocess.TimeoutExpired:
        pytest.fail("PointerHead sequence-parallel torchrun timed out")
