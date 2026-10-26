# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 decision unit tests: GDN unfuse, PointerHead, loss, checkpoints."""

from __future__ import annotations

import math
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel
from torch import nn

from areal.dataset.decision import (
    QUESTION_TYPE_IDS,
    ROLE_DECIDE,
    ROLE_OPT,
    SPECIAL,
    encode,
    encode_question_rows,
    rows_of,
    select_labelled_records,
)
from areal.engine.megatron_utils.megatron_bridge_patches import (
    patch_qwen35_hybrid_lora_specs,
)
from areal.models.mcore.qwen35_decision import (
    DecisionHeadMeta,
    PointerHead,
    load_decision_pointer_head,
    replace_output_layer_with_pointer_head,
    save_decision_pointer_head,
)
from areal.trainer.sft.decision import compute_packed_decision_loss, question_loss
from areal.utils.stats_tracker import DistributedStatsTracker

_CHOICE = QUESTION_TYPE_IDS["choice"]
_SCORE = QUESTION_TYPE_IDS["score"]


class _Tok:
    """Minimal tokenizer: specials are 1..5, other text is 100 + ord(c)."""

    def __init__(self):
        self._ids = {token: i + 1 for i, token in enumerate(SPECIAL)}

    def convert_tokens_to_ids(self, token):
        return self._ids[token]

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[100 + ord(ch) for ch in text])


class _Stage(GPTModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.post_process = True
        self.vocab_size = 16
        self.output_layer = nn.Linear(8, 16, bias=False)


class _Fused:
    """Sentinel for a fused Linear+Norm spec entry."""


def _ensure_module(monkeypatch, name: str) -> ModuleType:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        qual = ".".join(parts[:i])
        if qual not in sys.modules:
            monkeypatch.setitem(sys.modules, qual, ModuleType(qual))
        if i > 1:
            parent = sys.modules[".".join(parts[: i - 1])]
            child = parts[i - 1]
            if getattr(parent, child, None) is not sys.modules[qual]:
                monkeypatch.setattr(parent, child, sys.modules[qual], raising=False)
    return sys.modules[name]


def _hybrid_lora_classes(monkeypatch):
    """Make the three patch() imports succeed; return the classes it will bind."""
    try:
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
    except ImportError:
        layers = _ensure_module(monkeypatch, "megatron.core.tensor_parallel.layers")
        monkeypatch.setattr(
            layers, "ColumnParallelLinear", type("ColumnParallelLinear", (), {})
        )
        monkeypatch.setattr(
            layers, "RowParallelLinear", type("RowParallelLinear", (), {})
        )
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

    try:
        from mindspeed.core.megatron_basic.megatron_basic import PTNorm
    except ImportError:
        basic = _ensure_module(
            monkeypatch, "mindspeed.core.megatron_basic.megatron_basic"
        )
        monkeypatch.setattr(basic, "PTNorm", type("PTNorm", (), {}))
        from mindspeed.core.megatron_basic.megatron_basic import PTNorm

    try:
        from megatron.bridge.models.qwen_vl import qwen35_vl_provider
    except ImportError:
        pkg = _ensure_module(monkeypatch, "megatron.bridge.models.qwen_vl")
        provider = _ensure_module(
            monkeypatch, "megatron.bridge.models.qwen_vl.qwen35_vl_provider"
        )
        if not hasattr(
            provider, "get_transformer_block_with_experimental_attention_variant_spec"
        ):
            provider.get_transformer_block_with_experimental_attention_variant_spec = (
                lambda *a, **k: None
            )
        monkeypatch.setattr(pkg, "qwen35_vl_provider", provider, raising=False)
        from megatron.bridge.models.qwen_vl import qwen35_vl_provider

    return ColumnParallelLinear, RowParallelLinear, PTNorm, qwen35_vl_provider


def _layer(kind: str):
    if kind == "full":
        attn = SimpleNamespace(
            metainfo={"fuse_input_layernorm": True},
            submodules=SimpleNamespace(linear_qkv=_Fused, linear_proj=_Fused),
        )
    else:
        attn = SimpleNamespace(
            metainfo={"fuse_input_layernorm": True},
            submodules=SimpleNamespace(in_proj=_Fused, out_proj=_Fused),
        )
    mlp = SimpleNamespace(
        metainfo={"fuse_pre_mlp_layernorm": True},
        submodules=SimpleNamespace(linear_fc1=_Fused, linear_fc2=_Fused),
    )
    return SimpleNamespace(
        submodules=SimpleNamespace(
            self_attention=attn,
            mlp=mlp,
            input_layernorm=_Fused,
            pre_mlp_layernorm=_Fused,
        )
    )


def _hybrid_block():
    return SimpleNamespace(layer_specs=[_layer("full"), _layer("gdn")])


def _patched_block(monkeypatch, *, unfuse_gdn: bool):
    col, row, norm, provider = _hybrid_lora_classes(monkeypatch)
    monkeypatch.setattr(
        provider,
        "get_transformer_block_with_experimental_attention_variant_spec",
        lambda *a, **k: _hybrid_block(),
    )
    patch_qwen35_hybrid_lora_specs(unfuse_gdn=unfuse_gdn)
    block = provider.get_transformer_block_with_experimental_attention_variant_spec()
    return block, col, row, norm


def test_patch_unfuses_full_attn_mlp_and_gdn(monkeypatch):
    block, col, row, norm = _patched_block(monkeypatch, unfuse_gdn=True)
    full, gdn = block.layer_specs
    assert full.submodules.input_layernorm is norm
    assert full.submodules.self_attention.submodules.linear_qkv is col
    assert full.submodules.self_attention.metainfo["fuse_input_layernorm"] is False
    assert full.submodules.pre_mlp_layernorm is norm
    assert full.submodules.mlp.submodules.linear_fc1 is col
    assert full.submodules.mlp.metainfo["fuse_pre_mlp_layernorm"] is False

    assert gdn.submodules.input_layernorm is norm
    assert gdn.submodules.self_attention.submodules.in_proj is col
    assert gdn.submodules.self_attention.submodules.out_proj is row
    assert gdn.submodules.self_attention.metainfo["fuse_input_layernorm"] is False
    assert gdn.submodules.mlp.submodules.linear_fc1 is col


def test_patch_keeps_gdn_fused_when_disabled(monkeypatch):
    block, col, row, norm = _patched_block(monkeypatch, unfuse_gdn=False)
    del row
    full, gdn = block.layer_specs
    assert full.submodules.self_attention.submodules.linear_qkv is col
    assert full.submodules.input_layernorm is norm
    assert gdn.submodules.self_attention.submodules.in_proj is _Fused
    assert gdn.submodules.self_attention.submodules.out_proj is _Fused
    assert gdn.submodules.input_layernorm is _Fused
    assert gdn.submodules.self_attention.metainfo["fuse_input_layernorm"] is True
    assert gdn.submodules.mlp.submodules.linear_fc1 is col


def test_pointer_head_forward_shape_and_grad():
    torch.manual_seed(0)
    head = PointerHead(8, dp=4)
    hidden = torch.randn(5, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    projected, extra = head(hidden)
    assert extra is None
    assert projected.shape == (5, 3, 8)
    assert projected.dtype == torch.float32
    assert projected.reshape(-1, 8).shape == (15, 2 * 4)
    projected.sum().backward()
    assert head.q.weight.grad is not None
    assert head.k.weight.grad is not None
    assert torch.isfinite(head.q.weight.grad).all()
    assert hidden.grad is not None


def test_pointer_head_scores_are_scaled_dots():
    torch.manual_seed(1)
    head = PointerHead(8, dp=4)
    decide = torch.randn(8)
    options = torch.randn(6, 8)
    head.train()
    q_one = head.q(decide)
    k_one = head.k(options)
    torch.testing.assert_close(
        head.score_single(q_one, k_one),
        (k_one * q_one).sum(-1) * head.scale,
    )
    owner = torch.tensor([0, 0, 1, 1, 1])
    queries = torch.randn(2, 8)
    q_many = head.q(queries)
    k_many = head.k(options[:5])
    torch.testing.assert_close(
        head.score_many(q_many, k_many, owner),
        (k_many * q_many[owner]).sum(-1) * head.scale,
    )
    head.temperature = 2.5
    head.eval()
    torch.testing.assert_close(
        head.score_single(q_one, k_one),
        (k_one * q_one).sum(-1) * head.scale / 2.5,
    )


def _internal_record():
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


def test_encode_rows_mark_decide_and_opts():
    tok = _Tok()
    rec = _internal_record()
    enc = encode(tok, rec)
    state_id, q_id, opt_open, opt_close, decide_id = (
        tok.convert_tokens_to_ids(token) for token in SPECIAL
    )
    del q_id, opt_open
    assert enc["ids"][0] == state_id
    for idx in enc["decide_idx"]:
        assert enc["ids"][idx] == decide_id
    for opts in enc["opt_idx"]:
        assert all(enc["ids"][i] == opt_close for i in opts)

    state, _, rows = rows_of(enc)
    assert state[0] == state_id
    assert len(rows) == len(rec["questions"])
    for row, question in zip(rows, rec["questions"]):
        assert row["ids"][row["decide"]] == decide_id
        assert len(row["opts"]) == len(question["options"])
        assert all(row["ids"][i] == opt_close for i in row["opts"])

    packed = encode_question_rows(tok, rec)
    assert packed[0]["token_roles"].count(ROLE_DECIDE) == 1
    assert packed[0]["token_roles"].count(ROLE_OPT) == 2
    assert packed[0]["choice_label"][packed[0]["token_roles"].index(ROLE_DECIDE)] == 1
    assert packed[1]["soft_target"][packed[1]["token_roles"].index(ROLE_OPT)] == 0.25


def _proj_from_logits(z: torch.Tensor, dp: int = 8) -> torch.Tensor:
    scale = 1.0 / math.sqrt(dp)
    q_all = torch.zeros(z.numel() + 1, dp)
    k_all = torch.zeros(z.numel() + 1, dp)
    q_all[-1, 0] = 1.0
    k_all[:-1, 0] = z.float() / scale
    return torch.cat([q_all, k_all], dim=-1)


def _pack_question(z, label, qtype, soft=None, dp=8):
    k = z.numel()
    roles = torch.tensor([ROLE_OPT] * k + [ROLE_DECIDE])
    choice = torch.full((k + 1,), -1)
    choice[-1] = label
    qtypes = torch.full((k + 1,), -1)
    qtypes[-1] = qtype
    if soft is None:
        target = torch.full((k + 1,), -1.0)
    else:
        target = torch.tensor(list(soft) + [-1.0], dtype=torch.float32)
    return _proj_from_logits(z, dp), {
        "cu_seqlens": torch.tensor([0, k + 1], dtype=torch.int32),
        "token_roles": roles,
        "choice_label": choice,
        "question_type": qtypes,
        "soft_target": target,
    }


@pytest.mark.parametrize(
    "qtype,opts",
    [
        (_CHOICE, {}),
        (_CHOICE, {"label_smoothing": 0.05}),
        (_CHOICE, {"brier_w": 0.5}),
        (_CHOICE, {"focal_gamma": 1.0}),
        (_SCORE, {}),
    ],
)
def test_question_loss_matches_definitions(qtype, opts):
    z = torch.tensor([0.4, -0.7, 1.2])
    y = torch.tensor([1])
    expected = F.cross_entropy(
        z[None], y, label_smoothing=opts.get("label_smoothing", 0.0)
    )
    if opts.get("brier_w"):
        target = F.one_hot(y[0], 3).float()
        expected = expected + opts["brier_w"] * (z.softmax(-1) - target).square().sum()
    if opts.get("focal_gamma"):
        expected = (1 - torch.exp(-expected)).pow(opts["focal_gamma"]) * expected
    if qtype == _SCORE:
        p = z.softmax(-1)
        observed = (torch.arange(2) >= 1).to(p.dtype)
        expected = expected + 0.5 * (p.cumsum(-1)[:-1] - observed).square().mean()
    actual = question_loss(z, 1, qtype, None, 0.5, **opts)
    torch.testing.assert_close(actual, expected)


def test_soft_target_and_packed_loss_match_definitions():
    z = torch.tensor([2.0, 0.0, -2.0])
    t = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    expected = -(t * F.log_softmax(z, -1)).sum()
    actual = question_loss(z, 0, _CHOICE, t, 0.0, brier_w=0.5)
    torch.testing.assert_close(actual, expected)

    proj, input_ = _pack_question(z, 0, _CHOICE, soft=t.tolist())
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.sft.decision.stats_tracker", tracker):
        packed = compute_packed_decision_loss(proj, input_, brier_w=0.5)
    torch.testing.assert_close(packed, expected)


def test_packed_loss_decreases_on_synthetic_step():
    z = torch.tensor([0.1, 0.0, -0.2])
    proj, input_ = _pack_question(z, 0, _CHOICE)
    proj = proj.clone().requires_grad_(True)
    opt = torch.optim.SGD([proj], lr=0.5)
    tracker = DistributedStatsTracker()
    losses = []
    for _ in range(8):
        opt.zero_grad()
        with patch("areal.trainer.sft.decision.stats_tracker", tracker):
            loss = compute_packed_decision_loss(proj, input_)
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
        tracker.export(reset=True)
    assert losses[-1] < losses[0]
    expected = F.cross_entropy(z[None], torch.tensor([0]))
    with patch("areal.trainer.sft.decision.stats_tracker", tracker):
        first = compute_packed_decision_loss(_proj_from_logits(z), input_)
    torch.testing.assert_close(first, expected)


def test_head_checkpoint_round_trip(tmp_path):
    torch.manual_seed(2)
    stage = _Stage()
    head = replace_output_layer_with_pointer_head(
        stage, SimpleNamespace(hidden_size=8, sequence_parallel=False), head_dim=4
    )
    assert head is not None
    head.temperature = 1.75
    path = save_decision_pointer_head(
        stage,
        str(tmp_path),
        meta=DecisionHeadMeta(
            base="Qwen/Qwen3.5-4B",
            lora=16,
            weights="lora",
            temperature=1.75,
        ),
    )
    assert path == os.path.join(tmp_path, "head.pt")

    fresh = _Stage()
    loaded = replace_output_layer_with_pointer_head(
        fresh, SimpleNamespace(hidden_size=8, sequence_parallel=False), head_dim=4
    )
    assert loaded is not None
    with torch.no_grad():
        loaded.q.weight.zero_()
        loaded.k.bias.zero_()
    meta = load_decision_pointer_head(fresh, str(tmp_path))
    assert meta is not None
    assert meta.base == "Qwen/Qwen3.5-4B"
    assert meta.lora == 16
    assert meta.weights == "lora"
    assert meta.head_dim == 4
    assert loaded.temperature == 1.75
    for key, value in head.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value.float())

    payload = torch.load(path, map_location="cpu", weights_only=False)
    again = DecisionHeadMeta.from_dict(payload)
    assert again.base == "Qwen/Qwen3.5-4B"
    assert again.head_dim == 4
    assert again.lora == 16
    assert torch.equal(again.head["q.weight"], head.q.weight.detach().cpu())


def _require_npu():
    from areal.infra.platforms import current_platform

    if current_platform.device_type != "npu" or current_platform.device_count() < 1:
        pytest.skip("NPU is not available")
    current_platform.set_device(0)
    return torch.device("npu", 0)


@pytest.mark.npu
def test_npu_pointer_head_forward_backward():
    device = _require_npu()
    torch.manual_seed(0)
    head = PointerHead(8, dp=4).to(device)
    hidden = torch.randn(
        5, 3, 8, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    projected, extra = head(hidden)
    assert extra is None
    assert projected.shape == (5, 3, 8)
    assert projected.dtype == torch.float32
    assert projected.device.type == "npu"
    projected.sum().backward()
    assert head.q.weight.grad is not None
    assert head.q.weight.grad.device.type == "npu"
    assert torch.isfinite(head.q.weight.grad).all()
    assert hidden.grad is not None


@pytest.mark.npu
def test_npu_packed_loss_backward():
    device = _require_npu()
    z = torch.tensor([0.4, -0.7, 1.2], device=device)
    proj, input_ = _pack_question(z.cpu(), 1, _CHOICE)
    proj = proj.to(device).requires_grad_(True)
    input_ = {key: value.to(device) for key, value in input_.items()}
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.sft.decision.stats_tracker", tracker):
        loss = compute_packed_decision_loss(proj, input_)
    assert loss.device.type == "npu"
    loss.backward()
    assert proj.grad is not None
    assert torch.isfinite(proj.grad).all()
    expected = F.cross_entropy(z[None], torch.tensor([1], device=device))
    torch.testing.assert_close(loss, expected)


@pytest.mark.npu
def test_npu_head_checkpoint_round_trip(tmp_path):
    device = _require_npu()
    stage = _Stage()
    stage.output_layer = stage.output_layer.to(device)
    head = replace_output_layer_with_pointer_head(
        stage, SimpleNamespace(hidden_size=8, sequence_parallel=False), head_dim=4
    )
    assert head is not None
    assert head.q.weight.device.type == "npu"
    path = save_decision_pointer_head(stage, str(tmp_path))
    assert path is not None

    fresh = _Stage()
    fresh.output_layer = fresh.output_layer.to(device)
    loaded = replace_output_layer_with_pointer_head(
        fresh, SimpleNamespace(hidden_size=8, sequence_parallel=False), head_dim=4
    )
    assert loaded is not None
    with torch.no_grad():
        loaded.q.weight.zero_()
    meta = load_decision_pointer_head(fresh, str(tmp_path))
    assert meta is not None
    torch.testing.assert_close(loaded.q.weight, head.q.weight)
    assert loaded.q.weight.device.type == "npu"


def test_select_labelled_records_caps_per_source_then_total():
    records = [
        {"_meta": {"source": "yelp"}, "i": 0},
        {"_meta": {"source": "yelp"}, "i": 1},
        {"_meta": {"source": "yelp"}, "i": 2},
        {"_meta": {"source": "boolq"}, "i": 3},
        {"_meta": {"source": "boolq"}, "i": 4},
        {"state": "x", "i": 5},
    ]
    sliced = select_labelled_records(records, n_per_source=2)
    assert [row["i"] for row in sliced] == [0, 1, 3, 4, 5]
    assert [row["i"] for row in select_labelled_records(sliced, max_records=3)] == [
        0,
        1,
        3,
    ]
