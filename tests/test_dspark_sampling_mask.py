# SPDX-License-Identifier: Apache-2.0
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest
import torch

from areal.engine.sglang_dspark_policy import (
    CommittedMaskRows,
    VerifyPolicy,
    prepare_verify_policy,
)
from areal.engine.sglang_sampling_mask import candidate_distribution


@pytest.fixture
def dspark_runtime():
    try:
        installed = version("sglang")
    except PackageNotFoundError:
        pytest.skip("SGLang is not installed")
    if installed != "0.5.20":
        pytest.skip("DSpark verify requires SGLang 0.5.20")


class RecordingProducer:
    def publish(self, ids, counts, tokens, versions, leases):
        self.tokens = tokens.tolist()
        self.supports = [row[:n].tolist() for row, n in zip(ids, counts, strict=True)]
        self.versions = versions
        self.leases = leases
        return list(range(1, len(tokens) + 1))


def make_policy(device):
    # Row 0 distinguishes joint filtering from nucleus sampling after a top-k renorm.
    x = -torch.arange(128, device=device).float()
    logits = torch.stack(
        [(x * a).roll(i) for i, a in enumerate([0.004, 0.11, 0.22] * 3)]
    )
    info = SimpleNamespace(
        temperatures=x.new_tensor([[1.0], [0.7], [1.25]]),
        top_ps=x.new_full((3,), 0.95),
        top_ks=torch.full((3,), 50, device=device, dtype=torch.int32),
    )
    producer = RecordingProducer()
    policy = prepare_verify_policy(
        logits, info, 3, 50, producer, [4, 5, 6], ["a", "b", "c"]
    )
    return logits, info, policy, producer


def test_dspark_target_matches_full_vocab_joint_and_decode():
    logits, info, policy, _ = make_policy("cpu")
    p = torch.zeros_like(logits).scatter_(
        1, policy.rows.ids.long(), policy.logprobs.exp()
    )
    for row in range(len(logits)):
        req = row // 3
        scaled = logits[row].double() / info.temperatures[req]
        ranks = scaled.argsort(descending=True)
        mass = scaled[ranks].softmax(0)
        support = ranks[
            (mass.cumsum(0) - mass <= info.top_ps[req]) & (torch.arange(len(mass)) < 50)
        ]
        assert set(p[row].nonzero().flatten().tolist()) == set(support.tolist())
        torch.testing.assert_close(
            p[row, support].log().double(),
            scaled[support].log_softmax(0),
            atol=1e-5,
            rtol=0,
        )
        ids, lp, counts = candidate_distribution(
            logits[row : row + 1],
            info.temperatures[req : req + 1],
            info.top_ps[req : req + 1],
            info.top_ks[req : req + 1],
            candidate_bound=50,
        )
        assert torch.equal(ids[0], policy.rows.ids[row])
        assert torch.equal(lp[0], policy.logprobs[row])
        assert counts.item() == policy.rows.counts[row].item()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DSpark chain kernel requires CUDA"
)
@pytest.mark.parametrize("reject", [True, False])
@pytest.mark.parametrize("k", [4, 2049])
def test_dspark_sparse_final_draw(reject, k):
    from areal.engine.sglang_dspark_sampling import sample_chain

    device = "cuda"
    # Check every quantile of p and (p-q)+, including the two attainable RNG
    # endpoints, with zero mass on both a retained ID and a padded column.
    n = 256
    support = torch.tensor([3, 7, 5, 1], device=device)
    prob = torch.tensor([0.5, 0.375, 0.125, 0.0], device=device)
    if k > 4:
        # The last positive interval lies in a different kernel tile.
        support = torch.arange(k, device=device)
        support[[0, 3, 1, 7, 5, k - 1]] = support[[3, 0, 7, 1, k - 1, 5]]
        prob = torch.zeros(k, device=device)
        prob[[0, 1, k - 1]] = torch.tensor([0.5, 0.375, 0.125], device=device)
    ids = support.repeat(2 * n, 1)
    policy = VerifyPolicy(
        prob.log().repeat(2 * n, 1), CommittedMaskRows(None, ids, None, [], [], 2)
    )
    draft = torch.zeros(n, 1, max(8, k), device=device)
    draft[:, :, 3] = 0.75
    draft[:, :, 0] = 0.25
    proposals = torch.full((n, 1), 3, device=device)
    u = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
    u[0] = 0.0
    u[-1] = 1.0 - 2**-24
    coins = torch.stack([torch.full_like(u, 0.9 if reject else 0.25), u], dim=1)
    accepted, final = sample_chain(proposals, policy, draft, coins)
    assert (accepted == (0 if reject else 1)).all()
    if reject:
        expected = torch.where(u < 0.75, 7, 5)
    else:
        expected = torch.where(u < 0.5, 3, torch.where(u < 0.875, 7, 5))
    assert torch.equal(final, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DSpark chain kernel requires CUDA"
)
def test_dspark_accept_replacement_bonus_and_visible_commits(dspark_runtime):
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
        BuildOutTokens,
    )
    from sglang.srt.managers.auxiliary_output import CommittedTokens

    from areal.engine.sglang_dspark_worker import accept_joint_chain

    logits, info, policy, producer = make_policy("cuda")
    p = (
        torch.zeros_like(logits)
        .scatter_(1, policy.rows.ids.long(), policy.logprobs.exp())
        .view(3, 3, -1)
    )
    draft = p[:, :2].contiguous().clone()
    proposed = draft.argmax(-1)
    for b, j in [(0, 0), (1, 1)]:
        # Zero target mass at proposed ID guarantees rejection in this slot.
        proposed[b, j] = (proposed[b, j] + 64) % 128
        draft[b, j].zero_().scatter_(0, proposed[b, j : j + 1], 1.0)
    candidates = torch.cat([torch.zeros_like(proposed[:, :1]), proposed], dim=1)
    accepted, bonus, _ = accept_joint_chain(candidates, policy, draft)
    assert accepted.tolist() == [0, 1, 2]
    tokens = BuildOutTokens.execute(
        draft_tokens=proposed,
        correct_len=accepted,
        bonus=bonus,
        verify_num_draft_tokens=3,
        gamma=2,
    ).view(3, 3)
    out = SimpleNamespace()
    policy.finish(out, tokens, accepted + 1)
    reqs = [SimpleNamespace(customized_info=None) for _ in range(3)]
    # EOS trims the second request to one visible token; last request includes the bonus.
    lens = [1, 1, 3]
    commits = [
        CommittedTokens(1, tuple(tokens[b, :n].tolist())) for b, n in enumerate(lens)
    ]
    out.auxiliary_device_output.copy_to_host(None).consume(
        SimpleNamespace(reqs=reqs), commits
    )
    assert producer.versions == [4, 5, 6, 6, 6]
    assert producer.leases == ["a", "b", "c", "c", "c"]
    assert [r.customized_info["sampling_mask_refs"] for r in reqs] == [
        [1],
        [2],
        [3, 4, 5],
    ]
    assert all(t in s for t, s in zip(producer.tokens, producer.supports, strict=True))
    for b in range(3):
        n = accepted[b].item() + 1
        rows = torch.arange(n, device=p.device)
        torch.testing.assert_close(
            out.next_token_logprobs[b, :n],
            p[b, rows, tokens[b, :n].long()].log(),
            atol=1e-5,
            rtol=0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DSpark chain kernel requires CUDA"
)
def test_dspark_capped_verified_prefix_keeps_next_accepted_token(dspark_runtime):
    from areal.engine.sglang_dspark_worker import accept_joint_chain

    logits, _, policy, _ = make_policy("cuda")
    draft = (
        torch.zeros_like(logits)
        .scatter_(1, policy.rows.ids.long(), policy.logprobs.exp())
        .view(3, 3, -1)[:, :-1]
    )
    proposals = draft.argmax(-1)
    candidates = torch.cat([torch.zeros_like(proposals[:, :1]), proposals], dim=1)
    accepted, bonus, trimmed = accept_joint_chain(
        candidates,
        policy,
        draft,
        verify_lens=torch.tensor([1, 2, 3], device=draft.device),
    )
    assert accepted.tolist() == [0, 1, 2]
    assert trimmed.tolist() == [2, 1, 0]
    assert torch.equal(bonus[:2], proposals[[0, 1], [0, 1]])


def test_native_algorithm_preserves_dspark_contract(dspark_runtime, monkeypatch):
    from sglang.srt.speculative import spec_registry
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    from areal.engine.sglang_dspark_registration import register_dspark
    from areal.engine.sglang_dspark_worker import JointDSparkWorker
    from areal.engine.sglang_sampling_mask_plugin import JOINT_DSPARK

    monkeypatch.setattr(spec_registry, "_REGISTRY", {})
    register_dspark()
    algo = SpeculativeAlgorithm.from_string(JOINT_DSPARK)
    assert (
        algo.create_worker(SimpleNamespace(disable_overlap_schedule=False))
        is JointDSparkWorker
    )
    assert algo.is_dspark() and algo.supports_target_verify_for_draft()
    assert algo.get_num_tokens_per_req_for_target_verify(7, True) == 6
    assert algo.get_num_tokens_per_req_for_target_verify(7, False) == 7
    assert SpeculativeAlgorithm.from_string("DSPARK") is SpeculativeAlgorithm.DSPARK


@pytest.mark.parametrize("compact", [False, True])
def test_joint_executor_bypasses_only_metadata_adjustment(
    dspark_runtime, monkeypatch, compact
):
    from areal.engine.sglang_dspark_worker import (
        JointVerifyExecutor,
        TargetVerifyExecutor,
    )

    executor = JointVerifyExecutor.__new__(JointVerifyExecutor)
    batch = object()
    result = SimpleNamespace(logits_output=object())
    method = "run_compact" if compact else "run_non_compact"

    def forward(self, *, batch, sampling_info):
        assert sampling_info is None
        return (result, "hidden") if compact else result

    monkeypatch.setattr(TargetVerifyExecutor, method, forward)
    out = getattr(executor, method)(batch=batch, sampling_info=object())
    assert (out[0] if compact else out) is result
    assert executor.pending_output == (batch, result.logits_output)


def test_idle_prefill_participates_without_publishing_mask(dspark_runtime, monkeypatch):
    from areal.engine import sglang_dspark_worker as native

    model = SimpleNamespace(capture_aux_hidden_states=True)
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: True))
    calls = []

    def run(batch, is_verify, capture_hidden_mode):
        calls.append((is_verify, capture_hidden_mode, model.capture_aux_hidden_states))

    worker = SimpleNamespace(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=model), forward_batch_generation=run
        ),
        _decode_idle_result=lambda on_publish: on_publish,
    )
    monkeypatch.setattr(
        native, "get_parallel", lambda: SimpleNamespace(enable_dp_attention=True)
    )
    assert native.JointDSparkWorker._forward_prefill(worker, batch, 42) == 42
    assert calls == [(True, native.CaptureHiddenMode.NULL, False)]
    assert model.capture_aux_hidden_states is True
