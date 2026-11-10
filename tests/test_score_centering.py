# SPDX-License-Identifier: Apache-2.0
import copy
from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    MicroBatchSpec,
    PPOActorConfig,
    PPOConfig,
)
from areal.api.io_struct import ModelResponse
from areal.trainer.ppo.actor import grpo_loss_fn
from areal.utils.data import (
    concat_batch,
    pack_tensor_dict,
    split_batch,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.score_centering import (
    build_weight_fn,
    score_centering_correction,
    score_centering_logprobs,
    selected_logprobs,
    trajectory_heads,
    validate_heads,
)


@pytest.fixture
def isolated_actor_stats_tracker(monkeypatch):
    from areal.trainer.ppo import actor
    from areal.utils.stats_tracker import DistributedStatsTracker

    monkeypatch.setattr(actor, "stats_tracker", DistributedStatsTracker())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_backward_matches_dense_rounding_with_peaked_logits(device, dtype):
    """Preserve dense VJP rounding before BF16, including repeated selected IDs."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(31)
    logits = (10 * torch.randn(17, 8192, device=device)).to(dtype).requires_grad_()
    oracle = logits.detach().clone().requires_grad_()
    heads = logits.detach().topk(128, dim=-1).indices
    ids = torch.cat((heads[:, :1], heads), dim=-1)
    upstream = torch.randn(ids.shape, device=device) * 1e-3
    upstream[:, 0] = -upstream[:, 1:].sum(-1)
    actual, _ = selected_logprobs(logits, ids, chunk_size=7)
    expected = oracle.float().log_softmax(-1).gather(-1, ids)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(logits.grad, oracle.grad, rtol=0, atol=0)


def reference(logits, labels, ids, q, weight_fn=None):
    """Independent dense normalisation of the same head/tail residual."""
    lp = logits.log_softmax(-1)
    phead = lp.gather(-1, ids)
    with torch.no_grad():
        valid = torch.isfinite(q)
        p = phead.exp() * valid
        sampler = q.exp() * valid
        rho = (1 - sampler.sum(-1)).clamp_min(1e-6) / (1 - p.sum(-1)).clamp_min(1e-6)
        if weight_fn is None:
            coefficient = sampler - rho[:, None] * p
        else:
            tiny = torch.finfo(p.dtype).tiny
            ratio = torch.where(valid, p / sampler.clamp_min(tiny), torch.zeros_like(p))
            alpha = rho * weight_fn(torch.clamp(1.0 / rho, max=1.0 / 1e-6))
            coefficient = sampler * weight_fn(ratio) - alpha[:, None] * p
        coefficient = torch.where(valid, coefficient, torch.zeros_like(coefficient))
    return lp.gather(-1, labels[:, None]).squeeze(-1), (phead * coefficient).sum(-1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk", [1, 7, 100])
@pytest.mark.parametrize("width", [1, 9, 31])
def test_reference_values_and_gradients_match_dense(device, dtype, chunk, width):
    """Compare independent dense normalization, including overlapping sampled IDs."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(7)
    x = torch.randn(13, 31, device=device, dtype=dtype, requires_grad=True)
    oracle = (
        x.detach()
        .to(torch.float64 if dtype == torch.float64 else torch.float32)
        .requires_grad_()
    )
    ids = torch.randperm(31, device=device)[:width].expand(13, -1)
    labels = ids[:, 0]
    q = torch.randn_like(oracle).log_softmax(-1).gather(-1, ids).detach()
    lp, entropy, c = score_centering_logprobs(x, labels, ids, q, chunk)
    expected_lp, expected_c = reference(oracle, labels, ids, q)
    advantage = torch.randn(13, device=device, dtype=oracle.dtype)
    ((c - lp) * advantage).sum().backward()
    ((expected_c - expected_lp) * advantage).sum().backward()
    tol = 1e-10 if dtype == torch.float64 else 3e-5
    torch.testing.assert_close(lp, expected_lp, atol=tol, rtol=tol)
    torch.testing.assert_close(c, expected_c, atol=tol, rtol=tol)
    torch.testing.assert_close(
        x.grad,
        oracle.grad.to(dtype),
        atol=tol if dtype != torch.bfloat16 else 0.004,
        rtol=tol if dtype != torch.bfloat16 else 0.008,
    )
    assert not entropy.requires_grad


def test_on_policy_correction_has_zero_value_and_gradient():
    """With positive tail mass, the correction vanishes on-policy."""
    for width in [3]:
        x = torch.randn(8, 17, dtype=torch.float64, requires_grad=True)
        lp = x.log_softmax(-1)[:, :width]
        c = score_centering_correction(lp, lp.detach())
        c.sum().backward()
        torch.testing.assert_close(c, torch.zeros_like(c), atol=0, rtol=0)
        torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)


def test_zero_sampler_tail_is_floored_instead_of_infinite():
    """Both tail masses are floored, so a full sampler head stays finite."""
    head = torch.tensor([[-0.1]], dtype=torch.float64, requires_grad=True)
    q = torch.zeros(1, 1, dtype=torch.float64)  # sampler head mass = 1, no tail
    correction = score_centering_correction(head, q)
    correction.sum().backward()
    # The residual is detached, so the gradient is the residual itself:
    # q - rho * p with rho = eps / (1 - p.sum()), not q with rho = 0.
    with torch.no_grad():
        p = head.exp()
        rho = torch.tensor(1e-6, dtype=torch.float64) / (1 - p.sum())
        expected = q.exp() - rho * p
    assert torch.isfinite(correction).all()
    torch.testing.assert_close(head.grad, expected, rtol=1e-12, atol=0)


def test_on_policy_full_support_has_zero_score_gradient():
    """Full-support centering may change loss value, but not on-policy gradient."""
    torch.manual_seed(4)
    logits = torch.randn(8, 17, dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(-1)
    score_centering_correction(logp, logp.detach()).sum().backward()
    torch.testing.assert_close(
        logits.grad, torch.zeros_like(logits), rtol=0, atol=1e-12
    )


def test_head_tail_correction_matches_full_distribution_score_gradient():
    """Check the SC identity against an explicitly constructed full sampler."""
    torch.manual_seed(15)
    logits = torch.randn(5, 19, dtype=torch.float64, requires_grad=True)
    oracle = logits.detach().clone().requires_grad_()
    ids = torch.arange(7).expand(5, -1)
    logp = logits.log_softmax(-1)
    head = logp.gather(-1, ids)
    q_head = torch.randn(5, 7, dtype=torch.float64).softmax(-1) * 0.7
    with torch.no_grad():
        p = oracle.softmax(-1)
        rho = (1 - q_head.sum(-1)) / (1 - p[:, :7].sum(-1))
        q_full = rho[:, None] * p
        q_full[:, :7] = q_head
    score_centering_correction(head, q_head.log()).sum().backward()
    (q_full * oracle.log_softmax(-1)).sum().backward()
    torch.testing.assert_close(logits.grad, oracle.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("chunk", [1, 8])
def test_hidden_weight_gradients_and_adam_step_match(chunk):
    """Reference autograd must preserve hidden-state and weight updates."""
    torch.manual_seed(4)
    h = torch.randn(9, 5, requires_grad=True)
    w = torch.randn(5, 23, requires_grad=True)
    h2, w2 = h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    ids = torch.arange(7).expand(9, -1)
    labels = ids[:, 0]
    q = torch.randn(9, 23).log_softmax(-1)[:, :7]
    a = torch.randn(9)
    lp, _, c = score_centering_logprobs(h @ w, labels, ids, q, chunk)
    lp2, c2 = reference(h2 @ w2, labels, ids, q)
    ((c - lp) * a).mean().backward()
    ((c2 - lp2) * a).mean().backward()
    for actual, expected in [(h.grad, h2.grad), (w.grad, w2.grad)]:
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    o1 = torch.optim.Adam([h, w], lr=1e-4)
    o2 = torch.optim.Adam([h2, w2], lr=1e-4)
    o1.step()
    o2.step()
    torch.testing.assert_close(w, w2, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    "ids,logps",
    [
        (None, None),
        ([[1, 1]], [[-1.0, -2.0]]),
        ([[1, 2]], [[0.0, 0.0]]),
        ([[1, 2]], [[float("nan"), -2.0]]),
        ([[1]], [[-1.0]]),
    ],
)
def test_invalid_heads_are_rejected(ids, logps):
    """Malformed sampler evidence must never silently disable correction."""
    with pytest.raises(ValueError):
        validate_heads(ids, logps, 1, 2)


@pytest.mark.parametrize("width", [1, 12])
def test_unequal_trajectory_batching_preserves_head_axis(width):
    """Exercise concat/unpad, minibatch split and packing with K > sequence length."""
    trajectories = []
    for length in [4, 7]:
        response = ModelResponse(
            input_tokens=[1, 2],
            output_tokens=[3] * (length - 2),
            sampler_head_ids=[list(range(width))] * (length - 2),
            sampler_head_logprobs=[[-4.0] * width] * (length - 2),
        )
        heads = trajectory_heads(response, width)
        trajectory = {k: v.unsqueeze(0) for k, v in heads.items()}
        trajectory.update(
            input_ids=torch.arange(length)[None],
            attention_mask=torch.ones(1, length, dtype=torch.bool),
        )
        trajectories.append(trajectory)
    batch, meta = concat_batch(trajectories)
    restored = split_batch(batch, meta)
    for actual, expected in zip(restored, trajectories):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    mbs = split_padded_tensor_dict_into_mb_list(batch, MicroBatchSpec(n_mbs=2)).mbs
    assert len(mbs) == 2
    for mb in mbs:
        assert mb["sampler_head_ids"].shape[0] == 1
        packed = pack_tensor_dict(mb)
        assert packed["sampler_head_ids"].shape == (
            int(mb["attention_mask"].sum()),
            width,
        )


def test_score_centering_config_rejects_recomputed_behavior():
    """Behavior probabilities must originate in the sampler."""
    with pytest.raises(ValueError, match="recompute_logprob"):
        PPOActorConfig(
            backend="fsdp:d1", use_score_centering=True, recompute_logprob=True
        )


def test_empty_reference_batch_supports_backward():
    """Empty selection is differentiable and keeps its expected shape."""
    x = torch.empty(0, 17, requires_grad=True)
    lp, _, c = score_centering_logprobs(
        x,
        torch.empty(0, dtype=torch.long),
        torch.empty(0, 3, dtype=torch.long),
        torch.empty(0, 3),
    )
    (lp.sum() + c.sum()).backward()
    assert x.grad.shape == x.shape


def test_default_openai_payload_remains_unchanged():
    """The SC width never reaches the OpenAI SDK, but the proxy path still works.

    main rejects the OpenAI transport outright; this branch needs it because the
    AReaL agent workflow on vLLM carries the sampler heads through the proxy.
    """
    assert (
        "score_centering_top_k" not in GenerationHyperparameters().to_openai_args_dict()
    )
    assert (
        "score_centering_top_k"
        not in GenerationHyperparameters(score_centering_top_k=2).to_openai_args_dict()
    )


@pytest.mark.parametrize("width", [1, 12])
@pytest.mark.parametrize("tis_cap", [None, 2.0])
@pytest.mark.usefixtures("isolated_actor_stats_tracker")
def test_actor_alignment_and_fsdp_loss_match_dense_with_padding(width, tis_cap):
    """Exercise real actor alignment and FSDP loss hooks without mocking DTensor."""
    from functools import partial

    from areal.engine.fsdp_engine import FSDPEngine
    from areal.trainer.ppo.actor import PPOActor, grpo_loss_fn
    from areal.utils.data import pad_mb_list, unsqueeze_mb_list

    weight_fn = None if tis_cap is None else build_weight_fn("tis", 0.0, tis_cap)
    config = PPOActorConfig(
        backend="fsdp:d1",
        use_score_centering=True,
        sc_mode="tis" if tis_cap is not None else "none",
        sc_clip_low=0.0,
        sc_clip_high=tis_cap if tis_cap is not None else 5.0,
        recompute_logprob=False,
        kl_ctl=0,
        mb_spec=MicroBatchSpec(n_mbs=2),
    )
    actor = PPOActor(config, SimpleNamespace())
    raw = []
    for length in [4, 7]:
        ids = torch.arange(width).expand(1, length, width).clone()
        ids = (ids + torch.arange(length)[None, :, None]) % 17
        ids[:, :2] = 0
        q = torch.full((1, length, width), -4.0)
        q[:, :2] = -torch.inf
        mask = torch.tensor([[0, 0] + [1] * (length - 2)])
        raw.append(
            dict(
                input_ids=torch.arange(length)[None],
                attention_mask=torch.ones(1, length, dtype=torch.bool),
                loss_mask=mask,
                logprobs=torch.full((1, length), -4.0),
                rewards=torch.tensor([1.0]),
                sampler_head_ids=ids,
                sampler_head_logprobs=q,
            )
        )
    aligned = actor.compute_advantages(copy.deepcopy(raw))
    for old, new in zip(raw, aligned):
        torch.testing.assert_close(
            new["sampler_head_ids"],
            torch.roll(old["sampler_head_ids"], -1, 1),
            atol=0,
            rtol=0,
        )
    batch, _ = concat_batch(aligned)
    mbs = split_padded_tensor_dict_into_mb_list(batch, config.mb_spec)
    mbs.mbs = [pack_tensor_dict(mb) for mb in mbs.mbs]
    mbs = unsqueeze_mb_list(pad_mb_list(mbs))
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.config = config
    engine.enable_tree_training = False
    engine.parallel_helper = SimpleNamespace(tp_size=1, sp_size=1)
    engine.model_config = SimpleNamespace(vocab_size=17)
    for item in mbs:
        inputs, ctx = engine._prepare_mb_inputs(item)
        assert "sampler_head_ids" not in inputs
        n = ctx.mb_input["input_ids"].numel()
        logits = torch.randn(inputs["input_ids"].numel(), 17, requires_grad=True)
        oracle = logits.detach().clone().requires_grad_()
        labels = torch.roll(inputs["input_ids"], -1, -1).reshape(-1)[:n]
        lp, c = reference(
            oracle[:n],
            labels,
            ctx.mb_input["sampler_head_ids"],
            ctx.mb_input["sampler_head_logprobs"],
            weight_fn=weight_fn,
        )
        mask = ctx.mb_input["loss_mask"].bool()
        weight = (
            1.0
            if weight_fn is None
            else (lp.detach() - ctx.mb_input["logprobs"])
            .exp()
            .clamp(min=0.0, max=tis_cap)
        )
        expected = (
            torch.where(mask, ctx.mb_input["advantages"] * (c - weight * lp), 0).sum()
            / mask.sum()
        )
        loss = engine._compute_logprobs_and_loss(
            logits,
            ctx,
            partial(
                grpo_loss_fn,
                eps_clip=0.2,
                eps_clip_higher=None,
                c_clip=None,
                use_score_centering=True,
                sc_weight_fn=weight_fn,
            ),
            lambda data: data["loss_mask"].count_nonzero(),
            mask.sum(),
        )
        loss.backward()
        expected.backward()
        torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(logits.grad, oracle.grad, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            logits.grad[n:], torch.zeros_like(logits.grad[n:]), atol=0, rtol=0
        )


@pytest.mark.parametrize(
    "overrides",
    [
        dict(temperature=0.8),
        dict(top_k=10),
        dict(top_p=0.9),
        dict(greedy=True),
        dict(frequency_penalty=1),
        dict(score_centering_top_k=2000),  # above the documented 1024 head width
        dict(score_centering_top_k=True),  # a bool is not a width
    ],
)
def test_sampling_preconditions_are_enforced_at_config_load(overrides):
    """Truncated, greedy or penalised sampling must fail before any rollout."""
    kwargs = {"score_centering_top_k": 2}
    kwargs.update(overrides)
    gconfig = GenerationHyperparameters(**kwargs)
    with pytest.raises(ValueError):
        PPOConfig(
            actor=PPOActorConfig(backend="fsdp:d1", use_score_centering=True),
            rollout=InferenceEngineConfig(backend="vllm:d1"),
            gconfig=gconfig,
        )


def test_score_centering_rejects_distillation():
    """The distillation branch would replace the REINFORCE+SC objective."""
    logprobs = torch.tensor([[-1.0, -2.0]])
    data = {
        "logprobs": logprobs.clone(),
        "prox_logp": logprobs.clone(),  # otherwise the prox-logp check fires first
        "advantages": torch.tensor([[1.0, 1.0]]),
        "loss_mask": torch.ones(1, 2, dtype=torch.bool),
        "teacher_logp": torch.tensor([[-1.0, -2.0]]),
    }
    with pytest.raises(ValueError, match="distillation"):
        grpo_loss_fn(
            logprobs,
            torch.zeros_like(logprobs),
            data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            use_score_centering=True,
            sc_correction=torch.zeros_like(logprobs),
        )


def test_missing_heads_raise_value_error_not_keyerror():
    """SC on + a micro-batch without heads must name the missing key."""
    from areal.engine.fsdp_engine import FSDPEngine, FSDPTrainContext

    config = PPOActorConfig(
        backend="fsdp:d1", use_score_centering=True, recompute_logprob=False
    )
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.config = config
    engine.enable_tree_training = False
    engine.parallel_helper = SimpleNamespace(tp_size=1, sp_size=1)
    engine.model_config = SimpleNamespace(vocab_size=17)

    mb = {
        "input_ids": torch.arange(4)[None],
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1, 0]]),
        "logprobs": torch.zeros(1, 4),
        "advantages": torch.zeros(1, 4),
    }
    ctx = FSDPTrainContext(model_inputs={"input_ids": mb["input_ids"]}, mb_input=mb)
    logits = torch.randn(4, 17, requires_grad=True)
    args = (
        logits,
        ctx,
        lambda *a, **k: logits.mean(),
        lambda d: d["loss_mask"].count_nonzero(),
        mb["loss_mask"].sum(),
    )
    with pytest.raises(ValueError, match="no sampler top-k heads"):
        engine._compute_logprobs_and_loss(*args)

    # Only one of the two keys present must fail with the same, key-named error.
    mb["sampler_head_ids"] = torch.zeros(4, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="missing sampler_head_logprobs"):
        engine._compute_logprobs_and_loss(*args)
