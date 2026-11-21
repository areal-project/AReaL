# SPDX-License-Identifier: Apache-2.0
"""Alignment contract for teacher-scored pure-KD distillation.

Three independent conventions meet here, and getting any of them wrong shifts the
KD target by one token while still producing finite, plausible-looking losses:

1. **FSDP scoring** builds labels with ``torch.roll(input_ids, shifts=-1)``
   (``areal/engine/fsdp_engine.py`` around the ``_compute_logprobs`` helper), so
   score position ``i`` is the probability of token ``i + 1``.
2. **The advantage pipeline** rolls ``loss_mask`` and the rollout
   ``logprobs`` left by one (``areal/trainer/ppo/advantage.py``) to land in that
   same convention.
3. **The KD loss** consumes ``teacher_logp``, ``logprobs`` and ``old_logp`` as-is.

Therefore the KD path must **not** shift FSDP-produced ``teacher_logp`` again.
These tests pin that with an exactly known, one-token response.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from areal.trainer.ppo.actor import grpo_loss_fn


def _sampled_kd_per_token(
    *,
    logprobs,
    teacher_logp,
    old_logp,
    loss_mask,
    kl_estimator,
    distill_loss_weight,
):
    """Shim: run the real ``grpo_loss_fn`` and return (per-token term, coef).

    The rebased design no longer exposes a standalone helper; the KD term lives
    inside ``grpo_loss_fn``. This keeps the alignment assertions below expressed
    against production code.
    """
    lp = logprobs.detach().clone().requires_grad_(True)
    captured = {}

    def _stat(**kwargs):
        for key, value in kwargs.items():
            if key != "denominator":
                captured.setdefault(key, value)

    tracker = MagicMock()
    tracker.stat.side_effect = _stat
    data = {
        "logprobs": old_logp,
        "prox_logp": old_logp,
        "advantages": torch.zeros_like(lp),
        "loss_mask": loss_mask,
        "teacher_logp": teacher_logp,
        "rl_loss_weight": 0.0,
        "distill_loss_weight": distill_loss_weight,
        "kl_estimator": kl_estimator,
    }
    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        grpo_loss_fn(
            logprobs=lp,
            entropy=torch.zeros_like(lp),
            input_data=data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
        )
    term = captured.get("rkl_loss")
    # Upstream reports -advantages for k1, i.e. -(t - l).
    if kl_estimator == "k1" and term is not None:
        term = -term
    return term, distill_loss_weight


def _log_softmax_at(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reproduce the FSDP scoring convention: label at position i is token i+1."""
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _score_like_fsdp(input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Exact analogue of `_compute_logprobs`: roll labels left, then gather."""
    labels = torch.roll(input_ids, shifts=-1, dims=-1)
    return _log_softmax_at(logits, labels)


def test_single_token_response_is_scored_at_the_right_position():
    """Three prompt tokens, one response token, rest padding.

    The FSDP roll makes position 2 (the last prompt token) carry the
    probability of token 3, which is the response. The KD loss must use that
    value directly, with no additional shift.
    """
    seq_len = 6
    # Token ids are arbitrary but distinct so a misalignment cannot coincide.
    input_ids = torch.tensor([[10, 11, 12, 13, 0, 0]])
    vocab = 16

    def make_logits(peaked_at: int, strength: float = 8.0) -> torch.Tensor:
        logits = torch.zeros(1, seq_len, vocab)
        logits[:, :, peaked_at] = strength
        return logits

    # Student assigns high probability to token 13 at the prediction position.
    student = make_logits(13)
    # Teacher prefers token 13 more strongly -> teacher logp > student logp.
    teacher = make_logits(13, strength=12.0)

    student_lp = _score_like_fsdp(input_ids, student)
    teacher_lp = _score_like_fsdp(input_ids, teacher)

    # Precondition: position 2 is the prediction slot for token 3 (the response).
    assert int(input_ids[0, 3]) == 13
    assert teacher_lp[0, 2] > student_lp[0, 2]

    # Response mask in *token* convention (1 at the response, 0 elsewhere),
    # rolled left by the advantage pipeline.
    token_loss_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    loss_mask = torch.roll(token_loss_mask, shifts=-1, dims=-1).bool()
    assert loss_mask.sum() == 1
    assert bool(loss_mask[0, 2]), "prediction slot must be the last prompt token"

    # old_logp in the same convention (rollout logprobs, rolled left).
    old_lp = torch.roll(student_lp, shifts=-1, dims=-1)

    term, coef = _sampled_kd_per_token(
        logprobs=student_lp,
        teacher_logp=teacher_lp,
        old_logp=old_lp,
        loss_mask=loss_mask,
        kl_estimator="k1",
        distill_loss_weight=1.0,
    )
    # Only the active slot contributes. Since #1838 the reported KD value is
    # the signed log-ratio itself; the importance ratio lives in the policy
    # loss, which owns clipping and the backward pass.
    active = term[0, 2]
    expected = teacher_lp[0, 2] - student_lp[0, 2]
    torch.testing.assert_close(active, expected, rtol=1e-12, atol=1e-12)


def test_shifting_teacher_logp_again_would_change_the_loss():
    """A double shift must be detectable, i.e. the test above is not vacuous.

    Shifting `teacher_logp` one more position changes the KD target; if the
    production code ever did this, the per-token term would differ from the
    hand-computed value.
    """
    seq_len = 6
    input_ids = torch.tensor([[10, 11, 12, 13, 0, 0]])
    logits_s = torch.zeros(1, seq_len, 16)
    logits_t = torch.zeros(1, seq_len, 16)
    logits_s[:, :, 13] = 8.0
    logits_t[:, :, 13] = 12.0
    # Break symmetry so a shift changes the value.
    logits_t[:, 1, 12] = 6.0

    student_lp = _score_like_fsdp(input_ids, logits_s)
    teacher_lp = _score_like_fsdp(input_ids, logits_t)
    token_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    loss_mask = torch.roll(token_mask, shifts=-1, dims=-1).bool()
    old_lp = torch.roll(student_lp, shifts=-1, dims=-1)

    def loss_with(teacher: torch.Tensor) -> torch.Tensor:
        term, coef = _sampled_kd_per_token(
            logprobs=student_lp,
            teacher_logp=teacher,
            old_logp=old_lp,
            loss_mask=loss_mask,
            kl_estimator="k1",
            distill_loss_weight=1.0,
        )
        return coef * term.sum() / loss_mask.sum()

    correct = loss_with(teacher_lp)
    double_shifted = loss_with(torch.roll(teacher_lp, shifts=-1, dims=-1))
    assert not torch.isclose(correct, double_shifted, rtol=1e-9, atol=1e-12), (
        "a double shift must change the loss, otherwise the alignment test "
        "cannot distinguish correct from shifted code"
    )


def test_padding_positions_are_excluded_by_the_rolled_mask():
    """Padding tokens must not enter the KD numerator.

    The roll moves the last token's mask entry to -1 (dropped), so a padded
    tail cannot contribute even when its log-probs are finite.
    """
    seq_len = 5
    token_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
    loss_mask = torch.roll(token_mask, shifts=-1, dims=-1).bool()
    assert bool(loss_mask[0, 2])
    assert not bool(loss_mask[0, 4]), "padding slot must stay inactive"

    logprobs = torch.full((1, seq_len), -0.5)
    teacher = torch.full((1, seq_len), -0.4)
    old = torch.full((1, seq_len), -0.6)
    term, _ = _sampled_kd_per_token(
        logprobs=logprobs,
        teacher_logp=teacher,
        old_logp=old,
        loss_mask=loss_mask,
        kl_estimator="k1",
        distill_loss_weight=1.0,
    )
    assert term[0, 4].item() == 0.0
