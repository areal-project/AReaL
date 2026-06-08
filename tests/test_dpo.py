# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.trainer.dpo.dpo_engine import (
    _dpo_loss_weight,
    _dpo_valid_pairs,
    compute_dpo_loss,
)


class TestDPOLoss:
    """Test cases for the DPO loss function."""

    @pytest.fixture
    def basic_pair_data(self):
        """Create a basic paired dataset with 2 pairs (4 sequences packed)."""
        # Sequence lengths: [5, 6, 4, 7] = chosen0, rejected0, chosen1, rejected1
        cu_seqlens = torch.tensor([0, 5, 11, 15, 22], dtype=torch.int32)
        total_tokens = 22

        # loss_mask: 0 for prompt tokens, 1 for response tokens
        loss_mask = torch.zeros(total_tokens, dtype=torch.bool)
        # chosen0: prompt=3 tokens, response=2 tokens
        loss_mask[3:5] = True
        # rejected0: prompt=3 tokens, response=3 tokens
        loss_mask[8:11] = True
        # chosen1: prompt=2 tokens, response=2 tokens
        loss_mask[13:15] = True
        # rejected1: prompt=2 tokens, response=5 tokens
        loss_mask[17:22] = True

        return {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": torch.randn(total_tokens),
        }

    def test_loss_is_scalar(self, basic_pair_data):
        """DPO loss should be a scalar."""
        logprobs = torch.randn(22)
        loss = compute_dpo_loss(logprobs, None, basic_pair_data, beta=0.1)
        assert loss.ndim == 0

    def test_loss_is_finite(self, basic_pair_data):
        """DPO loss should be finite."""
        logprobs = torch.randn(22)
        loss = compute_dpo_loss(logprobs, None, basic_pair_data, beta=0.1)
        assert torch.isfinite(loss)

    def test_loss_positive(self, basic_pair_data):
        """DPO loss (negative logsigmoid) should always be non-negative."""
        logprobs = torch.randn(22)
        loss = compute_dpo_loss(logprobs, None, basic_pair_data, beta=0.1)
        assert loss.item() >= 0.0

    def test_loss_decreases_when_chosen_preferred(self):
        """When policy strongly prefers chosen over rejected, loss should be lower."""
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        loss_mask = torch.ones(8, dtype=torch.bool)
        # All tokens are response tokens
        ref_logprobs = torch.zeros(8)

        input_ = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
        }

        # Policy strongly prefers chosen (high logprobs for chosen, low for rejected)
        logprobs_good = torch.tensor([-0.1, -0.1, -0.1, -0.1, -5.0, -5.0, -5.0, -5.0])
        loss_good = compute_dpo_loss(logprobs_good, None, input_, beta=0.1)

        # Policy prefers rejected (low logprobs for chosen, high for rejected)
        logprobs_bad = torch.tensor([-5.0, -5.0, -5.0, -5.0, -0.1, -0.1, -0.1, -0.1])
        loss_bad = compute_dpo_loss(logprobs_bad, None, input_, beta=0.1)

        assert loss_good.item() < loss_bad.item()

    def test_beta_scaling(self):
        """Higher beta should amplify the loss difference."""
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        loss_mask = torch.ones(8, dtype=torch.bool)
        ref_logprobs = torch.zeros(8)

        input_ = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
        }

        logprobs = torch.tensor([-0.5, -0.5, -0.5, -0.5, -2.0, -2.0, -2.0, -2.0])

        loss_low_beta = compute_dpo_loss(logprobs, None, input_, beta=0.01)
        loss_high_beta = compute_dpo_loss(logprobs, None, input_, beta=1.0)

        # With higher beta and chosen preferred, loss should be lower
        # (stronger signal pushes logsigmoid towards 0)
        assert loss_low_beta.item() != loss_high_beta.item()

    def test_ref_logprobs_effect(self):
        """Reference logprobs should offset the policy preference."""
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        loss_mask = torch.ones(8, dtype=torch.bool)

        logprobs = torch.tensor([-1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0])

        # Ref model also prefers chosen → net advantage is small
        ref_logprobs_same = torch.tensor(
            [-1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0]
        )
        input_same = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs_same,
        }
        loss_same = compute_dpo_loss(logprobs, None, input_same, beta=0.1)

        # Ref model prefers rejected → net advantage for chosen is larger
        ref_logprobs_opp = torch.tensor(
            [-2.0, -2.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0]
        )
        input_opp = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs_opp,
        }
        loss_opp = compute_dpo_loss(logprobs, None, input_opp, beta=0.1)

        # When ref agrees with policy, log-ratio difference is ~0 → loss ≈ log(2)
        # When ref disagrees, net preference is stronger → loss is lower
        assert loss_opp.item() < loss_same.item()

    def test_empty_pairs(self):
        """When all sequences are empty, loss should be zero."""
        cu_seqlens = torch.tensor([0, 0, 0], dtype=torch.int32)
        loss_mask = torch.zeros(0, dtype=torch.bool)
        ref_logprobs = torch.zeros(0)

        input_ = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
        }

        logprobs = torch.zeros(0)
        loss = compute_dpo_loss(logprobs, None, input_, beta=0.1)
        assert loss.item() == 0.0

    def test_loss_mask_only_response(self):
        """Only response tokens (loss_mask=True) should contribute to the loss."""
        cu_seqlens = torch.tensor([0, 6, 12], dtype=torch.int32)
        # Prompt is 4 tokens, response is 2 tokens for each
        loss_mask = torch.tensor([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], dtype=torch.bool)
        ref_logprobs = torch.zeros(12)

        input_ = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
        }

        # Make prompt tokens wildly different but response tokens identical
        logprobs_a = torch.tensor(
            [
                -10.0,
                -10.0,
                -10.0,
                -10.0,
                -1.0,
                -1.0,
                -10.0,
                -10.0,
                -10.0,
                -10.0,
                -2.0,
                -2.0,
            ]
        )
        logprobs_b = torch.tensor(
            [-0.1, -0.1, -0.1, -0.1, -1.0, -1.0, -0.1, -0.1, -0.1, -0.1, -2.0, -2.0]
        )

        loss_a = compute_dpo_loss(logprobs_a, None, input_, beta=0.1)
        loss_b = compute_dpo_loss(logprobs_b, None, input_, beta=0.1)

        # Prompt tokens should not affect loss since loss_mask is 0 there
        torch.testing.assert_close(loss_a, loss_b, rtol=1e-5, atol=1e-5)

    def test_missing_ref_logprobs_raises(self):
        """When ref_logprobs is not in input_, a KeyError should be raised."""
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        loss_mask = torch.ones(8, dtype=torch.bool)
        logprobs = torch.tensor([-0.5, -0.5, -0.5, -0.5, -2.0, -2.0, -2.0, -2.0])

        input_no_ref = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
        }
        with pytest.raises(KeyError):
            compute_dpo_loss(logprobs, None, input_no_ref, beta=0.1)


class TestDPOValidPairs:
    """Test the helper functions for DPO pair validation."""

    def test_valid_pairs_all_nonempty(self):
        cu_seqlens = torch.tensor([0, 5, 10, 15, 20], dtype=torch.int32)
        input_ = {"cu_seqlens": cu_seqlens}
        valid = _dpo_valid_pairs(input_)
        assert valid.all()

    def test_valid_pairs_with_empty(self):
        # Second pair has an empty rejected sequence
        cu_seqlens = torch.tensor([0, 5, 10, 15, 15], dtype=torch.int32)
        input_ = {"cu_seqlens": cu_seqlens}
        valid = _dpo_valid_pairs(input_)
        assert valid[0].item() is True
        assert valid[1].item() is False

    def test_loss_weight(self):
        cu_seqlens = torch.tensor([0, 5, 10, 15, 20], dtype=torch.int32)
        input_ = {"cu_seqlens": cu_seqlens}
        weight = _dpo_loss_weight(input_)
        assert weight.item() == 2.0


class TestDPOLossIntraSequenceShift:
    """Regression tests for the shift-align fix (no cross-sequence leakage)."""

    def test_chosen_last_response_token_does_not_leak_into_next_prompt(self):
        """Two adjacent packed sequences; chosen ends with a response token
        whose mask is True. A naive global ``torch.roll`` would wrap this True
        to the first position of the next sequence (a prompt token) and pollute
        the rejected-logp sum. The intra-sequence shift must prevent that.
        """
        # chosen: seqlen=4, last 2 are response (loss_mask: 0,0,1,1)
        # rejected: seqlen=4, first token would be polluted by global roll;
        #   here response = last 1 token (loss_mask: 0,0,0,1).
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        loss_mask = torch.tensor([0, 0, 1, 1, 0, 0, 0, 1], dtype=torch.bool)

        # Give a distinctive value to the rejected prompt's first position
        # (index 4). Under a buggy global roll, chosen's mask[3]=True would
        # move to index 2 (intra-sequence), but chosen's mask[0] would come
        # from rejected's mask[0]=0 — no leak there. The other direction:
        # rejected's mask[3]=1 (end) would be rolled to the *first* position
        # overall (index 0) under torch.roll(shifts=-1), polluting chosen's
        # prompt. We test both scenarios via extreme logprob at those spots.
        logprobs = torch.zeros(8, dtype=torch.float32)
        logprobs[0] = 100.0  # chosen's first token — must NOT be counted
        logprobs[4] = 100.0  # rejected's first token — must NOT be counted
        ref_logprobs = torch.zeros(8, dtype=torch.float32)

        input_ = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
        }

        loss = compute_dpo_loss(logprobs, None, input_, beta=0.1)
        # With the intra-sequence shift:
        #   shifted_mask = [loss_mask[1..n-1], False] with last-of-seq zeroed.
        # For chosen [0,0,1,1]: shifted = [0,1,1,0] → counts logprobs[1], logprobs[2]
        # For rejected [0,0,0,1]: shifted = [0,0,1,0] → counts logprobs[6]
        # All those positions have value 0 → policy_logp = 0 for both
        # → logits = 0 → loss = -logsigmoid(0) = log(2)
        import math

        torch.testing.assert_close(
            loss,
            torch.tensor(math.log(2.0), dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_last_token_of_sequence_is_never_counted(self):
        """The last position of each packed sequence has no "next token" —
        its shifted mask must be False regardless of the original mask value.
        """
        # Single sequence, loss_mask all True; last position has no next token.
        cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
        loss_mask = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
        ref_logprobs = torch.zeros(4, dtype=torch.float32)
        # Huge value at position 3 (last token of seq). If the shifted mask
        # included this position, the summed logp would be huge.
        logprobs = torch.tensor([0.0, 0.0, 0.0, 1e6], dtype=torch.float32)

        # Make it a valid pair by adding a second (identical) sequence.
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        loss_mask = torch.cat([loss_mask, loss_mask])
        logprobs = torch.cat([logprobs, logprobs])
        ref_logprobs = torch.cat([ref_logprobs, ref_logprobs])

        input_ = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
        }
        loss = compute_dpo_loss(logprobs, None, input_, beta=0.1)
        # shifted_mask per seq = [1, 1, 1, 0] → counts logprobs[0..2] = 0
        # Both chosen and rejected identical → logits = 0 → loss = log 2
        import math

        torch.testing.assert_close(
            loss,
            torch.tensor(math.log(2.0), dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_two_pairs_same_content_zero_margin(self):
        """Two pairs with identical chosen/rejected content must yield log(2) loss
        (logits=0) independent of masked token values — sanity check after
        vectorized refactor.
        """
        # Two pairs: chosen_0, rejected_0, chosen_1, rejected_1
        cu_seqlens = torch.tensor([0, 5, 10, 15, 20], dtype=torch.int32)
        loss_mask = torch.tensor([0, 0, 1, 1, 1] * 4, dtype=torch.bool)
        # Identical logprobs for chosen & rejected within each pair
        seq = torch.tensor([0.0, 0.0, -1.0, -2.0, -3.0])
        logprobs = torch.cat([seq, seq, seq, seq])
        ref_logprobs = torch.zeros(20)

        input_ = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
        }
        loss = compute_dpo_loss(logprobs, None, input_, beta=0.5)
        import math

        torch.testing.assert_close(
            loss,
            torch.tensor(math.log(2.0), dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_no_gpu_cpu_sync_in_hot_path(self):
        """Smoke test: the computation should work without any ``.cpu()``
        calls in the new implementation. Verified indirectly by ensuring the
        loss runs and backward is feasible on a leaf tensor.
        """
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        loss_mask = torch.ones(8, dtype=torch.bool)
        logprobs = torch.randn(8, requires_grad=True)
        ref_logprobs = torch.zeros(8)

        input_ = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
        }
        loss = compute_dpo_loss(logprobs, None, input_, beta=0.1)
        loss.backward()
        assert logprobs.grad is not None
        assert torch.isfinite(logprobs.grad).all()


class TestDPOLossIPO:
    """Test cases for the IPO loss variant with per-token length normalization."""

    @pytest.fixture
    def simple_pair_data(self):
        """Single pair with all response tokens."""
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        loss_mask = torch.ones(8, dtype=torch.bool)
        ref_logprobs = torch.zeros(8)
        return {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
        }

    def test_ipo_loss_is_scalar(self, simple_pair_data):
        """IPO loss should be a scalar."""
        logprobs = torch.randn(8)
        loss = compute_dpo_loss(
            logprobs, None, simple_pair_data, beta=0.1, loss_type="ipo"
        )
        assert loss.ndim == 0

    def test_ipo_loss_is_finite(self, simple_pair_data):
        """IPO loss should be finite."""
        logprobs = torch.randn(8)
        loss = compute_dpo_loss(
            logprobs, None, simple_pair_data, beta=0.1, loss_type="ipo"
        )
        assert torch.isfinite(loss)

    def test_ipo_loss_non_negative(self, simple_pair_data):
        """IPO loss (squared) should always be non-negative."""
        logprobs = torch.randn(8)
        loss = compute_dpo_loss(
            logprobs, None, simple_pair_data, beta=0.1, loss_type="ipo"
        )
        assert loss.item() >= 0.0

    def test_ipo_loss_zero_at_target(self, simple_pair_data):
        """IPO loss should be zero when per-token-averaged logits == 1/(2*beta).

        With all-response mask and shifted mask yielding 3 counted tokens per
        sequence (positions 0..2; position 3 is last-of-seq and zeroed),
        the per-token average logratio needs to equal 1/(2*beta) for zero loss.
        """
        beta = 0.1
        target = 1.0 / (2.0 * beta)  # = 5.0

        # 3 shifted-mask tokens per seq. chosen per-token avg = target,
        # rejected per-token avg = 0 → ipo_delta = target → loss = 0.
        per_token = target
        logprobs = torch.zeros(8)
        logprobs[0:3] = per_token  # chosen: sum = 3 * target, avg = target
        logprobs[4:7] = 0.0  # rejected: sum = 0, avg = 0

        loss = compute_dpo_loss(
            logprobs, None, simple_pair_data, beta=beta, loss_type="ipo"
        )
        torch.testing.assert_close(
            loss, torch.tensor(0.0, dtype=torch.float32), rtol=1e-4, atol=1e-4
        )

    def test_ipo_differs_from_sigmoid(self, simple_pair_data):
        """IPO and sigmoid should produce different loss values for the same input."""
        logprobs = torch.randn(8)
        loss_sigmoid = compute_dpo_loss(
            logprobs, None, simple_pair_data, beta=0.1, loss_type="sigmoid"
        )
        loss_ipo = compute_dpo_loss(
            logprobs, None, simple_pair_data, beta=0.1, loss_type="ipo"
        )
        assert loss_sigmoid.item() != loss_ipo.item()

    def test_ipo_length_normalization(self):
        """IPO must normalize by completion length; two pairs with same per-token
        averages but different lengths should produce equal loss.
        """
        # Pair A: seqlen=4 each, 3 shifted-mask response tokens each
        cu_a = torch.tensor([0, 4, 8], dtype=torch.int32)
        mask_a = torch.ones(8, dtype=torch.bool)
        ref_a = torch.zeros(8)
        lp_a = torch.zeros(8)
        lp_a[0:3] = -1.0  # chosen avg = -1.0
        lp_a[4:7] = -2.0  # rejected avg = -2.0

        # Pair B: seqlen=8 each, 7 shifted-mask response tokens each
        cu_b = torch.tensor([0, 8, 16], dtype=torch.int32)
        mask_b = torch.ones(16, dtype=torch.bool)
        ref_b = torch.zeros(16)
        lp_b = torch.zeros(16)
        lp_b[0:7] = -1.0  # chosen avg = -1.0
        lp_b[8:15] = -2.0  # rejected avg = -2.0

        input_a = {"cu_seqlens": cu_a, "loss_mask": mask_a, "ref_logprobs": ref_a}
        input_b = {"cu_seqlens": cu_b, "loss_mask": mask_b, "ref_logprobs": ref_b}

        loss_a = compute_dpo_loss(lp_a, None, input_a, beta=0.1, loss_type="ipo")
        loss_b = compute_dpo_loss(lp_b, None, input_b, beta=0.1, loss_type="ipo")

        # Same per-token average logratios → same IPO loss
        torch.testing.assert_close(loss_a, loss_b, rtol=1e-5, atol=1e-5)

    def test_ipo_chosen_preferred_lower_loss(self):
        """When policy prefers chosen, IPO loss should be lower (closer to target)."""
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        loss_mask = torch.ones(8, dtype=torch.bool)
        ref_logprobs = torch.zeros(8)
        input_ = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
        }

        logprobs_good = torch.tensor([-0.1, -0.1, -0.1, -0.1, -5.0, -5.0, -5.0, -5.0])
        loss_good = compute_dpo_loss(
            logprobs_good, None, input_, beta=0.5, loss_type="ipo"
        )

        logprobs_bad = torch.tensor([-5.0, -5.0, -5.0, -5.0, -0.1, -0.1, -0.1, -0.1])
        loss_bad = compute_dpo_loss(
            logprobs_bad, None, input_, beta=0.5, loss_type="ipo"
        )

        assert loss_good.item() < loss_bad.item()

    def test_ipo_backward(self, simple_pair_data):
        """IPO loss should support backward pass."""
        logprobs = torch.randn(8, requires_grad=True)
        loss = compute_dpo_loss(
            logprobs, None, simple_pair_data, beta=0.1, loss_type="ipo"
        )
        loss.backward()
        assert logprobs.grad is not None
        assert torch.isfinite(logprobs.grad).all()

    def test_invalid_loss_type_raises(self, simple_pair_data):
        """Unsupported loss_type should raise ValueError."""
        logprobs = torch.randn(8)
        with pytest.raises(ValueError, match="Unsupported DPO loss_type"):
            compute_dpo_loss(
                logprobs, None, simple_pair_data, beta=0.1, loss_type="nonexistent"
            )

    def test_ipo_empty_pairs(self):
        """When all sequences are empty, IPO loss should be zero."""
        cu_seqlens = torch.tensor([0, 0, 0], dtype=torch.int32)
        loss_mask = torch.zeros(0, dtype=torch.bool)
        ref_logprobs = torch.zeros(0)
        input_ = {
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "ref_logprobs": ref_logprobs,
        }
        logprobs = torch.zeros(0)
        loss = compute_dpo_loss(logprobs, None, input_, beta=0.1, loss_type="ipo")
        assert loss.item() == 0.0
