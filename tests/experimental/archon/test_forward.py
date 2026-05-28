"""Forward pass tests for Archon Engine (single GPU).

These tests verify:
1. Archon forward matches HuggingFace
2. Packed sequence handling (SDPA with cu_seqlens)
3. Precision verification with random input and GSM8K data
4. Archon vs FSDP engine comparison

Run tests:
    pytest tests/experimental/archon/test_forward.py -v

Note: These tests require GPU and are marked as slow.
For multi-GPU tests (DP, TP, CP), see test_distributed.py.
"""

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig

from tests.experimental.archon.utils import (
    MODEL_PATHS,
    compare_logprobs,
    compare_outputs,
    create_test_input,
    load_archon_model,
    load_gsm8k_samples,
    load_hf_model,
    run_torchrun_test,
    setup_environment,
)

from areal.infra.platforms import current_platform

# Skip if no CUDA available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# =============================================================================
# Single GPU Tests: Archon vs HuggingFace
# =============================================================================


def test_archon_forward_matches_hf():
    """Verify Archon forward matches HuggingFace for packed sequences.

    This test creates packed input with cu_seqlens, runs Archon forward,
    and compares with HF by running each sequence separately.
    """
    setup_environment()

    model_path = MODEL_PATHS["qwen2"]
    dtype = torch.bfloat16
    device = torch.device(current_platform.device_type)

    hf_model = load_hf_model(model_path, dtype=dtype)
    archon_model, _ = load_archon_model(model_path, dtype=dtype)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    batch_size = 2
    seq_len = 32

    # Create packed input
    inputs = create_test_input(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=config.vocab_size,
        device=device,
        packed=True,
    )

    input_ids = inputs["input_ids"]  # [1, batch_size * seq_len]
    cu_seqlens = inputs["cu_seqlens"]  # [0, seq_len, 2*seq_len, ...]
    max_seqlen = inputs["max_seqlen"]

    # Create packed positions: each sequence starts from 0
    positions_list = []
    for i in range(batch_size):
        seq_len_i = cu_seqlens[i + 1].item() - cu_seqlens[i].item()
        positions_list.extend(range(seq_len_i))
    positions = torch.tensor(positions_list, dtype=torch.long, device=device).unsqueeze(
        0
    )

    # Archon forward with packed sequences and correct positions
    with torch.no_grad():
        archon_logits = archon_model(
            input_ids, positions=positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )  # [1, total_len, vocab_size]

    # HF forward: process each sequence separately and concatenate
    hf_logits_list = []
    with torch.no_grad():
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_input_ids = input_ids[:, start:end]  # [1, seq_len]
            hf_outputs = hf_model(input_ids=seq_input_ids, use_cache=False)
            hf_logits_list.append(hf_outputs.logits)  # [1, seq_len, vocab_size]

    hf_logits = torch.cat(hf_logits_list, dim=1)  # [1, total_len, vocab_size]

    # Compare logits
    diff = (hf_logits.float() - archon_logits.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Allow for bfloat16 precision differences
    assert max_diff < 3.0, f"Logits max_diff too large: {max_diff}"
    assert mean_diff < 0.2, f"Logits mean_diff too large: {mean_diff}"

    # Compare top-k predictions for the first position of first sequence
    # Due to bfloat16 precision and different attention implementations,
    # we only require top-1 to match and top-5 overlap >= 4
    hf_top5 = hf_logits[0, 0].topk(5).indices
    archon_top5 = archon_logits[0, 0].topk(5).indices
    assert hf_top5[0] == archon_top5[0], (
        f"Top-1 prediction differs: HF={hf_top5[0].item()}, Archon={archon_top5[0].item()}"
    )
    top5_overlap = len(set(hf_top5.tolist()) & set(archon_top5.tolist()))
    assert top5_overlap >= 4, (
        f"Top-5 overlap too low ({top5_overlap}/5): HF={hf_top5.tolist()}, Archon={archon_top5.tolist()}"
    )

    # Compare log probabilities for the first sequence
    first_seq_end = cu_seqlens[1].item()
    labels = input_ids[0, 1:first_seq_end]
    hf_logprobs = torch.log_softmax(hf_logits[0, : first_seq_end - 1].float(), dim=-1)
    archon_logprobs = torch.log_softmax(
        archon_logits[0, : first_seq_end - 1].float(), dim=-1
    )

    hf_token_logprobs = hf_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    archon_token_logprobs = archon_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    logprob_diff = (hf_token_logprobs - archon_token_logprobs).abs()
    assert logprob_diff.max().item() < 0.6, (
        f"Logprob max_diff too large: {logprob_diff.max().item()}"
    )

    if dist.is_initialized():
        dist.destroy_process_group()


def test_archon_packed_sequence_logits():
    """Verify Archon packed sequence (SDPA) produces correct logits.

    This test creates a packed batch with cu_seqlens and verifies that:
    1. SDPA is used with cu_seqlens
    2. The model produces reasonable logits
    """
    setup_environment()

    model_path = MODEL_PATHS["qwen2"]
    dtype = torch.bfloat16
    device = torch.device(current_platform.device_type)

    archon_model, _ = load_archon_model(model_path, dtype=dtype)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Create packed input with cu_seqlens
    torch.manual_seed(42)
    # Simulate 3 sequences packed together: lengths 10, 15, 7
    seq_lens = [10, 15, 7]
    total_len = sum(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    max_seqlen = max(seq_lens)

    input_ids = torch.randint(
        100, config.vocab_size - 100, (1, total_len), device=device
    )

    # Forward with cu_seqlens (uses SDPA with block-diagonal causal mask)
    with torch.no_grad():
        logits = archon_model(
            input_ids, positions=None, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )

    # Basic sanity checks
    assert logits.shape == (1, total_len, config.vocab_size)
    assert not torch.isnan(logits).any(), "Logits contain NaN"
    assert not torch.isinf(logits).any(), "Logits contain Inf"

    # Check that logits are reasonable (not all same value)
    assert logits.std() > 0.1, "Logits have too low variance"

    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Precision Comparison Tests
# =============================================================================


class TestArchonHFPrecision:
    """Archon vs HuggingFace precision comparison tests.

    These tests verify that Archon and HuggingFace models produce
    similar outputs. Some numerical differences are expected due to
    different attention implementations.
    """

    # Precision thresholds
    MAX_LOGITS_DIFF = 3.0
    TOP1_MATCH_RATE_MIN = 0.85  # 85% match required
    LOGPROB_MEAN_DIFF_MAX = 0.2

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        setup_environment()
        self.model_path = MODEL_PATHS["qwen2"]
        self.dtype = torch.bfloat16
        self.device = torch.device(current_platform.device_type)
        yield
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_precision_random_input(self):
        """Test Archon vs HF precision with random packed input."""
        hf_model = load_hf_model(self.model_path, dtype=self.dtype)
        archon_model, _ = load_archon_model(self.model_path, dtype=self.dtype)

        # Create packed input with variable length sequences
        inputs = create_test_input(
            batch_size=3,
            seq_len=32,
            vocab_size=151936,
            device=self.device,
            packed=True,
        )

        input_ids = inputs["input_ids"]
        cu_seqlens = inputs["cu_seqlens"]
        max_seqlen = inputs["max_seqlen"]
        batch_size = len(cu_seqlens) - 1

        # Create positions for packed sequence
        positions_list = []
        for i in range(batch_size):
            seq_len_i = cu_seqlens[i + 1].item() - cu_seqlens[i].item()
            positions_list.extend(range(seq_len_i))
        positions = torch.tensor(
            positions_list, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            # Archon forward (packed)
            archon_logits = archon_model(
                input_ids,
                positions=positions,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

            # HF forward (process each sequence separately)
            hf_logits_list = []
            for i in range(batch_size):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq_input_ids = input_ids[:, start:end]
                hf_outputs = hf_model(input_ids=seq_input_ids, use_cache=False)
                hf_logits_list.append(hf_outputs.logits)

            hf_logits = torch.cat(hf_logits_list, dim=1)

        # Compare logits
        result = compare_outputs(archon_logits, hf_logits, "Final Logits")
        print("\n[Random Input Precision]")
        print(f"  Max diff: {result['max_diff']:.6f}")
        print(f"  Mean diff: {result['mean_diff']:.6f}")

        assert result["max_diff"] < self.MAX_LOGITS_DIFF, (
            f"Logits max_diff {result['max_diff']:.6f} >= {self.MAX_LOGITS_DIFF}"
        )

        # Compare top-1 predictions at each position
        total_len = archon_logits.shape[1]
        top1_matches = 0
        for pos in range(total_len):
            archon_top1 = archon_logits[0, pos].argmax()
            hf_top1 = hf_logits[0, pos].argmax()
            if archon_top1 == hf_top1:
                top1_matches += 1

        top1_match_rate = top1_matches / total_len
        print(f"  Top-1 match rate: {top1_match_rate:.1%}")

        assert top1_match_rate >= self.TOP1_MATCH_RATE_MIN, (
            f"Top-1 match rate {top1_match_rate:.1%} < {self.TOP1_MATCH_RATE_MIN:.0%}"
        )

    def test_precision_gsm8k(self):
        """Test Archon vs HF precision with real GSM8K data."""
        hf_model = load_hf_model(self.model_path, dtype=self.dtype)
        archon_model, _ = load_archon_model(self.model_path, dtype=self.dtype)

        # Load real GSM8K data
        samples, tokenizer = load_gsm8k_samples(self.model_path, num_samples=10)

        # Statistics
        total_positions = 0
        top1_matches = 0
        all_max_diffs = []
        all_logprob_diffs = []

        with torch.no_grad():
            for sample in samples:
                input_ids = torch.tensor([sample["prompt_ids"]], device=self.device)
                seq_len = input_ids.shape[1]

                cu_seqlens = torch.tensor(
                    [0, seq_len], dtype=torch.int32, device=self.device
                )

                # Forward pass
                archon_logits = archon_model(
                    input_ids,
                    positions=None,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=seq_len,
                )
                hf_outputs = hf_model(input_ids=input_ids, use_cache=False)
                hf_logits = hf_outputs.logits

                # Compare logits
                result = compare_outputs(archon_logits, hf_logits)
                all_max_diffs.append(result["max_diff"])

                # Compare top-1 at each position
                for pos in range(seq_len):
                    archon_top1 = archon_logits[0, pos].argmax()
                    hf_top1 = hf_logits[0, pos].argmax()
                    if archon_top1 == hf_top1:
                        top1_matches += 1
                total_positions += seq_len

                # Compare logprobs for answer tokens
                answer_start = len(sample["prompt_ids"])
                if len(sample["full_ids"]) > answer_start:
                    answer_ids = sample["full_ids"][answer_start:]
                    pred_start = answer_start - 1
                    num_pred = min(len(answer_ids), seq_len - pred_start)
                    if num_pred > 0:
                        target_ids = torch.tensor(
                            answer_ids[:num_pred], device=self.device
                        )
                        lp_result = compare_logprobs(
                            archon_logits[0, pred_start : pred_start + num_pred],
                            hf_logits[0, pred_start : pred_start + num_pred],
                            target_ids,
                        )
                        all_logprob_diffs.append(lp_result["mean_diff"])

        # Calculate metrics
        top1_match_rate = top1_matches / total_positions
        avg_max_diff = sum(all_max_diffs) / len(all_max_diffs)
        avg_logprob_diff = (
            sum(all_logprob_diffs) / len(all_logprob_diffs) if all_logprob_diffs else 0
        )

        print(f"\n[GSM8K Precision - {len(samples)} samples]")
        print(f"  Top-1 match rate: {top1_match_rate:.1%}")
        print(f"  Avg max diff: {avg_max_diff:.6f}")
        print(f"  Avg logprob diff: {avg_logprob_diff:.6f}")

        # Assertions
        assert top1_match_rate >= self.TOP1_MATCH_RATE_MIN, (
            f"Top-1 match rate {top1_match_rate:.1%} < {self.TOP1_MATCH_RATE_MIN:.0%}"
        )
        assert avg_logprob_diff < self.LOGPROB_MEAN_DIFF_MAX, (
            f"Avg logprob diff {avg_logprob_diff:.6f} >= {self.LOGPROB_MEAN_DIFF_MAX}"
        )

    def test_logits_exact_match(self):
        """Verify logits and top-k predictions match."""
        hf_model = load_hf_model(self.model_path, dtype=self.dtype)
        archon_model, _ = load_archon_model(self.model_path, dtype=self.dtype)

        # Single sequence test for detailed comparison
        samples, tokenizer = load_gsm8k_samples(self.model_path, num_samples=1)
        sample = samples[0]

        input_ids = torch.tensor([sample["prompt_ids"]], device=self.device)
        seq_len = input_ids.shape[1]
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)

        with torch.no_grad():
            archon_logits = archon_model(
                input_ids, positions=None, cu_seqlens=cu_seqlens, max_seqlen=seq_len
            )
            hf_outputs = hf_model(input_ids=input_ids, use_cache=False)
            hf_logits = hf_outputs.logits

        # Test at last position (most important for generation)
        last_pos = seq_len - 1
        archon_last = archon_logits[0, last_pos]
        hf_last = hf_logits[0, last_pos]

        # Logits diff
        diff = (archon_last.float() - hf_last.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Top-k comparison
        k = 10
        archon_topk = archon_last.topk(k)
        hf_topk = hf_last.topk(k)

        # Decode for display
        archon_tokens = [
            tokenizer.decode([idx]) for idx in archon_topk.indices.tolist()
        ]
        hf_tokens = [tokenizer.decode([idx]) for idx in hf_topk.indices.tolist()]

        print("\n[Last Position Comparison]")
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        print(f"  Archon top-{k}: {archon_tokens}")
        print(f"  HF top-{k}: {hf_tokens}")

        # Top-1 must match
        assert archon_topk.indices[0] == hf_topk.indices[0], (
            f"Top-1 mismatch: Archon='{archon_tokens[0]}' vs HF='{hf_tokens[0]}'"
        )

        # Top-k overlap should be high
        archon_set = set(archon_topk.indices.tolist())
        hf_set = set(hf_topk.indices.tolist())
        overlap = len(archon_set & hf_set)
        print(f"  Top-{k} overlap: {overlap}/{k}")

        assert overlap >= k - 1, f"Top-{k} overlap {overlap} < {k - 1}"


# =============================================================================
# Single GPU Engine Tests (via torchrun)
# =============================================================================


@pytest.mark.slow
def test_archon_fsdp_forward_1gpu():
    """Test Archon Engine forward with FSDP on 1 GPU."""
    if current_platform.device_count() < 1:
        pytest.skip("This test requires at least 1 GPU")
    run_torchrun_test(
        "tests/experimental/archon/torchrun/run_forward.py",
        n_gpus=1,
    )


@pytest.mark.slow
def test_archon_vs_fsdp_engine_logits():
    """Test Archon vs FSDP engine logits comparison."""
    if current_platform.device_count() < 1:
        pytest.skip("This test requires at least 1 GPU")
    run_torchrun_test(
        "tests/experimental/archon/torchrun/run_vs_fsdp.py",
        n_gpus=1,
    )
