# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for profiling tools."""

import random


def generate_random_seq_lens(
    batch_size: int,
    total_len: int,
    min_seq_len: int = 64,
) -> list[int]:
    """Generate random sequence lengths that sum to total_len.

    Generates variable-length sequences for profiling with packed/padded batches.
    Each sequence length is randomized around the average remaining length.

    Args:
        batch_size: Number of sequences to generate.
        total_len: Target total length (sum of all sequence lengths).
        min_seq_len: Minimum length for each sequence.

    Returns:
        List of sequence lengths that sum to total_len.
    """
    seq_lens = []
    remaining = total_len

    for i in range(batch_size - 1):
        avg_remaining = remaining // (batch_size - i)
        min_len = max(min_seq_len, avg_remaining // 2)
        max_len = min(remaining - min_seq_len * (batch_size - i - 1), avg_remaining * 2)
        length = random.randint(min_len, max_len)
        seq_lens.append(length)
        remaining -= length

    seq_lens.append(remaining)
    return seq_lens
