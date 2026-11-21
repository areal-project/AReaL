# SPDX-License-Identifier: Apache-2.0
"""Padding helpers shared by the two sampler top-k transport paths.

An absent top-k slot carries no probability mass, and ``exp(-inf) == 0`` is what
expresses that: padding the log-probs with ``0.0`` would mean ``exp(0) == 1`` and
silently inflate the sampler head mass.  Token ids are padded with 0; the trainer
still gathers a log-probability for that id, so an absent slot only stays inert
because its row is masked out of the loss downstream.
"""

NEG_INF_PAD = float("-inf")


def zero_mass_row_count(rows: list[list[float]]) -> int:
    """Count rows whose log-probs carry no probability mass (all -inf)."""
    return sum(1 for row in rows if not row or all(lp <= -1e30 for lp in row))


def pad_topk_ids(token_ids: list[int], k: int) -> list[int]:
    """Right-pad a top-k token-id row to length *k* with the padding id 0.

    Callers must cap with :func:`truncate_topk` first; this only pads.
    """
    ids = list(token_ids[:k])
    return ids + [0] * max(0, k - len(ids))


def pad_topk_logps(logps: list[float], k: int) -> list[float]:
    """Right-pad a top-k log-prob row to length *k* with ``-inf`` (q = 0)."""
    ps = list(logps[:k])
    return ps + [NEG_INF_PAD] * max(0, k - len(ps))


def truncate_topk(
    ids: list[int], logps: list[float], k: int
) -> tuple[list[int], list[float]]:
    """Keep the *k* highest-logprob entries, preserving order only when uncut.

    vLLM's raw logprobs are "[sampled token] + top-k" (1 + k entries); the API
    layer de-duplicates token ids, so a row only lands exactly at *k* afterwards.
    A row wider than *k* therefore has to be sorted by logprob and cut to the top
    *k*: a ``[:k]`` slice keeps the sampled token plus the top ``k-1`` and
    silently drops the true k-th alternative.  A row already at or below *k* is
    returned untouched, so the measured exactly-k layout stays bit-identical.

    The sampled token is *not* removed: with it in the row the head covers the
    top k of the sampler distribution, and dropping it would cut most of the
    head mass.
    """
    if len(ids) != len(logps):
        raise ValueError(
            f"Sampler head row is misaligned: {len(ids)} ids vs {len(logps)} logps."
        )
    if k <= 0 or len(ids) <= k:
        return list(ids), list(logps)
    order = sorted(range(len(ids)), key=lambda i: logps[i], reverse=True)[:k]
    return [ids[i] for i in order], [logps[i] for i in order]
