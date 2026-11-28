# SPDX-License-Identifier: Apache-2.0
"""Share retained target probabilities across DSpark acceptance, PPO and row storage."""

from dataclasses import dataclass

import torch


@dataclass
class CommittedMaskRows:
    producer: object
    ids: torch.Tensor
    counts: torch.Tensor
    versions: list
    leases: list
    stride: int

    def copy_to_host(self, copy_tensor):
        # IDs/counts are per-call allocations. The producer enqueues its bounded D2H
        # after the scheduler's completion event, and only for visible tokens.
        return self

    def consume(self, batch, commits):
        rows, tokens, owners = [], [], []
        for i, commit in enumerate(commits):
            if commit is None:
                continue
            n = len(commit.token_ids)
            if n > self.stride:
                raise ValueError("Committed token count exceeds DSpark verify stride")
            rows.extend(range(i * self.stride, i * self.stride + n))
            tokens.extend(commit.token_ids)
            owners.extend([i] * n)
        if not rows:
            return
        index = torch.tensor(rows, device=self.ids.device, dtype=torch.long)
        from areal.engine.sglang_sampling_mask import publish_sampling_masks

        refs = publish_sampling_masks(
            self.producer,
            self.ids.index_select(0, index),
            self.counts.index_select(0, index),
            torch.tensor(tokens, device=self.ids.device, dtype=torch.int32),
            [self.versions[i] for i in owners],
            [self.leases[i] for i in owners],
        )
        for i, ref in zip(owners, refs, strict=True):
            req = batch.reqs[i]
            if req.customized_info is None:
                req.customized_info = {}
            req.customized_info.setdefault("sampling_mask_refs", []).append(ref)


@dataclass
class VerifyPolicy:
    logprobs: torch.Tensor
    rows: CommittedMaskRows

    def finish(self, output, tokens, commit_lens):
        rows = self.rows
        selected = rows.ids == tokens.reshape(-1, 1)
        logp = self.logprobs.masked_fill(~selected, -torch.inf).amax(-1)
        logp = logp.view(-1, rows.stride)
        valid = torch.arange(rows.stride, device=tokens.device) < commit_lens[:, None]
        torch._assert_async(
            (torch.isfinite(logp) | ~valid).all(),
            "DSpark committed token outside the restricted target policy",
        )
        output.next_token_logprobs = logp.masked_fill(~valid, 0)
        output.auxiliary_device_output = rows if rows.producer is not None else None


def prepare_verify_policy(logits, info, stride, bound, producer, versions, leases):
    from areal.engine.sglang_sampling_mask import candidate_distribution

    ids, logp, counts = candidate_distribution(
        logits,
        info.temperatures.reshape(-1).repeat_interleave(stride),
        info.top_ps.repeat_interleave(stride),
        info.top_ks.repeat_interleave(stride),
        candidate_bound=bound,
    )
    rows = CommittedMaskRows(producer, ids, counts, versions, leases, stride)
    return VerifyPolicy(logp, rows)
