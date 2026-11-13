# SPDX-License-Identifier: Apache-2.0
"""SGLang's opt-in sampler for the actual retained-action policy.

Top-p and top-k use joint truncation against full-vocabulary cumulative mass,
matching the pinned SGLang backend's filter_apply_order='joint'. The sampler
computes old logprob from exactly the logits and support used to sample.
"""

import os

import torch


def candidate_distribution(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    *,
    candidate_bound: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sorted IDs, restricted logprobs, and counts without a device sync.

    candidate_bound is the explicit server sampling top-k (or vocab size).
    A request exceeding that bound fails; it is never silently truncated.
    """
    if logits.ndim != 2 or not 0 < candidate_bound <= logits.shape[-1]:
        raise ValueError("Invalid sampling support candidate bound")
    torch._assert_async(
        ((temperatures >= 1e-6) & torch.isfinite(temperatures)).all(),
        "Invalid sampling temperature",
    )
    torch._assert_async(((top_ps > 0) & (top_ps <= 1)).all(), "Invalid sampling top_p")
    effective_k = torch.where(top_ks <= 0, logits.shape[-1], top_ks).clamp_max(
        logits.shape[-1]
    )
    torch._assert_async(
        (effective_k <= candidate_bound).all(),
        "Sampling request exceeds explicit top_k capacity",
    )
    x = logits.float() / temperatures.reshape(-1, 1)
    values, ids = torch.topk(x, candidate_bound, dim=-1, sorted=True)
    full_logz = torch.logsumexp(x, dim=-1, keepdim=True)
    torch._assert_async(
        torch.isfinite(full_logz).all(), "Sampling logits have a non-finite normalizer"
    )
    probabilities = (values - full_logz).exp()
    # Use unmodified probabilities for the prefix, including when top-k is active.
    prefix = probabilities.cumsum(-1) - probabilities
    keep = (
        (torch.arange(candidate_bound, device=x.device)[None, :] < effective_k[:, None])
        & ((top_ps[:, None] == 1) | (prefix <= top_ps[:, None]))
        & torch.isfinite(values)
    )
    counts = keep.sum(-1, dtype=torch.int64)
    torch._assert_async((counts > 0).all(), "Sampling mask contains an empty support")
    restricted = values.masked_fill(~keep, -torch.inf)
    return (
        ids.to(torch.int32),
        restricted - torch.logsumexp(restricted, dim=-1, keepdim=True),
        counts,
    )


def accumulate_reference_metadata(
    state, metadata: dict, key: str, increment: list
) -> None:
    """Accumulate internal nonstreaming chunks, including the final EOS chunk."""
    if key != "sampling_mask_refs":
        metadata[key] = increment
        return
    history = getattr(state, "_areal_sampling_mask_refs", None)
    if history is None:
        history = []
        state._areal_sampling_mask_refs = history
    history.extend(increment)
    metadata[key] = list(history)


def sample_candidate_columns(logp, sampling_seed, positions):
    """Draw within the saved support, honoring SGLang's per-request seeds."""
    if sampling_seed is None:
        return torch.multinomial(logp.exp(), 1)
    from sglang.srt.layers.sampler import multinomial_with_seed

    return multinomial_with_seed(logp, sampling_seed, positions).reshape(-1, 1)


def merge_sampling_batches(left, right, original_merge) -> None:
    """Preserve seeded requests when SGLang merges independently built batches."""
    versions = left._areal_mask_versions + right._areal_mask_versions
    if left.sampling_seed is not None or right.sampling_seed is not None:
        for batch in (left, right):
            if batch.sampling_seed is None:
                batch.sampling_seed = torch.randint(
                    0, 2**31, (len(batch),), dtype=torch.int64, device=batch.device
                )
    original_merge(left, right)
    left._areal_mask_versions = versions


def register_backend() -> None:
    """Register through SGLang's sampler factory API in each worker process."""
    from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
    from sglang.srt.layers.sampler import (
        Sampler,
        register_sampler_backend,
    )
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    if getattr(SamplingBatchInfo, "_areal_mask_installed", False):
        return
    original_create = SamplingBatchInfo.from_schedule_batch.__func__
    original_filter = SamplingBatchInfo.filter_batch
    original_merge = SamplingBatchInfo.merge_batch
    original_copy = SamplingBatchInfo.copy_for_forward

    @classmethod
    def from_schedule_batch(cls, batch, vocab_size):
        for request in batch.reqs:
            if request.rid.startswith(HEALTH_CHECK_RID_PREFIX):
                # SGLang's own readiness probes are not training rollouts.
                request.sampling_params.custom_params = {
                    "areal_sampling_mask_version": -1
                }
                request.sampling_params.temperature = 1.0
                request.sampling_params.top_p = 1.0
                request.sampling_params.top_k = 1
        result = original_create(cls, batch, vocab_size)
        # SGLang otherwise ignores request seeds outside deterministic mode.
        # Preserve explicit seeds, including mixed seeded/unseeded batches.
        seeds = [request.sampling_params.sampling_seed for request in batch.reqs]
        if result.sampling_seed is None and any(seed is not None for seed in seeds):
            random_seeds = torch.randint(0, 2**31, (len(seeds),)).tolist()
            result.sampling_seed = torch.tensor(
                [
                    seed if seed is not None else random_seeds[i]
                    for i, seed in enumerate(seeds)
                ],
                dtype=torch.int64,
                device=batch.device,
            )
        versions = []
        for request in batch.reqs:
            params = request.sampling_params
            custom = params.custom_params or {}
            if "areal_sampling_mask_version" not in custom:
                raise ValueError(
                    "sampling_mask_transfer server received a request without version metadata"
                )
            if (
                params.frequency_penalty != 0
                or params.presence_penalty != 0
                or params.repetition_penalty != 1
                or params.logit_bias
                or params.min_p != 0
            ):
                raise ValueError(
                    "sampling_mask_transfer does not yet support sampling penalties, min-p, or logit bias"
                )
            versions.append(int(custom["areal_sampling_mask_version"]))
        result._areal_mask_versions = versions
        return result

    def filter_batch(self, indices, indices_device):
        versions = [self._areal_mask_versions[i] for i in indices]
        original_filter(self, indices, indices_device)
        self._areal_mask_versions = versions

    def merge_batch(self, other):
        merge_sampling_batches(self, other, original_merge)

    def copy_for_forward(self):
        result = original_copy(self)
        result._areal_mask_versions = self._areal_mask_versions.copy()
        return result

    SamplingBatchInfo.copy_for_forward = copy_for_forward
    SamplingBatchInfo.from_schedule_batch = from_schedule_batch
    SamplingBatchInfo.filter_batch = filter_batch
    SamplingBatchInfo.merge_batch = merge_batch
    SamplingBatchInfo._areal_mask_installed = True

    class SamplingMaskSampler(Sampler):
        def __init__(self):
            super().__init__()
            from areal.engine.sglang_sampling_mask_plugin import runtime_settings
            from areal.infra.sampling_mask.service import MaskProducer

            backend, pp_size, attn_cp_size, speculative_algorithm = runtime_settings()
            if backend != "areal_sampling_mask":
                raise ValueError(
                    "sampling_mask_transfer sampler backend was overridden"
                )
            if speculative_algorithm or pp_size != 1 or attn_cp_size != 1:
                raise ValueError(
                    "sampling_mask_transfer requires non-speculative SGLang PP=1"
                )
            self.explicit_top_k = int(os.environ["AREAL_SAMPLING_MASK_TOP_K"])
            self.producer = (
                MaskProducer(
                    torch.cuda.current_device(),
                    max_sessions=int(os.environ["AREAL_SAMPLING_MASK_RECEIVERS"]),
                    retained_versions=int(
                        os.environ.get("AREAL_SAMPLING_MASK_RETAINED_VERSIONS", "2")
                    ),
                )
                if torch.distributed.get_rank(self.tp_sync_group) == 0
                else None
            )

        def forward(
            self,
            logits_output,
            sampling_info,
            return_logprob,
            top_logprobs_nums,
            token_ids_logprobs,
            positions,
        ):
            if sampling_info.has_custom_logit_processor or sampling_info.grammars:
                raise ValueError(
                    "sampling_mask_transfer does not support custom processors or grammars"
                )
            if any(top_logprobs_nums or ()) or any(token_ids_logprobs or ()):
                raise ValueError(
                    "sampling_mask_transfer does not support auxiliary top/token logprob requests"
                )
            logits = logits_output.next_token_logits
            bound = (
                min(self.explicit_top_k, logits.shape[-1])
                if self.explicit_top_k > 0
                else logits.shape[-1]
            )
            ids, logp, counts = candidate_distribution(
                logits,
                sampling_info.temperatures,
                sampling_info.top_ps,
                sampling_info.top_ks,
                candidate_bound=bound,
            )
            column = sample_candidate_columns(
                logp, sampling_info.sampling_seed, positions
            )
            tokens = ids.gather(1, column).reshape(-1)
            self._sync_token_ids_across_tp(tokens, sampling_info)
            # Sync may choose another TP rank's draw. Recompute its column so
            # old_logprob and the persisted support refer to the returned token.
            matches = ids == tokens[:, None]
            torch._assert_async(
                (matches & torch.isfinite(logp)).any(-1).all(),
                "Synchronized token is outside sampling support",
            )
            column = matches.to(torch.int32).argmax(-1, keepdim=True)
            logits_output.next_token_logprobs = logp.gather(1, column).reshape(-1)
            logits_output.customized_info = {
                "sampling_mask_refs": self.producer.publish(
                    ids, counts, tokens, sampling_info._areal_mask_versions
                )
                if self.producer is not None
                else [[0] * 5 for _ in range(ids.shape[0])]
            }
            return tokens

    register_sampler_backend("areal_sampling_mask", SamplingMaskSampler)
