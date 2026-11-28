# SPDX-License-Identifier: Apache-2.0
"""SGLang's opt-in sampler for the actual retained-action policy.

Top-p and top-k use joint truncation against full-vocabulary cumulative mass,
matching the pinned SGLang backend's filter_apply_order='joint'. The sampler
computes old logprob from exactly the logits and support used to sample.
"""

import os
from functools import cache

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
    leases = left._areal_mask_leases + right._areal_mask_leases
    if left.sampling_seed is not None or right.sampling_seed is not None:
        for batch in (left, right):
            if batch.sampling_seed is None:
                batch.sampling_seed = torch.randint(
                    0, 2**31, (len(batch),), dtype=torch.int64, device=batch.device
                )
    original_merge(left, right)
    left._areal_mask_versions = versions
    left._areal_mask_leases = leases


def publish_sampling_masks(producer, ids, counts, tokens, versions, leases):
    """Readiness probes never own training data or acquire a workflow lease."""
    if len(leases) != len(versions) or len(leases) != ids.shape[0]:
        raise ValueError("Sampling-mask lease metadata is misaligned")
    if any(
        lease is None and version != -1
        for lease, version in zip(leases, versions, strict=True)
    ):
        raise ValueError("Training request is missing sampling-mask workflow lease")
    if all(lease is not None for lease in leases):
        return producer.publish(ids, counts, tokens, versions, leases)
    live = [i for i, lease in enumerate(leases) if lease is not None]
    refs = [[0] * 5 for _ in leases]
    if live:
        indices = torch.tensor(live, dtype=torch.int64, device=ids.device)
        published = producer.publish(
            ids[indices],
            counts[indices],
            tokens[indices],
            [versions[i] for i in live],
            [leases[i] for i in live],
        )
        for index, reference in zip(live, published, strict=True):
            refs[index] = reference
    return refs


def register_backend() -> None:
    """Register through SGLang's sampler factory API in each worker process."""
    from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
    from sglang.srt.layers.sampler import (
        Sampler,
        register_sampler_backend,
    )
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    from areal.infra.sampling_mask.lifetime import LEASE_KEY, validate_lease

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
        has_request_seed = any(seed is not None for seed in seeds)
        if has_request_seed:
            from areal.engine.sglang_sampling_mask_plugin import (
                JOINT_DSPARK,
                runtime_settings,
            )

            if runtime_settings()[3] == JOINT_DSPARK:
                raise ValueError(
                    "DSpark joint verification does not support request seeds"
                )
        if result.sampling_seed is None and has_request_seed:
            random_seeds = torch.randint(0, 2**31, (len(seeds),)).tolist()
            result.sampling_seed = torch.tensor(
                [
                    seed if seed is not None else random_seeds[i]
                    for i, seed in enumerate(seeds)
                ],
                dtype=torch.int64,
                device=batch.device,
            )
        versions, leases = [], []
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
                or params.min_new_tokens != 0
                or params.min_p != 0
            ):
                raise ValueError(
                    "sampling_mask_transfer does not yet support sampling penalties, min-p, or logit bias"
                )
            versions.append(int(custom["areal_sampling_mask_version"]))
            leases.append(
                None
                if request.rid.startswith(HEALTH_CHECK_RID_PREFIX)
                else validate_lease(custom.get(LEASE_KEY))
            )
        result._areal_mask_versions = versions
        result._areal_mask_leases = leases
        return result

    def filter_batch(self, indices, indices_device):
        versions = [self._areal_mask_versions[i] for i in indices]
        leases = [self._areal_mask_leases[i] for i in indices]
        original_filter(self, indices, indices_device)
        self._areal_mask_versions = versions
        self._areal_mask_leases = leases

    def merge_batch(self, other):
        merge_sampling_batches(self, other, original_merge)

    def copy_for_forward(self):
        result = original_copy(self)
        result._areal_mask_versions = self._areal_mask_versions.copy()
        result._areal_mask_leases = self._areal_mask_leases.copy()
        return result

    SamplingBatchInfo.copy_for_forward = copy_for_forward
    SamplingBatchInfo.from_schedule_batch = from_schedule_batch
    SamplingBatchInfo.filter_batch = filter_batch
    SamplingBatchInfo.merge_batch = merge_batch
    SamplingBatchInfo._areal_mask_installed = True

    # DSpark constructs both target and draft runners in the same worker.
    @cache
    def get_producer(device: int, max_sessions: int):
        from areal.infra.sampling_mask.service import MaskProducer

        return MaskProducer(device, max_sessions=max_sessions)

    class SamplingMaskSampler(Sampler):
        def __init__(self):
            super().__init__()
            from areal.engine.sglang_sampling_mask_plugin import (
                JOINT_DSPARK,
                runtime_settings,
            )

            backend, pp_size, attn_cp_size, speculative_algorithm = runtime_settings()
            if backend != "areal_sampling_mask":
                raise ValueError(
                    "sampling_mask_transfer sampler backend was overridden"
                )
            self.dspark = speculative_algorithm == JOINT_DSPARK
            if (
                (speculative_algorithm and not self.dspark)
                or pp_size != 1
                or attn_cp_size != 1
            ):
                raise ValueError(
                    "sampling_mask_transfer requires SGLang PP=1, attention CP=1 and a supported verify policy"
                )
            self.explicit_top_k = int(os.environ["AREAL_SAMPLING_MASK_TOP_K"])
            self.producer = (
                get_producer(
                    torch.cuda.current_device(),
                    max_sessions=int(os.environ["AREAL_SAMPLING_MASK_RECEIVERS"]),
                )
                if torch.distributed.get_rank(self.tp_sync_group) == 0
                else None
            )

        def prepare_verify_policy(self, logits, batch, stride):
            from areal.engine.sglang_dspark_policy import prepare_verify_policy

            info = batch.sampling_info
            if info.has_custom_logit_processor or info.grammars or info.is_any_greedy:
                raise ValueError(
                    "DSpark joint verification requires stochastic requests without grammar or custom processors"
                )
            if any(batch.top_logprobs_nums or ()) or any(
                batch.token_ids_logprobs or ()
            ):
                raise ValueError(
                    "DSpark joint verification only supports selected-token logprob requests"
                )
            return prepare_verify_policy(
                logits,
                info,
                stride,
                self._bound(logits),
                self.producer,
                info._areal_mask_versions,
                info._areal_mask_leases,
            )

        def _bound(self, logits):
            return (
                min(self.explicit_top_k, logits.shape[-1])
                if self.explicit_top_k > 0
                else logits.shape[-1]
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
            ids, logp, counts = candidate_distribution(
                logits,
                sampling_info.temperatures,
                sampling_info.top_ps,
                sampling_info.top_ks,
                candidate_bound=self._bound(logits),
            )
            column = sample_candidate_columns(
                logp, None if self.dspark else sampling_info.sampling_seed, positions
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
            if self.dspark:
                from areal.engine.sglang_dspark_policy import CommittedMaskRows

                if self.producer is not None:
                    logits_output.auxiliary_device_output = CommittedMaskRows(
                        self.producer,
                        ids,
                        counts,
                        sampling_info._areal_mask_versions,
                        sampling_info._areal_mask_leases,
                        1,
                    )
                return tokens
            logits_output.customized_info = {
                "sampling_mask_refs": publish_sampling_masks(
                    self.producer,
                    ids,
                    counts,
                    tokens,
                    sampling_info._areal_mask_versions,
                    sampling_info._areal_mask_leases,
                )
                if self.producer is not None
                else [[0] * 5 for _ in range(ids.shape[0])]
            }
            return tokens

    register_sampler_backend("areal_sampling_mask", SamplingMaskSampler)
