# SPDX-License-Identifier: Apache-2.0
"""Numerical and wire-format contracts for frozen sampling supports."""

import pytest
import torch

from areal.utils.sampling_mask import (
    SamplingMask,
    estimate_mask_memory,
    pack_uint24,
    restricted_logprobs,
    unpack_uint24,
)


def fixture_data():
    torch.manual_seed(19)
    logits = torch.randn(4, 11, requires_grad=True)
    ids = torch.tensor([1, 4, 9, 0, 2, 3, 10, 7, 5, 6], dtype=torch.int32)
    offsets = torch.tensor([0, 3, 7, 8, 10], dtype=torch.int64)
    labels = torch.tensor([4, 10, 7, 5])
    mask = torch.zeros_like(logits, dtype=torch.bool)
    rows = torch.repeat_interleave(torch.arange(4), offsets.diff())
    mask[rows, ids.long()] = True
    return logits, ids, offsets, labels, mask


def calculate(logits, ids, offsets, labels, temperature=0.7):
    return restricted_logprobs(
        logits, labels, ids, offsets, temperature=temperature, vocab_size=11
    )


def clipped_loss(logp, old, advantages):
    ratio = (logp - old.detach()).exp()
    return -torch.minimum(
        ratio * advantages.detach(), ratio.clamp(0.8, 1.2) * advantages.detach()
    ).mean()


@pytest.mark.parametrize("temperature", [0.3, 1.0, 1.7])
def test_restricted_probability_and_gradient_match_dense(temperature):
    """The denominator contributes exactly the dense masked-softmax gradient."""
    logits, ids, offsets, labels, mask = fixture_data()
    reference = logits.detach().clone().requires_grad_()
    logp, entropy = calculate(logits, ids, offsets, labels, temperature)
    dense = (reference / temperature).masked_fill(~mask, -torch.inf).log_softmax(-1)
    expected = dense.gather(1, labels[:, None]).squeeze(1)
    expected_entropy = -(dense.exp() * dense.masked_fill(~mask, 0)).sum(-1)
    torch.testing.assert_close(logp, expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(entropy, expected_entropy, atol=1e-5, rtol=0)
    torch.testing.assert_close(dense.exp().sum(-1), torch.ones(4), atol=1e-6, rtol=0)
    advantages = torch.tensor([0.4, -0.7, 0.2, -0.1])
    old = expected.detach() - torch.tensor([0.04, -0.05, 0.0, 0.03])
    clipped_loss(logp, old, advantages).backward()
    clipped_loss(expected, old, advantages).backward()
    torch.testing.assert_close(logits.grad, reference.grad, atol=1e-4, rtol=0)
    assert logits.grad[0, 1] != 0  # Non-action candidate: denominator gradient.
    torch.testing.assert_close(
        logits.grad[~mask], torch.zeros_like(logits.grad[~mask]), atol=0, rtol=0
    )


def test_support_shift_preserves_probability_and_initial_ratio():
    """Uniformly shifting all retained effective logits preserves ratio=1."""
    logits, ids, offsets, labels, mask = fixture_data()
    old, _ = calculate(logits, ids, offsets, labels)
    shifted = logits.detach() + mask * torch.tensor([5.0, -4.0, 11.0, 0.5])[:, None]
    new, _ = calculate(shifted, ids, offsets, labels)
    torch.testing.assert_close(new, old, atol=1e-5, rtol=0)
    torch.testing.assert_close((new - old).exp(), torch.ones(4), atol=1e-5, rtol=0)


def test_excluded_logits_do_not_change_actor_loss_or_gradient():
    """A fixed support isolates the pure PPO actor objective from excluded logits."""
    logits, ids, offsets, labels, mask = fixture_data()
    old, _ = calculate(logits.detach(), ids, offsets, labels)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    first, _ = calculate(logits, ids, offsets, labels)
    changed = (logits.detach() + (~mask) * 1000).requires_grad_()
    second, _ = calculate(changed, ids, offsets, labels)
    loss = clipped_loss(first, old, advantages)
    other_loss = clipped_loss(second, old, advantages)
    torch.testing.assert_close(loss, other_loss, atol=0, rtol=0)
    loss.backward()
    other_loss.backward()
    torch.testing.assert_close(logits.grad, changed.grad, atol=0, rtol=0)
    torch.testing.assert_close(
        changed.grad[~mask], torch.zeros_like(changed.grad[~mask]), atol=0, rtol=0
    )
    assert not torch.equal(changed.topk(3).indices, logits.topk(3).indices)


def test_full_support_matches_original_full_vocabulary():
    """No truncation reduces to the ordinary full-vocabulary PPO objective."""
    torch.manual_seed(13)
    logits = torch.randn(3, 11, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    labels = torch.tensor([0, 3, 10])
    ids = torch.arange(11, dtype=torch.int32).repeat(3)
    offsets = torch.arange(4, dtype=torch.int64) * 11
    actual, _ = calculate(logits, ids, offsets, labels, 1.0)
    expected = reference.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
    advantages = torch.tensor([-1.0, 0.7, 0.2])
    old = expected.detach()
    loss, expected_loss = (
        clipped_loss(actual, old, advantages),
        clipped_loss(expected, old, advantages),
    )
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(loss, expected_loss, atol=1e-5, rtol=0)
    torch.testing.assert_close(logits.grad, reference.grad, atol=1e-4, rtol=0)


def test_uint24_boundary_ids_round_trip():
    """Little-endian packing keeps all 24 bits and consumes exactly three bytes."""
    ids = torch.tensor(
        [0, 1, 255, 256, 65535, 65536, 248319, 2**24 - 1], dtype=torch.int32
    )
    packed = pack_uint24(ids)
    assert packed.numel() == 3 * ids.numel()
    assert packed[3].tolist() == [0, 1, 0]
    torch.testing.assert_close(unpack_uint24(packed), ids, atol=0, rtol=0)


def test_http_metadata_counters_keep_byte_totals_and_integer_json_encoding(monkeypatch):
    import orjson

    from areal.infra.sampling_mask import metrics
    from areal.utils.stats_tracker import DistributedStatsTracker

    tracker = DistributedStatsTracker("sampling_mask/http")
    monkeypatch.setattr(metrics.stats_tracker, "get", lambda name: tracker)
    refs = [
        [12345678901234, 987654321, 42, 1, 7],
        [12345678901234, 987654321, 43, 1, 7],
    ]
    metrics.record_response_body(2**30 + 1)
    metrics.record_response_body(17)
    metrics.record_reference_json(refs)
    exported = tracker.export()
    assert exported["sampling_mask/http/generate_response_body_bytes"] == 2**30 + 18
    assert exported["sampling_mask/http/generate_responses"] == 2
    assert exported["sampling_mask/http/reference_json_value_bytes"] == len(
        orjson.dumps(refs)
    )


def test_csr_reordering_preserves_tokens_and_payload_counts():
    """Variable-length rows survive permutation and repeated selection."""
    _, ids, offsets, labels, _ = fixture_data()
    support = SamplingMask(pack_uint24(ids), offsets, 11, 7)
    support.validate(labels, weight_version=7)
    rows = torch.tensor([3, 0, 2, 0])
    reordered = support.select_rows(rows)
    reordered.validate(labels[rows], weight_version=7)
    assert reordered.offsets.tolist() == [0, 2, 5, 6, 9]
    assert support.statistics()["id_bytes"] == 30
    assert support.statistics()["offset_bytes"] == 40


@pytest.mark.parametrize(
    "case", ["empty", "duplicate", "outside", "missing", "version", "protocol"]
)
def test_invalid_support_fails_explicitly(case):
    """Malformed and stale supports must never fall back to a full softmax."""
    ids, offsets, labels = (
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([0, 2]),
        torch.tensor([1]),
    )
    version, protocol = 3, 1
    if case == "empty":
        offsets, labels = torch.tensor([0, 0, 2]), torch.tensor([1, 1])
    elif case == "duplicate":
        ids = torch.tensor([1, 1], dtype=torch.int32)
    elif case == "outside":
        ids = torch.tensor([1, 11], dtype=torch.int32)
    elif case == "missing":
        labels = torch.tensor([4])
    elif case == "version":
        version = 4
    elif case == "protocol":
        protocol = 2
    support = SamplingMask(pack_uint24(ids), offsets, 11, version, protocol)
    with pytest.raises(ValueError):
        support.validate(labels, weight_version=3)


def test_budget_uses_explicit_top_k_and_rejects_full_vocab_overflow():
    """The preflight never introduces a hidden candidate cap."""
    arguments = dict(
        max_tokens=512 * 8192,
        vocab_size=248320,
        data_parallel_size=4,
        chunk_tokens=1024,
        buffer_bytes=8 * 1024**2,
    )
    result = estimate_mask_memory(top_k=128, **arguments)
    assert (
        result["host_bytes"] > (512 * 8192 // 4) * 128 * 3 + (512 * 8192 // 4 + 1) * 8
    )
    with pytest.raises(ValueError, match="preflight exceeds"):
        estimate_mask_memory(top_k=-1, **arguments)


@pytest.mark.parametrize(
    "top_k,top_p,temperature",
    [(11, 1.0, 1.0), (4, 0.8, 0.7), (1, 0.95, 1.5), (11, 0.6, 0.8)],
)
def test_sampling_support_and_old_logprob_match_joint_reference(
    top_k, top_p, temperature
):
    """Temperature/top-p/top-k select the exact joint support used to sample."""
    from areal.engine.sglang_sampling_mask import candidate_distribution

    logits, _, _, _, _ = fixture_data()
    ids, logp, counts = candidate_distribution(
        logits.detach(),
        torch.full((4, 1), temperature),
        torch.full((4,), top_p),
        torch.full((4,), top_k),
        candidate_bound=top_k,
    )
    ordered, order = logits.detach().div(temperature).sort(-1, descending=True)
    probabilities = ordered.softmax(-1)
    keep = (probabilities.cumsum(-1) - probabilities <= top_p) & (
        torch.arange(11)[None, :] < top_k
    )
    expected = ordered.masked_fill(~keep, -torch.inf).log_softmax(-1)
    torch.testing.assert_close(ids.long(), order[:, :top_k], atol=0, rtol=0)
    torch.testing.assert_close(logp, expected[:, :top_k], atol=1e-5, rtol=0)
    torch.testing.assert_close(counts, keep.sum(-1), atol=0, rtol=0)
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones(4), atol=1e-5, rtol=0)


def test_nonstreaming_internal_chunks_accumulate_through_eos():
    """SGLang force-stream fragments must not replace earlier references."""
    from types import SimpleNamespace

    from areal.engine.sglang_sampling_mask import accumulate_reference_metadata

    state = SimpleNamespace()
    metadata = {}
    for values in [[1, 2], [3], [4, 5]]:
        accumulate_reference_metadata(state, metadata, "sampling_mask_refs", values)
    assert metadata["sampling_mask_refs"] == [1, 2, 3, 4, 5]


def test_sparse_inplace_backward_matches_autograd():
    """Chunked LM-head's sparse VJP includes normalization and zero excluded rows."""
    from areal.utils.sampling_mask import (
        restricted_logits_backward_,
        restricted_policy_terms,
    )

    logits, ids, offsets, labels, _ = fixture_data()
    logp, _, maximum, denominator = restricted_policy_terms(
        logits, labels, ids, offsets, temperature=0.7, vocab_size=11
    )
    upstream = torch.tensor([0.3, -0.2, 0.8, -0.9])
    logp.backward(upstream)
    actual = logits.detach().clone()
    restricted_logits_backward_(
        actual,
        labels,
        ids,
        offsets,
        maximum.detach(),
        denominator.detach(),
        upstream,
        temperature=0.7,
        tp_rank=0,
    )
    torch.testing.assert_close(actual, logits.grad, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "advantage,ratio",
    [(1.0, 0.7), (1.0, 1.3), (-1.0, 0.7), (-1.0, 1.3), (1.0, 0.8), (-1.0, 1.2)],
)
def test_existing_ppo_clip_uses_restricted_ratio(advantage, ratio):
    """Exercise the actual PPO function on both signs and clipping boundaries."""
    from areal.utils.functional import ppo_actor_loss_fn

    current = torch.tensor([[ratio]], dtype=torch.float32).log().requires_grad_()
    old = torch.zeros_like(current)
    advantages = torch.full_like(current, advantage)
    loss, _ = ppo_actor_loss_fn(
        current, old, old, advantages, 0.2, torch.ones_like(current, dtype=torch.bool)
    )
    expected = -min(ratio * advantage, min(max(ratio, 0.8), 1.2) * advantage)
    torch.testing.assert_close(loss, torch.tensor(expected), atol=1e-6, rtol=0)


def test_mask_references_survive_sequence_reorder_pack_padding():
    """Metadata follows the same data operators as tokens and behavior logprobs."""
    from areal.api.cli_args import MicroBatchSpec
    from areal.api.io_struct import ModelResponse
    from areal.infra.sampling_mask.protocol import trajectory_references
    from areal.utils.data import (
        concat_padded_tensors,
        pack_tensor_dict,
        split_padded_tensor_dict_into_mb_list,
    )

    examples = []
    for index, length in enumerate([3, 7, 2]):
        response = ModelResponse(
            input_tokens=[8, 9],
            output_tokens=list(range(length)),
            output_versions=[3] * length,
            sampling_mask_refs=[
                [123, 456, index * 10 + i + 1, 1, 3] for i in range(length)
            ],
        )
        references = trajectory_references(response)
        examples.append(
            {
                "input_ids": torch.tensor(
                    [response.input_tokens + response.output_tokens]
                ),
                "attention_mask": torch.ones((1, length + 2), dtype=torch.bool),
                **{key: value.unsqueeze(0) for key, value in references.items()},
            }
        )
    batch = concat_padded_tensors(examples)
    packed = pack_tensor_dict(batch)
    assert packed["sampling_mask_row"].tolist() == [
        0,
        0,
        1,
        2,
        3,
        0,
        0,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        0,
        0,
        21,
        22,
    ]
    microbatches = split_padded_tensor_dict_into_mb_list(
        batch, MicroBatchSpec(max_tokens_per_mb=12), sync_mbs=False
    )
    for original in microbatches.mbs:
        rows = original["sampling_mask_row"]
        valid = rows > 0
        torch.testing.assert_close(
            original["input_ids"][valid], (rows[valid] - 1) % 10, atol=0, rtol=0
        )


def test_chunk_store_reorder_release_and_capacity():
    """Numeric indexing avoids per-token objects and frees complete CSR blocks."""
    import numpy as np

    from areal.infra.sampling_mask.store import RowStore

    _, ids, offsets, labels, _ = fixture_data()
    store = RowStore(100000)
    rows = np.array([1, 3, 7, 9])
    store.add(rows, pack_uint24(ids), offsets, labels, torch.full_like(labels, 4))
    packed, reordered_offsets, tokens, versions = store.read(np.array([9, 1, 7, 1]))
    SamplingMask(packed, reordered_offsets, 11, 4).validate(tokens, weight_version=4)
    assert tokens.tolist() == [5, 4, 7, 4]
    assert versions.tolist() == [4] * 4
    store.delete([1, 3])
    assert store.locate(np.array([1, 7])).tolist()[0] == 0
    store.discard_before(5)
    assert store.nbytes == 0
    with pytest.raises(MemoryError):
        RowStore(100).add(
            rows, pack_uint24(ids), offsets, labels, torch.full_like(labels, 4)
        )


def test_batch_identity_survives_epoch_shuffle_and_padding():
    from areal.infra.sampling_mask.protocol import REFERENCE_KEYS, batch_identity

    rows = torch.tensor([[7, 8, 3], [7, 8, 1], [7, 8, 2], [0, 0, 0]])
    original = {key: rows[:, i] for i, key in enumerate(REFERENCE_KEYS)}
    reordered = torch.cat([rows[[2, 0, 1, 2]], torch.zeros(5, 3, dtype=torch.int64)])
    other = {key: reordered[:, i] for i, key in enumerate(REFERENCE_KEYS)}
    assert batch_identity(original) == batch_identity(other)
    other[REFERENCE_KEYS[-1]][0] = 9
    assert batch_identity(original) != batch_identity(other)


@pytest.mark.parametrize(
    "kl_ctl,kl_estimator", [(0.0, "k3"), (0.1, "k1"), (0.1, "k2"), (0.1, "k3")]
)
def test_feature_configuration_anchors_ppo_to_rollout_probability(
    monkeypatch, kl_ctl, kl_estimator
):
    from types import SimpleNamespace

    from areal.api.cli_args import GenerationHyperparameters
    from areal.infra.sampling_mask import config as mask_config
    from areal.infra.sampling_mask.config import configure

    original_find = mask_config.importlib.util.find_spec
    monkeypatch.setattr(
        mask_config.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            submodule_search_locations=["/runtime/nvidia/nccl"]
        )
        if name == "nvidia.nccl"
        else original_find(name),
    )
    spec = SimpleNamespace(env_vars={"LD_LIBRARY_PATH": "/runtime/other"})

    actor = SimpleNamespace(
        _version="v1",
        backend="megatron:d1t2p1",
        use_score_centering=False,
        use_sapo_loss=False,
        use_cispo_loss=False,
        prox_logp_method="reuse_train_logp",
        recompute_logprob=True,
        use_decoupled_loss=True,
        kl_ctl=kl_ctl,
        kl_estimator=kl_estimator,
        scheduling_spec=[spec],
    )
    config = SimpleNamespace(
        sampling_mask_transfer=True,
        actor=actor,
        rollout=SimpleNamespace(
            _version="v1", backend="sglang:d1t2p1", max_head_offpolicyness=1
        ),
        sglang=SimpleNamespace(speculative_algorithm=None),
        teacher=None,
        mopd=None,
        eval_gconfig=None,
        gconfig=GenerationHyperparameters(top_k=128, top_p=0.95, temperature=0.8),
    )
    if kl_ctl != 0 and kl_estimator != "k1":
        with pytest.raises(ValueError, match="kl_estimator='k1'"):
            configure(config)
        return
    configure(config)
    configure(config)
    assert actor.prox_logp_method == "recompute"
    assert not actor.recompute_logprob
    assert not actor.use_decoupled_loss
    assert actor.temperature == 0.8
    assert config.gconfig.top_k == 128
    assert spec.env_vars["NCCL_CUMEM_ENABLE"] == "1"
    assert spec.env_vars["LD_LIBRARY_PATH"] == "/runtime/nvidia/nccl/lib:/runtime/other"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the Megatron chunked LM-head",
)
@pytest.mark.parametrize("chunk_size", [1, 5, 32])
def test_megatron_chunked_mask_gradient_matches_dense(chunk_size):
    """Real chunked Megatron autograd reuses frozen supports during recomputation."""
    from areal.infra.sampling_mask.consumer import LoadedSamplingMask
    from areal.models.mcore.vocab_parallel_head import _ChunkedVocabParallelLMHead

    torch.manual_seed(77)
    hidden = torch.randn(4, 1, 16, device="cuda", requires_grad=True)
    weight = torch.randn(11, 16, device="cuda", requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    _, ids, offsets, labels, mask = fixture_data()
    support = LoadedSamplingMask(pack_uint24(ids), offsets, 11)
    actual = _ChunkedVocabParallelLMHead.apply(
        hidden,
        weight,
        None,
        labels.cuda(),
        0.7,
        chunk_size,
        False,
        False,
        False,
        None,
        0.5,
        True,
        support,
    )[0]
    logits = (reference_hidden.reshape(4, 16) @ reference_weight.t()) * 0.5 / 0.7
    dense = logits.masked_fill(~mask.cuda(), -torch.inf).log_softmax(-1)
    expected = dense.gather(1, labels.cuda()[:, None]).flatten()
    upstream = torch.tensor([0.1, -0.3, 0.2, -0.6], device="cuda")
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=1e-4, rtol=0)
    torch.testing.assert_close(weight.grad, reference_weight.grad, atol=1e-4, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_full_support_matches_disabled_megatron_head_ppo_loss_and_gradients():
    """Exercise both real LM-head branches, including the original fused kernels."""
    from areal.infra.sampling_mask.consumer import LoadedSamplingMask
    from areal.models.mcore.vocab_parallel_head import _ChunkedVocabParallelLMHead

    torch.manual_seed(41)
    hidden = torch.randn(4, 1, 16, device="cuda", requires_grad=True)
    weight = torch.randn(32, 16, device="cuda", requires_grad=True)
    original_hidden = hidden.detach().clone().requires_grad_()
    original_weight = weight.detach().clone().requires_grad_()
    labels = torch.tensor([0, 9, 18, 31], device="cuda")
    support = LoadedSamplingMask(
        pack_uint24(torch.arange(32, dtype=torch.int32).repeat(4)),
        torch.arange(5, dtype=torch.int64) * 32,
        32,
    )

    def forward(h, w, mask):
        return _ChunkedVocabParallelLMHead.apply(
            h,
            w,
            None,
            labels,
            0.8,
            3,
            False,
            False,
            False,
            None,
            0.5,
            True,
            mask,
        )[0]

    actual = forward(hidden, weight, support)
    original = forward(original_hidden, original_weight, None)
    old = original.detach() - torch.tensor([0.03, -0.04, 0.01, -0.02], device="cuda")
    advantages = torch.tensor([0.3, -0.7, 0.5, -0.2], device="cuda")
    loss, original_loss = (
        clipped_loss(actual, old, advantages),
        clipped_loss(original, old, advantages),
    )
    loss.backward()
    original_loss.backward()
    torch.testing.assert_close(actual, original, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        (actual - old).exp(), (original - old).exp(), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(loss, original_loss, atol=1e-5, rtol=0)
    torch.testing.assert_close(hidden.grad, original_hidden.grad, atol=1e-4, rtol=0)
    torch.testing.assert_close(weight.grad, original_weight.grad, atol=1e-4, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_sampling_reproduces_draws_within_saved_support():
    """The actual seeded SGLang kernel keeps candidate-column shape and seeds."""
    from areal.engine.sglang_sampling_mask import (
        candidate_distribution,
        sample_candidate_columns,
    )

    device = torch.device("cuda")
    logits = torch.zeros(128, 17, device=device)
    ids, logp, _ = candidate_distribution(
        logits,
        torch.ones(128, 1, device=device),
        torch.full((128,), 0.9, device=device),
        torch.full((128,), 8, dtype=torch.int32, device=device),
        candidate_bound=8,
    )
    seeds = torch.full((128,), 19, dtype=torch.int64, device=device)
    positions = torch.arange(128, dtype=torch.int64, device=device)
    first = sample_candidate_columns(logp, seeds, positions)
    repeated = sample_candidate_columns(logp, seeds, positions)
    different = sample_candidate_columns(logp, seeds + 1, positions)
    assert first.shape == (128, 1)
    torch.testing.assert_close(first, repeated, atol=0, rtol=0)
    assert torch.any(first != different)
    tokens = ids.gather(1, first)
    assert torch.all((ids == tokens).any(-1))


def test_sampler_without_truncation_retains_full_vocabulary_despite_roundoff():
    from areal.engine.sglang_sampling_mask import candidate_distribution

    torch.manual_seed(0)
    logits = (torch.randn(8, 4096) * 4).requires_grad_()
    ids, logp, counts = candidate_distribution(
        logits,
        torch.ones(8, 1),
        torch.ones(8),
        torch.full((8,), 4096),
        candidate_bound=4096,
    )
    torch.testing.assert_close(counts, torch.full((8,), 4096), atol=0, rtol=0)
    expected = logits.log_softmax(-1).gather(1, ids.long())
    torch.testing.assert_close(logp, expected, atol=1e-5, rtol=0)
    weights = torch.randn_like(logp)
    actual_grad = torch.autograd.grad(
        (logp * weights).sum(), logits, retain_graph=True
    )[0]
    reference_grad = torch.autograd.grad((expected * weights).sum(), logits)[0]
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-4, rtol=0)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_sampler_rejects_nonfinite_normalizer_without_truncation(invalid):
    from areal.engine.sglang_sampling_mask import candidate_distribution

    logits = torch.full((1, 8), invalid)
    with pytest.raises(RuntimeError, match="non-finite normalizer"):
        candidate_distribution(
            logits,
            torch.ones(1, 1),
            torch.ones(1),
            torch.full((1,), 8),
            candidate_bound=8,
        )


@pytest.mark.parametrize("seeded_first", [True, False])
def test_scheduler_merge_preserves_explicit_seeds_and_row_alignment(seeded_first):
    from types import SimpleNamespace

    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    from areal.engine.sglang_sampling_mask import merge_sampling_batches

    def batch(size, seeded, version):
        value = SamplingBatchInfo(
            temperatures=torch.ones(size, 1),
            top_ps=torch.ones(size),
            top_ks=torch.full((size,), 8),
            min_ps=torch.zeros(size),
            is_all_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=True,
            need_min_p_sampling=False,
            vocab_size=17,
            device="cpu",
            sampling_seed=torch.arange(size) + 123 if seeded else None,
            penalizer_orchestrator=SimpleNamespace(merge=lambda other: None),
        )
        value._areal_mask_versions = [version] * size
        return value

    left, right = batch(2, seeded_first, 5), batch(3, not seeded_first, 6)
    original = (left if seeded_first else right).sampling_seed.clone()
    merge_sampling_batches(left, right, SamplingBatchInfo.merge_batch)
    assert left.sampling_seed.shape == (5,)
    torch.testing.assert_close(
        left.sampling_seed[:2] if seeded_first else left.sampling_seed[2:],
        original,
        atol=0,
        rtol=0,
    )
    assert left._areal_mask_versions == [5, 5, 6, 6, 6]


@pytest.mark.parametrize(
    "enabled,expected",
    [(True, "1"), (False, "0")],
)
def test_weight_update_environment_preserves_sampling_mask_cumem(
    monkeypatch, enabled, expected
):
    """Late weight-update setup must not switch one GIN peer to legacy IPC."""
    from types import SimpleNamespace

    from areal.trainer import rl_trainer

    spec = SimpleNamespace(env_vars={})
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            sampling_mask_transfer=enabled,
            actor=SimpleNamespace(scheduling_spec=[spec]),
        ),
        rollout_alloc=SimpleNamespace(backend="sglang"),
    )
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
    rl_trainer.PPOTrainer._amend_xccl_weight_update_envvar(trainer)
    assert spec.env_vars["NCCL_CUMEM_ENABLE"] == expected


@pytest.mark.parametrize("logical_owner", [False, True])
def test_metrics_count_tp_logical_rows_separately_from_transport_owners(logical_owner):
    """TP1 may own GIN while only TP0 contributes replicated CSR statistics."""
    from types import SimpleNamespace

    import numpy as np

    from areal.infra.sampling_mask.consumer import aggregate_metrics

    consumer = SimpleNamespace(
        _candidate_histogram=np.array([0, 1, 2, 0], dtype=np.int64),
        id_bytes=15,
        offset_bytes=32,
        control_bytes=91,
        _sessions={1: SimpleNamespace(payload_bytes=15)},
        _cache_bytes=256,
    )
    metrics = aggregate_metrics(
        consumer, vocab_size=3, logical_owner=logical_owner, cpu_group=None
    )
    assert metrics["sampling_mask/lifetime/received_rows"] == (
        3 if logical_owner else 0
    )
    assert metrics["sampling_mask/lifetime/id_payload_bytes"] == (
        15 if logical_owner else 0
    )
    assert metrics["sampling_mask/lifetime/gin_payload_bytes"] == 15
    assert metrics["sampling_mask/lifetime/control_body_bytes"] == 91
    assert metrics["sampling_mask/cpu_cache_bytes_max_rank"] == 256
    if logical_owner:
        assert metrics["sampling_mask/lifetime/candidate_p95"] == 2
        assert metrics["sampling_mask/lifetime/candidate_mean"] == pytest.approx(5 / 3)


def test_decode_block_compaction_preserves_reordered_rows_and_version_expiry():
    """Coalescing decode allocations cannot change IDs, labels or row handles."""
    import numpy as np

    from areal.infra.sampling_mask.store import RowStore

    store = RowStore(1 << 20)
    for index in range(256):
        store.add(
            np.array([2 * index + 1, 2 * index + 2]),
            pack_uint24(torch.tensor([1, 2, 3, 4], dtype=torch.int32)),
            torch.tensor([0, 1, 4]),
            torch.tensor([1, 3]),
            torch.tensor([1, 2]),
        )
    rows = np.arange(512, 0, -1)
    before = store.read(rows)
    previous_bytes = store.nbytes
    store.compact()
    assert len(store._blocks) == 1
    assert store.nbytes <= previous_bytes
    for actual, expected in zip(store.read(rows), before, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    store.discard_before(2)
    assert not store.locate(np.arange(1, 513, 2)).any()
    packed, offsets, tokens, versions = store.read(np.arange(512, 0, -2))
    SamplingMask(packed, offsets, 5, 2).validate(tokens, weight_version=2)
    assert torch.all(versions == 2)
