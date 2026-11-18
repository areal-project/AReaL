# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from areal.engine.awex.frozen_draft import FrozenDraftState, get_draft_model


def _models(device="cpu"):
    target = nn.Linear(4, 4, bias=False, device=device)
    draft = nn.Linear(4, 4, bias=False, device=device)
    draft.register_parameter("shared", target.weight)
    draft.register_buffer("derived", torch.arange(4, device=device), persistent=False)
    return target, draft


@pytest.mark.parametrize("cycle", [1, 3])
def test_frozen_draft_survives_repeated_release_preserving_shared_update(cycle):
    target, draft = _models()
    state = FrozenDraftState(target, draft)
    expected = draft.weight.detach().clone()
    derived = draft.derived.clone()
    identity = id(draft.weight), draft.weight.data_ptr(), draft.derived.data_ptr()
    for _ in range(cycle):
        with torch.no_grad():
            draft.weight.zero_()
            draft.derived.zero_()
            target.weight.add_(1)
        updated = target.weight.detach().clone()
        state.restore()
        torch.testing.assert_close(draft.weight, expected, rtol=0, atol=0)
        torch.testing.assert_close(draft.derived, derived, rtol=0, atol=0)
        torch.testing.assert_close(draft.shared, updated, rtol=0, atol=0)
        assert identity == (
            id(draft.weight),
            draft.weight.data_ptr(),
            draft.derived.data_ptr(),
        )
    assert state.tensor_count == 2


def test_frozen_draft_excludes_storage_alias_with_distinct_parameter():
    target, draft = _models()
    draft.shared = nn.Parameter(target.weight.detach().view_as(target.weight))
    state = FrozenDraftState(target, draft)
    with torch.no_grad():
        target.weight.add_(1)
    expected = target.weight.detach().clone()
    state.restore()
    torch.testing.assert_close(draft.shared, expected, rtol=0, atol=0)


def test_frozen_draft_rejects_replaced_tensor():
    target, draft = _models()
    state = FrozenDraftState(target, draft)
    draft.weight = nn.Parameter(draft.weight.detach().clone())
    with pytest.raises(RuntimeError, match="identity changed"):
        state.restore()


@pytest.mark.parametrize("nested", [False, True])
def test_get_draft_model_supports_eagle_and_direct_runner(nested):
    model = object()
    runner = SimpleNamespace(model=model)
    worker = (
        SimpleNamespace(_draft_worker=SimpleNamespace(draft_runner=runner))
        if nested
        else SimpleNamespace(draft_model_runner=runner)
    )
    assert get_draft_model(SimpleNamespace(draft_worker=worker)) is model
    assert get_draft_model(SimpleNamespace(draft_worker=None)) is None


def test_get_draft_model_rejects_unrecognized_worker():
    with pytest.raises(RuntimeError, match="Cannot locate"):
        get_draft_model(SimpleNamespace(draft_worker=object()))
