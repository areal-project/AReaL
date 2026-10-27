# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import pytest
import torch

from areal.engine.megatron_utils.qwen3_vl_token_mask import install_qwen3_vl_token_mask


class _Group:
    def __init__(self, size=1, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _Embedding(torch.nn.Module):
    def forward(self, input_ids, position_ids=None):
        self.ids = input_ids.clone()
        return input_ids.T[..., None].float().expand(-1, -1, 4)


class _Language(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = _Embedding()
        self.config = SimpleNamespace(hidden_size=4)
        self.rotary_pos_emb = SimpleNamespace()

    def forward(self, **kwargs):
        self.inputs = kwargs
        return kwargs["decoder_input"]


class _Vision(torch.nn.Module):
    def forward(self, hidden_states, grid_thw):
        count = int(grid_thw.prod(-1).sum()) // 4
        return torch.full((count, 4), -9.0), []


def _model(monkeypatch, cp_size=1, cp_rank=0, pre_process=True):
    native = pytest.importorskip(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model"
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda *args: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda *args: None)
    model = torch.nn.Module()
    model.forward = MethodType(native.Qwen3VLModel.forward, model)
    model.image_token_id, model.video_token_id, model.vision_start_token_id = (
        100,
        101,
        102,
    )
    model.square_merge_size = 4
    model.pre_process = pre_process
    model.use_dist_train = False
    model.config = SimpleNamespace(
        vision_dp_when_cp=False, sequence_parallel=False, spatial_merge_size=2
    )
    model.pg_collection = SimpleNamespace(cp=_Group(cp_size, cp_rank), tp=_Group())
    model.language_model, model.vision_model = _Language(), _Vision()
    return model


def _inputs():
    # Text contains image_pad/video_pad and a trailing vision_start. The real
    # image and video remain in separate payload spans, in their original order.
    return dict(
        input_ids=torch.tensor(
            [[100, 102, 100, 100, 103, 101, 102, 101, 103, 102, 8, 8]]
        ),
        mm_token_type_ids=torch.tensor([[0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0]]),
        attention_mask=torch.tensor([[True] * 10 + [False] * 2]),
        position_ids=None,
        pixel_values=torch.arange(8).float()[:, None],
        pixel_values_videos=torch.arange(4).float()[:, None],
        image_grid_thw=torch.tensor([[1, 2, 4]]),
        video_grid_thw=torch.tensor([[1, 2, 2]]),
    )


@pytest.mark.parametrize(
    "cp_size,cp_rank,packed",
    [
        (1, 0, False),
        (2, 0, False),
        (2, 1, False),
        (1, 0, True),
        (2, 0, True),
        (2, 1, True),
    ],
)
def test_native_forward_generated_specials_use_typed_scatter_and_rope(
    monkeypatch, cp_size, cp_rank, packed
):
    model = _model(monkeypatch, cp_size, cp_rank)
    install_qwen3_vl_token_mask([model])
    original_forward = model.forward
    install_qwen3_vl_token_mask([model])
    assert model.forward is original_forward
    data = _inputs()
    ids_before = data["input_ids"].clone()
    if packed:
        data["packed_seq_params"] = SimpleNamespace()
    model(**data)
    torch.testing.assert_close(
        model.language_model.embedding.ids, ids_before, rtol=0, atol=0
    )
    torch.testing.assert_close(data["input_ids"], ids_before, rtol=0, atol=0)
    actual_positions = model.language_model.inputs["position_ids"].clone()
    actual_visual_mask = model.language_model.inputs["visual_pos_masks"].clone()

    # Replace only text occurrences with ordinary text. Position/scatter results
    # must be identical, while the actual word embeddings retain the special IDs.
    clean = _inputs()
    clean["input_ids"][0, [0, 5, 9]] = 7
    if packed:
        clean["packed_seq_params"] = SimpleNamespace()
    reference = _model(monkeypatch, cp_size, cp_rank)
    install_qwen3_vl_token_mask([reference])
    reference(**clean)
    torch.testing.assert_close(
        actual_positions,
        reference.language_model.inputs["position_ids"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_visual_mask,
        reference.language_model.inputs["visual_pos_masks"],
        rtol=0,
        atol=0,
    )


def test_native_forward_mask_patch_is_scoped_to_model(monkeypatch):
    patched = _model(monkeypatch)
    untouched = _model(monkeypatch)
    install_qwen3_vl_token_mask([patched])
    patched(**_inputs())
    with pytest.raises((AssertionError, RuntimeError, IndexError)):
        untouched(**_inputs())


@pytest.mark.parametrize("missing", [True, False])
def test_native_forward_requires_aligned_token_types(monkeypatch, missing):
    model = _model(monkeypatch)
    install_qwen3_vl_token_mask([model])
    data = _inputs()
    data["mm_token_type_ids"] = None if missing else torch.zeros(1, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="token.types|token types|mm_token_type_ids"):
        model(**data)


def test_native_forward_non_embedding_pipeline_stage_uses_typed_rope(monkeypatch):
    model = _model(monkeypatch, cp_size=2, cp_rank=1, pre_process=False)
    install_qwen3_vl_token_mask([model])
    data = _inputs()
    data["packed_seq_params"] = SimpleNamespace()
    model(**data)
    assert model.language_model.inputs["position_ids"].shape == (3, 1, 6)
    assert model.language_model.inputs["input_ids"].shape == (1, 6)
