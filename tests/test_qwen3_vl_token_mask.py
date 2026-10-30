# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams

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
    # Exercise native forward with CPU-only CP indices for these mask tests.
    monkeypatch.setattr(
        native, "get_packed_seq_cp_partition_indices", _packed_cp_indices, raising=False
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


def _packed_cp_indices(
    packed_seq_params: PackedSeqParams,
    *,
    total_tokens: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
    cp_group: _Group,
) -> torch.Tensor:
    """Select zigzag CP chunks without a distributed group or CUDA kernel."""
    boundaries = packed_seq_params.cu_seqlens_q_padded.tolist()
    assert boundaries[-1] == total_tokens
    assert cp_group.size() == cp_size and cp_group.rank() == cp_rank
    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        assert (end - start) % (2 * cp_size) == 0
        chunk_size = (end - start) // (2 * cp_size)
        for chunk in (cp_rank, 2 * cp_size - 1 - cp_rank):
            chunks.append(
                torch.arange(
                    start + chunk * chunk_size,
                    start + (chunk + 1) * chunk_size,
                    device=device,
                )
            )
    return torch.cat(chunks)


def _packed_seq_params() -> PackedSeqParams:
    """Match the THD metadata for ten valid tokens and two padding tokens."""
    cu_seqlens = torch.tensor([0, 10], dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([0, 12], dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=12,
        max_seqlen_kv=12,
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
        data["packed_seq_params"] = _packed_seq_params()
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
        clean["packed_seq_params"] = _packed_seq_params()
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
    data["packed_seq_params"] = _packed_seq_params()
    model(**data)
    assert model.language_model.inputs["position_ids"].shape == (3, 1, 6)
    assert model.language_model.inputs["input_ids"].shape == (1, 6)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("empty_grid", [False, True])
def test_native_forward_without_visual_grid_has_no_deepstack_mask(
    monkeypatch, packed, empty_grid
):
    """Text-only CP microbatches must not pass a mask without visual features."""
    model = _model(monkeypatch, cp_size=2)
    install_qwen3_vl_token_mask([model])
    data = _inputs()
    data["mm_token_type_ids"].zero_()
    data["pixel_values"] = torch.empty(0, 1) if empty_grid else None
    data["image_grid_thw"] = torch.empty(0, 3, dtype=torch.long) if empty_grid else None
    data["pixel_values_videos"] = None
    data["video_grid_thw"] = None
    if packed:
        data["packed_seq_params"] = _packed_seq_params()

    model(**data)

    assert model.language_model.inputs["visual_pos_masks"] is None
    assert model.language_model.inputs["deepstack_visual_embeds"] is None


@pytest.mark.parametrize("modality", ["image", "video"])
def test_native_forward_single_visual_modality_preserves_mask(monkeypatch, modality):
    """A real image or video still supplies its CP-local visual mask."""
    model = _model(monkeypatch, cp_size=2)
    install_qwen3_vl_token_mask([model])
    data = _inputs()
    data["packed_seq_params"] = _packed_seq_params()
    if modality == "image":
        data["pixel_values_videos"] = None
        data["video_grid_thw"] = None
        data["mm_token_type_ids"][0, 7] = 0
    else:
        data["pixel_values"] = None
        data["image_grid_thw"] = None
        data["mm_token_type_ids"][0, 2:4] = 0

    model(**data)

    assert model.language_model.inputs["visual_pos_masks"] is not None
    assert model.language_model.inputs["deepstack_visual_embeds"] is not None


def reorganize_inputs(**kwargs):
    """Model the native no-pixels return, which retains a supplied mask."""
    if kwargs["pixel_values"] is None:
        return None, kwargs["video_grid_thw"], kwargs["video_input_mask"]
    return kwargs["pixel_values"], kwargs["image_grid_thw"], kwargs["image_input_mask"]


def get_rope_index(input_ids):
    return input_ids


def preprocess_packed_seqs(ids, *args, **kwargs):
    return ids, None


def _native_visual_contract_forward(
    self,
    input_ids,
    mm_token_type_ids,
    attention_mask,
    packed_seq_params,
    pixel_values,
    pixel_values_videos,
    image_grid_thw,
    video_grid_thw,
):
    inputs = dict(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        image_input_mask=None,
        video_input_mask=None,
    )
    _, grid, mask = reorganize_inputs(**inputs)
    deepstack = [] if grid is not None and grid.shape[0] > 0 else None
    _, _, packed_mask = reorganize_inputs(**{**inputs, "input_ids": input_ids.clone()})
    if packed_mask is not None and deepstack is None:
        raise TypeError("visual mask without deepstack features")
    return mask, packed_mask


@pytest.mark.parametrize("grid_size", [None, 0, 1])
def test_token_mask_wrapper_matches_native_visual_feature_gate(grid_size):
    """Packed text and empty grids must not invent a deepstack mask."""
    model = torch.nn.Module()
    model.forward = MethodType(_native_visual_contract_forward, model)
    model.image_token_id = 100
    model.video_token_id = 101
    model.vision_start_token_id = 102
    model.pg_collection = SimpleNamespace()
    install_qwen3_vl_token_mask([model])
    ids = torch.tensor([[100, 8, 8, 8]])
    types = torch.tensor([[int(grid_size == 1), 0, 0, 0]])
    grid = None if grid_size is None else torch.zeros(grid_size, 3, dtype=torch.long)
    pixels = None if grid_size is None else torch.zeros(grid_size, 1)

    mask, packed_mask = model(
        input_ids=ids,
        mm_token_type_ids=types,
        attention_mask=torch.ones_like(ids, dtype=torch.bool),
        packed_seq_params=SimpleNamespace(),
        pixel_values=pixels,
        pixel_values_videos=None,
        image_grid_thw=grid,
        video_grid_thw=None,
    )

    if grid_size == 1:
        assert mask is not None
        assert packed_mask is not None
    else:
        assert mask is None
        assert packed_mask is None
