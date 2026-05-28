from types import MethodType, SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from areal.api.cli_args import MicroBatchSpec
from areal.engine.core.train_engine import compute_total_loss_weight
from areal.infra.controller.train_controller import (
    _dispatch_tensors,
    _pad_eval_batch,
)
from areal.trainer.rw.rw_engine import (
    RWController,
    RWEngine,
    _rw_loss_weight,
    compute_rw_loss,
)
from areal.trainer.sft.lm_engine import LMController
from areal.utils.data import (
    MicroBatchList,
    concat_padded_tensors,
    make_dummy_eval_item,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.stats_tracker import DistributedStatsTracker


def _make_item(idx: int, seqlen: int = 2) -> dict[str, object]:
    return {
        "input_ids": torch.full((1, seqlen), idx + 1, dtype=torch.long),
        "attention_mask": torch.ones((1, seqlen), dtype=torch.bool),
        "loss_mask": torch.ones((1, seqlen), dtype=torch.bool),
        "meta": {"id": idx},
    }


def _flatten_splits(splits: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    return [item for group in splits for item in group]


def _count_dummies(items: list[dict[str, object]]) -> int:
    return sum(
        int(cast(torch.Tensor, item["attention_mask"]).sum().item() == 0)
        for item in items
    )


def _make_rw_pair(
    pair_idx: int, chosen_len: int = 3, rejected_len: int = 2
) -> tuple[dict[str, object], dict[str, object]]:
    chosen: dict[str, object] = {
        "input_ids": torch.full((1, chosen_len), pair_idx * 2 + 1, dtype=torch.long),
        "attention_mask": torch.ones((1, chosen_len), dtype=torch.bool),
        "meta": {"pair": pair_idx, "role": "chosen"},
    }
    rejected: dict[str, object] = {
        "input_ids": torch.full((1, rejected_len), pair_idx * 2 + 2, dtype=torch.long),
        "attention_mask": torch.ones((1, rejected_len), dtype=torch.bool),
        "meta": {"pair": pair_idx, "role": "rejected"},
    }
    return chosen, rejected


def _build_rw_batch(n_pairs: int, chosen_len: int = 3, rejected_len: int = 2):
    items: list[dict[str, object]] = []
    for p in range(n_pairs):
        c, r = _make_rw_pair(p, chosen_len, rejected_len)
        items.extend([c, r])
    return items


def _make_rw_input(seqlens: list[int]) -> dict[str, torch.Tensor]:
    cu_seqlens = [0]
    for seqlen in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
    return {"cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32)}


class TestEvalBatchPadding:
    def test_pad_eval_batch_no_padding_when_divisible(self):
        items = [_make_item(i) for i in range(8)]
        (padded,) = _pad_eval_batch((items,), dp_size=4)
        assert len(padded) == 8
        assert _count_dummies(padded) == 0

    def test_pad_eval_batch_pads_when_not_divisible(self):
        items = [_make_item(i) for i in range(7)]
        (padded,) = _pad_eval_batch((items,), dp_size=4)
        assert len(padded) == 8
        assert _count_dummies(padded) == 1

    def test_pad_eval_batch_pads_when_n_less_than_dp(self):
        items = [_make_item(i) for i in range(2)]
        (padded,) = _pad_eval_batch((items,), dp_size=4)
        assert len(padded) == 4
        assert _count_dummies(padded) == 2

    def test_dispatch_tensors_raises_when_not_divisible(self):
        items = [_make_item(i) for i in range(7)]
        with pytest.raises(ValueError, match="divisible"):
            _dispatch_tensors(items, dp_size=4)

    def test_pad_then_dispatch_end_to_end(self):
        items = [_make_item(i) for i in range(7)]
        (padded,) = _pad_eval_batch((items,), dp_size=4)
        splits, _ = _dispatch_tensors(padded, dp_size=4)
        assert all(len(group) == 2 for group in splits)
        assert _count_dummies(_flatten_splits(splits)) == 1

    def test_make_dummy_eval_item_schema(self):
        template: dict[str, object] = {
            "input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[True, True, False]], dtype=torch.bool),
            "loss_mask": torch.tensor([[1, 1, 0]], dtype=torch.int32),
            "multi_modal_input": [{"image": torch.tensor([1.0])}],
            "meta": {"tag": ["x"]},
        }

        dummy = make_dummy_eval_item(template)
        assert set(dummy.keys()) == set(template.keys())
        assert dummy["multi_modal_input"] == [{}]
        torch.testing.assert_close(
            dummy["attention_mask"],
            torch.zeros((1, 1), dtype=torch.bool),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            dummy["loss_mask"],
            torch.zeros((1, 1), dtype=cast(torch.Tensor, template["loss_mask"]).dtype),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            dummy["input_ids"],
            torch.zeros((1, 1), dtype=cast(torch.Tensor, template["input_ids"]).dtype),
            rtol=0.0,
            atol=0.0,
        )
        cast(dict[str, list[str]], template["meta"])["tag"].append("y")
        assert dummy["meta"] == {"tag": ["x"]}

    def test_pad_eval_batch_keeps_multimodal_payload_aligned(self):
        items: list[dict[str, object]] = []
        for _ in range(3):
            items.append(
                {
                    "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
                    "attention_mask": torch.tensor(
                        [[True, True, True]], dtype=torch.bool
                    ),
                    "loss_mask": torch.tensor([[1, 1, 1]], dtype=torch.int32),
                    "multi_modal_input": [
                        {
                            "pixel_values": torch.ones((1, 2, 2), dtype=torch.float32),
                            "image_grid_thw": torch.tensor(
                                [[1, 1, 1]], dtype=torch.int32
                            ),
                        }
                    ],
                }
            )

        (padded,) = _pad_eval_batch((items,), dp_size=4)
        batched = concat_padded_tensors(padded)
        mb_list = split_padded_tensor_dict_into_mb_list(batched, MicroBatchSpec())

        assert len(padded) == 4
        assert len(batched["multi_modal_input"]) == batched["attention_mask"].shape[0]
        assert padded[-1]["multi_modal_input"] == [{}]
        for mb in mb_list.mbs:
            assert len(mb["multi_modal_input"]) == mb["attention_mask"].shape[0]

    def test_pad_eval_batch_group_size_odd_dp(self):
        items = [_make_item(i) for i in range(2)]
        (padded,) = _pad_eval_batch((items,), dp_size=3, group_size=2)
        assert len(padded) == 6
        assert len(padded) % 2 == 0
        assert len(padded) % 3 == 0
        assert _count_dummies(padded) == 4

    def test_pad_eval_batch_group_size_even_dp(self):
        items = [_make_item(i) for i in range(6)]
        (padded,) = _pad_eval_batch((items,), dp_size=4, group_size=2)
        assert len(padded) == 8
        assert len(padded) % 2 == 0
        assert len(padded) % 4 == 0
        assert _count_dummies(padded) == 2

    def test_pad_eval_batch_group_size_no_pad_needed(self):
        items = [_make_item(i) for i in range(6)]
        (padded,) = _pad_eval_batch((items,), dp_size=3, group_size=2)
        assert len(padded) == 6
        assert _count_dummies(padded) == 0

    def test_pad_eval_batch_default_group_size_unchanged(self):
        items = [_make_item(i) for i in range(7)]
        (padded,) = _pad_eval_batch((items,), dp_size=4)
        assert len(padded) == 8
        assert _count_dummies(padded) == 1

    def test_compute_total_loss_weight_allows_local_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        mb_list = MicroBatchList(
            data={},
            mb_spec=MicroBatchSpec(),
            mbs=[{"attention_mask": torch.zeros((1, 1), dtype=torch.bool)}],
            group_lens=[1],
        )

        def _mock_all_reduce(
            tensor: torch.Tensor, group: dist.ProcessGroup | None = None
        ):
            del group
            tensor.add_(3.0)

        monkeypatch.setattr(dist, "all_reduce", _mock_all_reduce)

        total_weight = compute_total_loss_weight(
            mb_list=mb_list,
            loss_weight_fn=lambda _mb: torch.tensor(0.0),
            dp_group=cast(dist.ProcessGroup, object()),
        )

        torch.testing.assert_close(total_weight, torch.tensor(3.0), rtol=0.0, atol=0.0)


class TestRWDispatchGrouping:
    def test_dispatch_group_size_preserves_rw_pairs(self):
        dp_size = 4
        items = _build_rw_batch(n_pairs=8)
        (padded,) = _pad_eval_batch((items,), dp_size=dp_size, group_size=2)
        assert len(padded) == 16
        splits, _ = _dispatch_tensors(padded, dp_size=dp_size, group_size=2)

        for shard_items in splits:
            assert len(shard_items) % 2 == 0, (
                f"Shard received odd item count {len(shard_items)}"
            )
            for j in range(0, len(shard_items), 2):
                c_meta = shard_items[j].get("meta", {})
                r_meta = shard_items[j + 1].get("meta", {})
                if c_meta.get("pair") is not None:
                    assert c_meta["pair"] == r_meta["pair"], (
                        f"Pair split across items: {c_meta} vs {r_meta}"
                    )
                    assert c_meta["role"] == "chosen"
                    assert r_meta["role"] == "rejected"

    def test_dispatch_group_size_rw_5_examples_dp4(self):
        dp_size = 4
        items = _build_rw_batch(n_pairs=5)
        assert len(items) == 10

        (padded,) = _pad_eval_batch((items,), dp_size=dp_size, group_size=2)
        assert len(padded) == 16
        assert len(padded) % (dp_size * 2) == 0

        splits, _ = _dispatch_tensors(padded, dp_size=dp_size, group_size=2)
        assert len(splits) == dp_size
        for shard_items in splits:
            assert len(shard_items) % 2 == 0, (
                f"Shard received odd item count {len(shard_items)}, view(-1, 2) would fail"
            )

    def test_dispatch_group_size_not_divisible_raises(self):
        items = _build_rw_batch(n_pairs=3)
        items.append(items[0])
        with pytest.raises(ValueError, match="divisible by group_size"):
            _dispatch_tensors(items, dp_size=2, group_size=2)

    def test_dispatch_group_size_1_unchanged(self):
        items = [_make_item(i, seqlen=i + 1) for i in range(8)]
        _, indices_default = _dispatch_tensors(items, dp_size=4)
        _, indices_gs1 = _dispatch_tensors(items, dp_size=4, group_size=1)
        assert indices_default == indices_gs1

    @pytest.mark.parametrize(
        "dp_size, group_size, n_items",
        [
            (3, 2, 12),
            (4, 2, 16),
            (6, 2, 24),
            (4, 3, 24),
            (3, 2, 6),
            (4, 2, 8),
        ],
    )
    def test_pad_and_dispatch_group_matrix(
        self, dp_size: int, group_size: int, n_items: int
    ):
        items = [_make_item(i, seqlen=i % 5 + 1) for i in range(n_items)]
        (padded,) = _pad_eval_batch((items,), dp_size=dp_size, group_size=group_size)
        assert len(padded) % (dp_size * group_size) == 0

        splits, _ = _dispatch_tensors(padded, dp_size=dp_size, group_size=group_size)
        assert len(splits) == dp_size
        for shard in splits:
            assert len(shard) % group_size == 0

    @pytest.mark.parametrize(
        "dp_size, group_size, n_raw",
        [
            (3, 2, 5),
            (4, 2, 7),
            (4, 3, 10),
            (6, 2, 3),
        ],
    )
    def test_pad_aligns_non_divisible_input(
        self, dp_size: int, group_size: int, n_raw: int
    ):
        items = [_make_item(i) for i in range(n_raw)]
        (padded,) = _pad_eval_batch((items,), dp_size=dp_size, group_size=group_size)
        target = dp_size * group_size
        assert len(padded) % target == 0
        assert len(padded) >= n_raw


class TestRWDummyPairSemantics:
    def test_rw_loss_weight_counts_only_valid_pairs(self):
        input_data = _make_rw_input([5, 4, 0, 0])

        loss_weight = _rw_loss_weight(input_data)

        torch.testing.assert_close(loss_weight, torch.tensor(1.0), rtol=0.0, atol=0.0)

    def test_compute_rw_loss_ignores_dummy_pairs_in_loss_and_metrics(self):
        tracker = DistributedStatsTracker()
        input_data = _make_rw_input([5, 4, 0, 0])
        scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        expected_loss = -F.logsigmoid(torch.tensor(2.0))

        with patch("areal.trainer.rw.rw_engine.stats_tracker", tracker):
            loss = compute_rw_loss(scores, input_data)

        stats = tracker.export(reset=True)

        torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)
        assert stats["n_pairs"] == 1.0
        assert stats["correct_ratio/avg"] == 1.0
        assert stats["pos_score/avg"] == 3.0
        assert stats["neg_score/avg"] == 1.0
        torch.testing.assert_close(
            torch.tensor(stats["loss/avg"]),
            expected_loss,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_compute_rw_loss_returns_zero_for_all_dummy_pairs(self):
        tracker = DistributedStatsTracker()
        input_data = _make_rw_input([0, 0])

        with patch("areal.trainer.rw.rw_engine.stats_tracker", tracker):
            loss = compute_rw_loss(torch.tensor([], dtype=torch.float32), input_data)

        stats = tracker.export(reset=True)

        torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0.0, atol=0.0)
        assert stats["n_pairs"] == 0.0
        assert "loss/avg" not in stats


class TestExplicitEvalDispatchControllers:
    def test_lm_controller_evaluate_lm_explicitly_pads_batch(self):
        controller = LMController.__new__(LMController)
        cast(Any, controller).train_alloc = SimpleNamespace(
            parallel=SimpleNamespace(dp_size=4)
        )
        captured: dict[str, Any] = {}

        def _capture_call(self, method: str, *args, **kwargs):
            captured["method"] = method
            captured["args"] = args
            captured["kwargs"] = kwargs
            return None

        controller._custom_function_call = MethodType(_capture_call, controller)

        items = [_make_item(i) for i in range(7)]
        controller.evaluate_lm(items)

        assert captured["method"] == "evaluate_lm"
        padded_items = cast(list[dict[str, object]], captured["args"][0])
        assert len(padded_items) == 8
        assert captured["kwargs"] == {"rpc_meta": {"broadcast": True}}

    def test_rw_controller_evaluate_rw_explicitly_pads_pairs(self):
        controller = RWController.__new__(RWController)
        cast(Any, controller).train_alloc = SimpleNamespace(
            parallel=SimpleNamespace(dp_size=4)
        )
        captured: dict[str, Any] = {}

        def _capture_call(self, method: str, *args, **kwargs):
            captured["method"] = method
            captured["args"] = args
            captured["kwargs"] = kwargs
            return None

        controller._custom_function_call = MethodType(_capture_call, controller)

        items = _build_rw_batch(n_pairs=5)
        controller.evaluate_rw(items)

        assert captured["method"] == "evaluate_rw"
        assert captured["kwargs"]["group_size"] == 2
        padded_items = cast(list[dict[str, object]], captured["args"][0])
        assert len(padded_items) == 16

    def test_evaluate_rw_logs_zero_pair_denominator_for_all_dummy_local_batch(self):
        tracker = DistributedStatsTracker()

        class _DummyEngine:
            def eval(self) -> None:
                return None

            def eval_batch(self, **kwargs) -> None:
                del kwargs
                return None

        with patch("areal.trainer.rw.rw_engine.stats_tracker", tracker):
            RWEngine(_DummyEngine())._evaluate_rw(_make_rw_input([0, 0]))

        stats = tracker.export(reset=True)

        assert stats["n_pairs"] == 0.0
        assert "loss/avg" not in stats
