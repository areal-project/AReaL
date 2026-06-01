import importlib.util
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

try:
    import transformers.tokenization_utils_base  # noqa: F401
except ImportError:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = object
    sys.modules["transformers"] = transformers_stub

    tokenizer_base_stub = types.ModuleType("transformers.tokenization_utils_base")
    tokenizer_base_stub.PreTrainedTokenizerBase = object
    sys.modules["transformers.tokenization_utils_base"] = tokenizer_base_stub

math_utils_pkg_stub = types.ModuleType("deepscaler.rewards.math_utils")
math_utils_stub = types.ModuleType("deepscaler.rewards.math_utils.utils")
math_utils_stub.extract_answer = lambda answer: answer
math_utils_stub.grade_answer_mathd = lambda answer, ground_truth: answer == ground_truth
math_utils_stub.grade_answer_sympy = lambda answer, ground_truth: answer == ground_truth
sys.modules["deepscaler.rewards.math_utils"] = math_utils_pkg_stub
sys.modules["deepscaler.rewards.math_utils.utils"] = math_utils_stub


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


math_rewardv2 = _load_module("deepscaler.rewards.math_rewardv2", "deepscaler/rewards/math_rewardv2.py")

verl_stub = types.ModuleType("verl")
verl_stub.DataProto = object
sys.modules["verl"] = verl_stub

local_reward_manager = _load_module("test_local_reward_manager", "verl/trainer/reward_manager.py")
server_reward_manager = _load_module("test_server_reward_manager", "verl/trainer/reward_manager_with_server.py")


SPECIAL_TOKEN_IDS = {
    "<Parallel>": 1,
    "</Parallel>": 2,
    "<Thread>": 3,
    "</Thread>": 4,
    "<Outlines>": 5,
    "</Outlines>": 6,
    "<Subtask>": 7,
    "<Trial>": 8,
}


def _parallel_block(outline_tokens):
    return [
        1,  # <Parallel>
        5,  # <Outlines>
        *outline_tokens,
        6,  # </Outlines>
        3, 20, 21, 4,  # <Thread> ... </Thread>
        3, 30, 4,  # <Thread> ... </Thread>
        2,  # </Parallel>
    ]


def _cfg():
    return {
        "subtask_beta": 0.5,
        "trial_beta": 0.25,
        "parallel_ratio_beta": 0.75,
        "latency_alpha": 0.1,
        "group_shaping_eps": 1e-8,
    }


def _sample_scores():
    return [
        {"reward": 1.0, "second_reward": 0.0},
        {"reward": 1.0, "second_reward": 0.0},
        {"reward": -1.0, "second_reward": 0.0},
        {"reward": 2.0, "second_reward": 0.0},
    ]


def _sample_extra_infos():
    return [
        {
            "correct": True,
            "subtask_ratio": 0.2,
            "trial_ratio": 0.4,
            "parallel_ratio": 0.6,
            "acceleration_ratio": 0.1,
        },
        {
            "correct": True,
            "subtask_ratio": 0.6,
            "trial_ratio": 0.2,
            "parallel_ratio": 0.8,
            "acceleration_ratio": 0.3,
        },
        {
            "correct": False,
            "subtask_ratio": 0.5,
            "trial_ratio": 0.5,
            "parallel_ratio": 0.5,
            "acceleration_ratio": 0.5,
        },
        {
            "correct": True,
            "subtask_ratio": 0.5,
            "trial_ratio": 0.5,
            "parallel_ratio": 0.5,
            "acceleration_ratio": 0.5,
        },
    ]


def test_parallel_stats_recognize_subtask_and_trial_tags():
    subtask_tokens = _parallel_block([7, 100, 101])
    subtask_stats = math_rewardv2.get_parallel_stats(subtask_tokens, SPECIAL_TOKEN_IDS)
    assert subtask_stats["num_subtask_tokens"] == 3
    assert subtask_stats["num_trial_tokens"] == 0
    assert math.isclose(subtask_stats["subtask_ratio"], 3 / len(subtask_tokens))

    trial_tokens = _parallel_block([8, 100, 101])
    trial_stats = math_rewardv2.get_parallel_stats(trial_tokens, SPECIAL_TOKEN_IDS)
    assert trial_stats["num_subtask_tokens"] == 0
    assert trial_stats["num_trial_tokens"] == 3
    assert math.isclose(trial_stats["trial_ratio"], 3 / len(trial_tokens))


def test_mixed_subtask_trial_blocks_are_unclassified():
    stats = math_rewardv2.get_parallel_stats(_parallel_block([7, 8, 100]), SPECIAL_TOKEN_IDS)
    assert stats["num_subtask_tokens"] == 0
    assert stats["num_trial_tokens"] == 0
    assert stats["subtask_ratio"] == 0.0
    assert stats["trial_ratio"] == 0.0


def test_parallel_rewardv2_flat_bonus_is_preserved():
    config = math_rewardv2.RewardConfig(
        correct_reward=1.0,
        incorrect_reward=-1.0,
        acceleration_ratio_reward=0.0,
        parallel_rewardv2=0.25,
    )
    stats = {
        "with_parallel": True,
        "acceleration_ratio": 0.0,
    }
    reward, extra_info = math_rewardv2.calculate_reward(config, "", True, stats)
    assert reward == pytest.approx(1.25)
    assert extra_info["parallel_reward"] == pytest.approx(0.25)


def test_parallel_bonus_is_group_normalized_and_correct_only():
    scores, extra_infos = local_reward_manager._apply_groupwise_parallel_bonus(
        _sample_scores(),
        _sample_extra_infos(),
        np.array(["g1", "g1", "g2", "g2"], dtype=object),
        _cfg(),
    )

    expected_bonus_0 = 0.5 * -1.0 + 0.25 * 1.0 + 0.75 * -1.0 + 0.1 * -1.0
    expected_bonus_1 = 0.5 * 1.0 + 0.25 * -1.0 + 0.75 * 1.0 + 0.1 * 1.0

    assert extra_infos[0]["parallel_bonus_subtask_z"] == pytest.approx(-1.0)
    assert extra_infos[1]["parallel_bonus_subtask_z"] == pytest.approx(1.0)
    assert extra_infos[2]["parallel_bonus_subtask_z"] == pytest.approx(0.0)
    assert extra_infos[3]["parallel_bonus_subtask_z"] == pytest.approx(0.0)

    assert extra_infos[0]["parallel_rewardv2_bonus"] == pytest.approx(expected_bonus_0, abs=1e-6)
    assert extra_infos[1]["parallel_rewardv2_bonus"] == pytest.approx(expected_bonus_1, abs=1e-6)
    assert extra_infos[2]["parallel_rewardv2_bonus"] == pytest.approx(0.0)
    assert extra_infos[3]["parallel_rewardv2_bonus"] == pytest.approx(0.0)

    assert scores[0]["reward"] == pytest.approx(1.0 + expected_bonus_0, abs=1e-6)
    assert scores[1]["reward"] == pytest.approx(1.0 + expected_bonus_1, abs=1e-6)
    assert scores[2]["reward"] == pytest.approx(-1.0)
    assert scores[3]["reward"] == pytest.approx(2.0)


def test_server_and_local_parallel_bonus_helpers_match():
    local_scores, local_infos = local_reward_manager._apply_groupwise_parallel_bonus(
        _sample_scores(),
        _sample_extra_infos(),
        np.array(["g1", "g1", "g2", "g2"], dtype=object),
        _cfg(),
    )
    server_scores, server_infos = server_reward_manager._apply_groupwise_parallel_bonus(
        _sample_scores(),
        _sample_extra_infos(),
        np.array(["g1", "g1", "g2", "g2"], dtype=object),
        _cfg(),
    )

    assert local_scores == server_scores
    assert local_infos == server_infos


def test_reward_manager_writes_final_reward_to_last_valid_token(monkeypatch):
    canned_results = [
        (0, {"reward": 1.0, "second_reward": 0.0}, 2, "seq-0", {
            "correct": True,
            "subtask_ratio": 0.2,
            "trial_ratio": 0.4,
            "parallel_ratio": 0.6,
            "acceleration_ratio": 0.1,
        }),
        (1, {"reward": -1.0, "second_reward": 0.0}, 3, "seq-1", {
            "correct": False,
            "subtask_ratio": 0.8,
            "trial_ratio": 0.1,
            "parallel_ratio": 0.9,
            "acceleration_ratio": 0.4,
        }),
    ]

    def fake_process_item(args):
        return canned_results[args[0]]

    monkeypatch.setattr(local_reward_manager, "process_item", fake_process_item)

    class FakeItem:
        batch = {}
        non_tensor_batch = {}

    class FakeData:
        def __init__(self):
            self.batch = {
                "responses": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
            }
            self.non_tensor_batch = {
                "uid": np.array(["g1", "g1"], dtype=object),
            }
            self._items = [FakeItem(), FakeItem()]

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            return self._items[idx]

    manager = local_reward_manager.RewardManager(tokenizer=None, num_examine=0, config=_cfg())
    result = manager(FakeData(), return_dict=True)

    reward_tensor = result["reward_tensor"]["main_reward_tensor"]
    extra_info = result["reward_extra_info"]
    expected_bonus = 0.5 * -1.0 + 0.25 * 1.0 + 0.75 * -1.0 + 0.1 * -1.0

    assert reward_tensor.shape == torch.Size([2, 3])
    assert torch.allclose(reward_tensor[0], torch.tensor([0.0, 1.0 + expected_bonus, 0.0]))
    assert torch.allclose(reward_tensor[1], torch.tensor([0.0, 0.0, -1.0]))
    assert extra_info["parallel_rewardv2_bonus"] == [pytest.approx(expected_bonus, abs=1e-6), pytest.approx(0.0)]
