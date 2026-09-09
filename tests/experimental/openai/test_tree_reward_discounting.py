# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from areal.api import ModelResponse
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward


def _make_dummy_completion(cid: str, content: str = "ok") -> ChatCompletion:
    return ChatCompletion(
        id=cid,
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(content=content, role="assistant"),
            )
        ],
        created=1000,
        model="test-model",
        object="chat.completion",
    )


def _make_dummy_response(input_tokens, output_tokens) -> ModelResponse:
    return ModelResponse(
        input_tokens=list(input_tokens),
        output_tokens=list(output_tokens),
        output_logprobs=[0.0] * len(output_tokens),
        output_versions=[0] * len(output_tokens),
        stop_reason="stop",
    )


class TestTreeRewardDiscounting:
    def test_linear_chain_equivalence_to_linear_mode(self):
        # 3-turn linear conversation: Turn 1 -> Turn 2 -> Turn 3
        cache_tree = InteractionCache(session_id="linear_tree")
        cache_linear = InteractionCache(session_id="linear_seq")

        # Turn 1
        m1 = [{"role": "user", "content": "hello"}]
        c1 = _make_dummy_completion("c1")
        i1_t = InteractionWithTokenLogpReward(
            messages=m1,
            output_message_list=[{"role": "assistant", "content": "hi"}],
            completion=c1,
        )
        i1_l = InteractionWithTokenLogpReward(
            messages=m1,
            output_message_list=[{"role": "assistant", "content": "hi"}],
            completion=c1,
        )
        cache_tree["c1"] = i1_t
        cache_linear["c1"] = i1_l
        cache_tree.set_reward("c1", 0.1)
        cache_linear.set_reward("c1", 0.1)

        # Turn 2
        m2 = m1 + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "step 2"}]
        c2 = _make_dummy_completion("c2")
        i2_t = InteractionWithTokenLogpReward(
            messages=m2,
            output_message_list=[{"role": "assistant", "content": "reply 2"}],
            completion=c2,
        )
        i2_l = InteractionWithTokenLogpReward(
            messages=m2,
            output_message_list=[{"role": "assistant", "content": "reply 2"}],
            completion=c2,
        )
        cache_tree["c2"] = i2_t
        cache_linear["c2"] = i2_l
        cache_tree.set_reward("c2", 0.2)
        cache_linear.set_reward("c2", 0.2)

        # Turn 3
        m3 = m2 + [{"role": "assistant", "content": "reply 2"}, {"role": "user", "content": "step 3"}]
        c3 = _make_dummy_completion("c3")
        i3_t = InteractionWithTokenLogpReward(
            messages=m3,
            output_message_list=[{"role": "assistant", "content": "reply 3"}],
            completion=c3,
        )
        i3_l = InteractionWithTokenLogpReward(
            messages=m3,
            output_message_list=[{"role": "assistant", "content": "reply 3"}],
            completion=c3,
        )
        cache_tree["c3"] = i3_t
        cache_linear["c3"] = i3_l
        cache_tree.set_reward("c3", 1.0)
        cache_linear.set_reward("c3", 1.0)

        gamma = 0.9
        res_tree = cache_tree.apply_reward_discount(turn_discount=gamma, discount_mode="tree")
        res_linear = cache_linear.apply_reward_discount(turn_discount=gamma, discount_mode="linear")

        # In a linear chain, tree discount is mathematically identical to linear discount
        for cid in ["c1", "c2", "c3"]:
            assert pytest.approx(res_tree[cid].reward) == res_linear[cid].reward
            assert res_tree[cid].original_reward is not None

        # Turn 3: 1.0
        assert pytest.approx(res_tree["c3"].reward) == 1.0
        assert pytest.approx(res_tree["c3"].original_reward) == 1.0

        # Turn 2: 0.2 + 0.9 * 1.0 = 1.1
        assert pytest.approx(res_tree["c2"].reward) == 1.1
        assert pytest.approx(res_tree["c2"].original_reward) == 0.2

        # Turn 1: 0.1 + 0.9 * 1.1 = 1.09
        assert pytest.approx(res_tree["c1"].reward) == 1.09
        assert pytest.approx(res_tree["c1"].original_reward) == 0.1

    def test_branching_tree_isolation(self):
        # Root (0.0) -> Branch A (0.0) -> Leaf A1 (+1.0)
        #            -> Branch B (0.0) -> Leaf B1 (-1.0)
        cache = InteractionCache(session_id="branching_tree")

        # Root
        m_root = [{"role": "user", "content": "solve task"}]
        i_root = InteractionWithTokenLogpReward(
            messages=m_root,
            output_message_list=[{"role": "assistant", "content": "root"}],
            completion=_make_dummy_completion("root"),
        )
        cache["root"] = i_root
        cache.set_reward("root", 0.0)

        # Branch A
        m_a = m_root + [{"role": "assistant", "content": "root"}, {"role": "user", "content": "explore path A"}]
        i_a = InteractionWithTokenLogpReward(
            messages=m_a,
            output_message_list=[{"role": "assistant", "content": "a"}],
            completion=_make_dummy_completion("a"),
        )
        cache["a"] = i_a
        cache.set_reward("a", 0.0)

        # Leaf A1
        m_a1 = m_a + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "finalize path A"}]
        i_a1 = InteractionWithTokenLogpReward(
            messages=m_a1,
            output_message_list=[{"role": "assistant", "content": "a1"}],
            completion=_make_dummy_completion("a1"),
        )
        cache["a1"] = i_a1
        cache.set_reward("a1", 1.0)

        # Branch B
        m_b = m_root + [{"role": "assistant", "content": "root"}, {"role": "user", "content": "explore path B"}]
        i_b = InteractionWithTokenLogpReward(
            messages=m_b,
            output_message_list=[{"role": "assistant", "content": "b"}],
            completion=_make_dummy_completion("b"),
        )
        cache["b"] = i_b
        cache.set_reward("b", 0.0)

        # Leaf B1
        m_b1 = m_b + [{"role": "assistant", "content": "b"}, {"role": "user", "content": "finalize path B"}]
        i_b1 = InteractionWithTokenLogpReward(
            messages=m_b1,
            output_message_list=[{"role": "assistant", "content": "b1"}],
            completion=_make_dummy_completion("b1"),
        )
        cache["b1"] = i_b1
        cache.set_reward("b1", -1.0)

        # Verify parent links were automatically established
        assert i_a.parent is i_root
        assert i_a1.parent is i_a
        assert i_b.parent is i_root
        assert i_b1.parent is i_b

        gamma = 0.8
        res = cache.apply_reward_discount(turn_discount=gamma, discount_mode="tree")

        # Leaf A1 retains +1.0
        assert pytest.approx(res["a1"].reward) == 1.0
        # Branch A receives 0.0 + 0.8 * 1.0 = +0.8 (NOT corrupted by Branch B's -1.0)
        assert pytest.approx(res["a fatal" if False else "a"].reward) == 0.8

        # Leaf B1 retains -1.0
        assert pytest.approx(res["b1"].reward) == -1.0
        # Branch B receives 0.0 + 0.8 * (-1.0) = -0.8
        assert pytest.approx(res["b"].reward) == -0.8

        # Root receives mean of child returns: 0.0 + 0.8 * (0.8 + (-0.8)) / 2 = 0.0
        assert pytest.approx(res["root"].reward) == 0.0

    def test_tensor_cache_sync_and_original_reward(self):
        cache = InteractionCache(session_id="cache_sync")
        m1 = [{"role": "user", "content": "hello"}]
        resp1 = _make_dummy_response([1, 2, 3], [4, 5])
        i1 = InteractionWithTokenLogpReward(
            messages=m1,
            output_message_list=[{"role": "assistant", "content": "hi"}],
            completion=_make_dummy_completion("c1"),
            model_response=resp1,
        )
        cache["c1"] = i1
        cache.set_reward("c1", 0.5)

        # Materialize tensor dict before discounting
        tensor_dict_before = i1.to_tensor_dict()
        assert float(tensor_dict_before["rewards"].item()) == 0.5
        assert float(tensor_dict_before["original_rewards"].item()) == 0.5

        m2 = m1 + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "step 2"}]
        resp2 = _make_dummy_response([1, 2, 3, 4, 5, 6], [7, 8])
        i2 = InteractionWithTokenLogpReward(
            messages=m2,
            output_message_list=[{"role": "assistant", "content": "reply 2"}],
            completion=_make_dummy_completion("c2"),
            model_response=resp2,
        )
        cache["c2"] = i2
        cache.set_reward("c2", 2.0)

        cache.apply_reward_discount(turn_discount=0.5, discount_mode="tree")

        # Check that tensor cache for c1 was updated
        tensor_dict_after = i1.to_tensor_dict()
        # c1 discounted reward = 0.5 + 0.5 * 2.0 = 1.5
        assert pytest.approx(float(tensor_dict_after["rewards"].item())) == 1.5
        # original reward remains 0.5
        assert pytest.approx(float(tensor_dict_after["original_rewards"].item())) == 0.5


class TestInteractionTurnIdResolution:
    def test_turn_id_property(self):
        m1 = [{"role": "user", "content": "turn 0"}]
        i1 = InteractionWithTokenLogpReward(messages=m1)
        assert i1.turn_id == 0

        m2 = m1 + [{"role": "assistant", "content": "ans 0"}, {"role": "user", "content": "turn 1"}]
        i2 = InteractionWithTokenLogpReward(messages=m2, parent=i1)
        assert i2.turn_id == 1

        m3 = m2 + [{"role": "assistant", "content": "ans 1"}, {"role": "user", "content": "turn 2"}]
        i3 = InteractionWithTokenLogpReward(messages=m3, parent=i2)
        assert i3.turn_id == 2

    def test_hf_mode_turn_ids_tensor(self):
        m1 = [{"role": "user", "content": "turn 0"}]
        resp1 = _make_dummy_response([10, 11], [12, 13])
        i1 = InteractionWithTokenLogpReward(
            messages=m1,
            chat_template_type="hf",
            model_response=resp1,
        )
        t1 = i1.to_tensor_dict()
        turn_ids_1 = t1["turn_ids"].squeeze(0).tolist()
        assert turn_ids_1 == [-1, -1, 0, 0]

        m2 = m1 + [{"role": "assistant", "content": "ans 0"}, {"role": "user", "content": "turn 1"}]
        resp2 = _make_dummy_response([10, 11, 12, 13, 14], [15, 16, 17])
        i2 = InteractionWithTokenLogpReward(
            messages=m2,
            chat_template_type="hf",
            parent=i1,
            model_response=resp2,
        )
        t2 = i2.to_tensor_dict()
        turn_ids_2 = t2["turn_ids"].squeeze(0).tolist()
        # Input tokens masked with -1, generated tokens assigned turn_id=1
        assert turn_ids_2 == [-1, -1, -1, -1, -1, 1, 1, 1]
