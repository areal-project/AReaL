"""Tests for examples.swe.prefix_matchers."""

import pytest

from examples.swe.prefix_matchers import _messages_match, swe_prefix_matcher

# ---------------------------------------------------------------------------
# _messages_match: system / user messages
# ---------------------------------------------------------------------------


class TestMessagesMatchBasic:
    def test_identical_user_messages(self):
        m = {"role": "user", "content": "hello"}
        assert _messages_match(m, m)

    def test_identical_system_messages(self):
        m = {"role": "system", "content": "You are helpful."}
        assert _messages_match(m, m)

    def test_different_content(self):
        a = {"role": "user", "content": "hello"}
        b = {"role": "user", "content": "world"}
        assert not _messages_match(a, b)

    def test_different_role(self):
        a = {"role": "user", "content": "hello"}
        b = {"role": "assistant", "content": "hello"}
        assert not _messages_match(a, b)

    def test_thinking_must_match(self):
        a = {"role": "assistant", "content": "ok", "thinking": "step1"}
        b = {"role": "assistant", "content": "ok", "thinking": "step2"}
        assert not _messages_match(a, b)

    def test_thinking_present_vs_absent(self):
        a = {"role": "assistant", "content": "ok", "thinking": "step1"}
        b = {"role": "assistant", "content": "ok"}
        assert not _messages_match(a, b)

    def test_thinking_both_match(self):
        a = {"role": "assistant", "content": "ok", "thinking": "step1"}
        b = {"role": "assistant", "content": "ok", "thinking": "step1"}
        assert _messages_match(a, b)


# ---------------------------------------------------------------------------
# _messages_match: tool_calls (assistant with tool invocations)
# ---------------------------------------------------------------------------


class TestMessagesMatchToolCalls:
    def _make_tc(self, tc_id, name="Bash", args='{"command": "ls"}'):
        return {
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": args},
        }

    def test_same_id_different_args_matches(self):
        """Core case: same tool_call ID but different arguments should match."""
        a = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                self._make_tc("tc_1", args='{"command": "cd /testbed && git log"}')
            ],
        }
        b = {
            "role": "assistant",
            "content": "",
            "tool_calls": [self._make_tc("tc_1", args='{"command": "git log"}')],
        }
        assert _messages_match(a, b)

    def test_different_id_does_not_match(self):
        a = {
            "role": "assistant",
            "content": "",
            "tool_calls": [self._make_tc("tc_1")],
        }
        b = {
            "role": "assistant",
            "content": "",
            "tool_calls": [self._make_tc("tc_2")],
        }
        assert not _messages_match(a, b)

    def test_different_tool_call_count(self):
        a = {
            "role": "assistant",
            "content": "",
            "tool_calls": [self._make_tc("tc_1")],
        }
        b = {
            "role": "assistant",
            "content": "",
            "tool_calls": [self._make_tc("tc_1"), self._make_tc("tc_2")],
        }
        assert not _messages_match(a, b)

    def test_multiple_tool_calls_same_ids(self):
        a = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                self._make_tc("tc_1", args='{"command": "cd /testbed && ls"}'),
                self._make_tc("tc_2", args='{"content": "foo  \\n"}'),
            ],
        }
        b = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                self._make_tc("tc_1", args='{"command": "ls"}'),
                self._make_tc("tc_2", args='{"content": "foo\\n"}'),
            ],
        }
        assert _messages_match(a, b)

    def test_tool_calls_present_vs_absent(self):
        a = {
            "role": "assistant",
            "content": "text",
            "tool_calls": [self._make_tc("tc_1")],
        }
        b = {"role": "assistant", "content": "text"}
        assert not _messages_match(a, b)

    def test_content_differs_with_tool_calls(self):
        """Even if tool_call IDs match, content must also match."""
        a = {
            "role": "assistant",
            "content": "thinking A",
            "tool_calls": [self._make_tc("tc_1")],
        }
        b = {
            "role": "assistant",
            "content": "thinking B",
            "tool_calls": [self._make_tc("tc_1")],
        }
        assert not _messages_match(a, b)


# ---------------------------------------------------------------------------
# _messages_match: tool response (role=tool)
# ---------------------------------------------------------------------------


class TestMessagesMatchToolResponse:
    def test_same_tool_call_id_different_content(self):
        a = {"role": "tool", "content": "output v1", "tool_call_id": "tc_1"}
        b = {"role": "tool", "content": "output v2", "tool_call_id": "tc_1"}
        assert _messages_match(a, b)

    def test_different_tool_call_id(self):
        a = {"role": "tool", "content": "output", "tool_call_id": "tc_1"}
        b = {"role": "tool", "content": "output", "tool_call_id": "tc_2"}
        assert not _messages_match(a, b)

    def test_tool_vs_non_tool_role(self):
        a = {"role": "tool", "content": "output", "tool_call_id": "tc_1"}
        b = {"role": "user", "content": "output"}
        assert not _messages_match(a, b)


# ---------------------------------------------------------------------------
# swe_prefix_matcher: integration
# ---------------------------------------------------------------------------


class TestSwePrefixMatcher:
    def test_exact_prefix(self):
        a = [{"role": "user", "content": "hi"}]
        b = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert swe_prefix_matcher(a, b)

    def test_a_longer_than_b(self):
        a = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        b = [{"role": "user", "content": "hi"}]
        assert not swe_prefix_matcher(a, b)

    def test_empty_prefix(self):
        assert swe_prefix_matcher([], [{"role": "user", "content": "hi"}])

    def test_both_empty(self):
        assert swe_prefix_matcher([], [])

    def test_cd_prefix_stripped_in_tool_call(self):
        """Full integration: cd prefix stripped by CLI doesn't break matching."""
        tc = {
            "id": "tc_1",
            "type": "function",
            "function": {
                "name": "Bash",
                "arguments": '{"command": "cd /testbed && git log"}',
            },
        }
        tc_stripped = {
            "id": "tc_1",
            "type": "function",
            "function": {
                "name": "Bash",
                "arguments": '{"command": "git log"}',
            },
        }
        parent_data = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "", "tool_calls": [tc]},
        ]
        child_msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "", "tool_calls": [tc_stripped]},
            {"role": "tool", "content": "commit abc123", "tool_call_id": "tc_1"},
            {"role": "user", "content": "Now run tests"},
        ]
        assert swe_prefix_matcher(parent_data, child_msgs)

    def test_trailing_whitespace_stripped_in_tool_call(self):
        """Trailing whitespace stripped by CLI doesn't break matching."""
        tc_original = {
            "id": "tc_1",
            "type": "function",
            "function": {
                "name": "Write",
                "arguments": '{"content": "def foo():\\n    \\n    pass"}',
            },
        }
        tc_stripped = {
            "id": "tc_1",
            "type": "function",
            "function": {
                "name": "Write",
                "arguments": '{"content": "def foo():\\n\\n    pass"}',
            },
        }
        a = [{"role": "assistant", "content": "", "tool_calls": [tc_original]}]
        b = [
            {"role": "assistant", "content": "", "tool_calls": [tc_stripped]},
            {"role": "tool", "content": "OK", "tool_call_id": "tc_1"},
        ]
        assert swe_prefix_matcher(a, b)

    def test_system_prompt_mismatch_fails(self):
        """If system prompt changes (e.g. compaction), match fails immediately."""
        a = [{"role": "system", "content": "You are helpful."}]
        b = [{"role": "system", "content": "You are helpful. [Summary: ...]"}]
        assert not swe_prefix_matcher(a, b)

    def test_content_mismatch_in_user_msg_fails(self):
        a = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question A"},
        ]
        b = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question B"},
        ]
        assert not swe_prefix_matcher(a, b)

    def test_mixed_cd_and_trailing_whitespace(self):
        """Multiple mismatch types in one chain still match."""
        parent_data = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": '{"command": "cd /testbed/pkg && npm test"}',
                        },
                    }
                ],
            },
        ]
        child_msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": '{"command": "npm test"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "PASS", "tool_call_id": "tc_1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_2",
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "arguments": '{"content": "line1\\n    \\nline3"}',
                        },
                    }
                ],
            },
        ]
        # parent_data is prefix of child_msgs up to index 2
        assert swe_prefix_matcher(parent_data, child_msgs)


# ---------------------------------------------------------------------------
# Integration with InteractionCache
# ---------------------------------------------------------------------------


class TestSwePrefixMatcherWithCache:
    """End-to-end: swe_prefix_matcher plugged into InteractionCache."""

    @pytest.fixture()
    def mock_interaction(self):
        from unittest.mock import MagicMock

        from openai.types.responses.response import Response
        from openai.types.responses.response_output_message import ResponseOutputMessage
        from openai.types.responses.response_output_text import ResponseOutputText

        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        def _create(id, messages, response_text="ok", created=1):
            output = ResponseOutputMessage(
                id=id,
                content=[
                    ResponseOutputText(
                        annotations=[], text=response_text, type="output_text"
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
            response = Response(
                id=id,
                created_at=float(created),
                model="None",
                object="response",
                output=[output],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            )
            completion = MagicMock()
            completion.id = id
            completion.created = created
            model_response = MagicMock()

            interaction = InteractionWithTokenLogpReward(
                model_response=model_response,
                chat_template_type="concat",
            )
            interaction.messages = messages
            interaction.output_message_list = [output.model_dump(exclude_none=True)]
            interaction.completion = completion
            interaction.response = response
            return interaction

        return _create

    def test_full_chain_with_cd_prefix_no_filters(self, mock_interaction):
        """Without any preprocessors, swe_prefix_matcher preserves the chain."""
        from areal.experimental.openai.cache import InteractionCache

        m1 = [{"role": "user", "content": "Hello"}]
        a1 = mock_interaction(id="a1", messages=m1, created=1)
        out_a1 = a1.response.output[0].model_dump(exclude_none=True)

        # Turn 2: output has cd prefix
        m2 = m1 + [out_a1, {"role": "user", "content": "Run command"}]
        a2 = mock_interaction(id="a2", messages=m2, created=2)
        a2.output_message_list = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": '{"command": "cd /testbed && git log"}',
                        },
                    }
                ],
            }
        ]

        # Turn 3: agent stripped cd prefix
        m3 = m2 + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": '{"command": "git log"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "abc123 init", "tool_call_id": "tc_1"},
        ]
        a3 = mock_interaction(
            id="a3", messages=m3, response_text="I see the log", created=3
        )
        out_a3 = a3.response.output[0].model_dump(exclude_none=True)

        # Turn 4: trailing whitespace stripped
        m4 = m3 + [out_a3, {"role": "user", "content": "Write a fix"}]
        a4 = mock_interaction(id="a4", messages=m4, created=4)
        a4.output_message_list = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_2",
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "arguments": '{"content": "def fix():\\n    \\n    pass"}',
                        },
                    }
                ],
            }
        ]

        m5 = m4 + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_2",
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "arguments": '{"content": "def fix():\\n\\n    pass"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "Written", "tool_call_id": "tc_2"},
        ]
        a5 = mock_interaction(id="a5", messages=m5, response_text="All done", created=5)

        all_interactions = [a1, a2, a3, a4, a5]

        # --- Default matcher: chain breaks ---
        for a in all_interactions:
            a.parent = None
        cache_default = InteractionCache(session_id="test")
        for a in all_interactions:
            cache_default[a.completion.id] = a

        def _chain(leaf):
            ids = []
            cur = leaf
            while cur is not None:
                ids.append(cur.interaction_id)
                cur = cur.parent
            return list(reversed(ids))

        leaves_default = cache_default.export_interactions(style="concat")
        assert len(leaves_default) > 1, "Default matcher should break the chain"

        # --- swe_prefix_matcher: full chain preserved ---
        for a in all_interactions:
            a.parent = None
        cache_swe = InteractionCache(
            session_id="test", prefix_matcher=swe_prefix_matcher
        )
        for a in all_interactions:
            cache_swe[a.completion.id] = a

        leaves_swe = cache_swe.export_interactions(style="concat")
        chains = [_chain(leaf) for leaf in leaves_swe.values()]
        assert len(leaves_swe) == 1, f"Expected 1 leaf, got {len(leaves_swe)}: {chains}"
        assert chains[0] == ["a1", "a2", "a3", "a4", "a5"], f"Got: {chains[0]}"
