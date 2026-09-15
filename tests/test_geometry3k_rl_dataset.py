# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from examples.vlm import geometry3k_grpo

from areal.dataset import geometry3k
from areal.workflow.openai import geometry3k_agent


def _load_sample(monkeypatch, template):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template=template,
    )
    processor = SimpleNamespace(
        tokenizer=tokenizer,
        image_processor=SimpleNamespace(image_processor_type="generic"),
        image_token="<image>",
    )
    dataset = Dataset.from_list(
        [{"problem": "Use <think>...</think>. Find x.", "answer": "42", "images": []}]
    )
    monkeypatch.setattr(geometry3k, "load_dataset", lambda **_: dataset)
    sample = geometry3k.get_geometry3k_rl_dataset("unused", "train", processor)[0]
    return tokenizer, sample


@pytest.mark.parametrize(
    "assistant_header",
    [
        "<|im_start|>assistant\n",
        "<start_of_turn>model\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "Assistant: ",
    ],
)
@pytest.mark.parametrize("prefill", ["", "<think>\n", "<think>\n\n</think>\n\n"])
def test_geometry3k_dataset_detects_prefill_without_role_separator_assumptions(
    monkeypatch, assistant_header, prefill
):
    """Only template-added open thinking tags affect either reward entrypoint."""
    template = (
        "{{ messages[0]['content'] }}\nUSER_END\n"
        "{% if add_generation_prompt %}"
        "{{ " + json.dumps(assistant_header + prefill) + " }}"
        "{% endif %}"
    )
    tokenizer, sample = _load_sample(monkeypatch, template)
    expected = prefill == "<think>\n"

    assert sample["think_prefilled"] is expected
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample["messages_chat"][0]["content"][1]["text"]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    assert sample["messages"] == rendered

    monkeypatch.setattr(geometry3k_agent, "acc_reward", lambda *_: 1.0)
    completion = ("" if expected else "<think>") + "reason</think>\\boxed{42}"
    reward = geometry3k_grpo.geometry3k_reward_fn(
        prompt=sample["messages"],
        completions=completion,
        prompt_ids=[1],
        completion_ids=[2],
        **sample,
    )
    assert reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    "template",
    [
        "{% if add_generation_prompt %}changed-prefix{% endif %}"
        "{{ messages[0]['content'] }}"
        "{% if add_generation_prompt %}Assistant: <think>\n{% endif %}",
        "{{ messages[0]['content'] }}Assistant: <think>\n",
        "{{ messages[0]['content'] }}<think>\n",
    ],
)
def test_geometry3k_dataset_does_not_guess_prefill_without_an_appended_suffix(
    monkeypatch, template
):
    """Rewritten/unchanged prompts must not be mistaken for an added prefill."""
    _, sample = _load_sample(monkeypatch, template)

    assert sample["think_prefilled"] is False
