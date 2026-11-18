# SPDX-License-Identifier: Apache-2.0

import json

import torch

from examples.mopd import experience_opd


def test_experience_hooks_select_and_align_teacher_view(tmp_path, monkeypatch):
    store_path = tmp_path / "experiences.json"
    data_id = "env:data-anatomy-threejs-sub16-bread-display-env@0.1.0"
    store_path.write_text(json.dumps({data_id: "use the tool"}), encoding="utf-8")
    monkeypatch.setattr(experience_opd, "load_hf_tokenizer", lambda _: object())
    monkeypatch.setattr(experience_opd, "_user_turn_markers", lambda _: ([10], [12]))
    monkeypatch.setattr(experience_opd, "_encode", lambda *_: [99, 98])
    rollout_hook = experience_opd.ExperienceRolloutHook(
        experience_path=str(store_path)
    )
    teacher_hook = experience_opd.ExperienceTeacherHook(
        experience_path=str(store_path), tokenizer_path="unused"
    )
    input_data, state = rollout_hook.pre_rollout({"data_id": data_id})
    trajectory = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
        "loss_mask": torch.tensor([[0, 0, 0, 1, 1]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),
    }

    rollout_hook.post_rollout(trajectory, state)
    views, positions = teacher_hook.pre_compute_logp([trajectory])
    aligned = teacher_hook.post_compute_logp(
        [torch.arange(7, dtype=torch.float32).unsqueeze(0)], positions
    )

    assert input_data == {"data_id": data_id}
    assert trajectory["_experience_opd"]["instance_id"] == data_id
    torch.testing.assert_close(
        views[0]["input_ids"], torch.tensor([[10, 11, 99, 98, 12, 13, 14]]),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        aligned[0], torch.tensor([[0.0, 1.0, 4.0, 5.0, 6.0]]), rtol=0, atol=0
    )
    assert trajectory["input_ids"].shape[-1] == 5


def test_experience_teacher_leaves_pipeline_dummy_alone(tmp_path, monkeypatch):
    store_path = tmp_path / "experiences.json"
    store_path.write_text(json.dumps({"task-1": "use the tool"}), encoding="utf-8")
    monkeypatch.setattr(experience_opd, "load_hf_tokenizer", lambda _: object())
    monkeypatch.setattr(experience_opd, "_user_turn_markers", lambda _: ([10], [12]))
    hook = experience_opd.ExperienceTeacherHook(
        experience_path=str(store_path), tokenizer_path="unused"
    )
    dummy = {
        "input_ids": torch.zeros((1, 1), dtype=torch.long),
        "loss_mask": torch.zeros((1, 1), dtype=torch.long),
        "attention_mask": torch.ones((1, 1), dtype=torch.bool),
    }

    views, positions = hook.pre_compute_logp([dummy])

    torch.testing.assert_close(
        views[0]["input_ids"], dummy["input_ids"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        positions[0], torch.zeros((1, 1), dtype=torch.long), rtol=0, atol=0
    )
