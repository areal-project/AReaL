# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from examples.prefix_replay.prepare_routed_replay import (
    prepare_routed_replay_dataset,
)


def test_prepare_routed_replay_dataset_assigns_source_teachers(tmp_path):
    """Rows from each source retain content and receive only their source route."""
    game_path = tmp_path / "game.jsonl"
    student_path = tmp_path / "student.jsonl"
    output_path = tmp_path / "routed.jsonl"
    game_path.write_text(
        json.dumps(
            {
                "instance_id": "game-1",
                "messages": [{"role": "user", "content": "game"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    student_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "student-1",
                        "messages": [{"role": "user", "content": "student 1"}],
                    }
                ),
                json.dumps(
                    {
                        "instance_id": "student-2",
                        "messages": [{"role": "user", "content": "student 2"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = prepare_routed_replay_dataset(
        [("game", game_path), ("student_init", student_path)],
        output_path,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert report["route_counts"] == {"game": 1, "student_init": 2}
    assert {(row["instance_id"], row["replay_teacher"]) for row in rows} == {
        ("game-1", "game"),
        ("student-1", "student_init"),
        ("student-2", "student_init"),
    }


def test_prepare_routed_replay_dataset_rejects_conflicting_source_route(tmp_path):
    """Existing route metadata cannot silently send a row to another teacher."""
    game_path = tmp_path / "game.jsonl"
    output_path = tmp_path / "routed.jsonl"
    game_path.write_text(
        json.dumps({"replay_teacher": "student_init", "messages": []}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicts with source route"):
        prepare_routed_replay_dataset(
            [("game", game_path)],
            output_path,
            workers=2,
            chunk_records=1,
        )

    assert not output_path.exists()


def test_prepare_routed_replay_dataset_rejects_source_output_alias(tmp_path):
    """The atomic publish cannot overwrite a source replay dataset."""
    game_path = tmp_path / "game.jsonl"
    game_path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "game"}]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot overwrite a source"):
        prepare_routed_replay_dataset([("game", game_path)], game_path)


def test_prepare_routed_replay_dataset_parallel_matches_serial_output(tmp_path):
    """Bounded process workers preserve deterministic interleaving and bytes."""
    game_path = tmp_path / "game.jsonl"
    student_path = tmp_path / "student.jsonl"
    serial_path = tmp_path / "serial.jsonl"
    parallel_path = tmp_path / "parallel.jsonl"
    game_path.write_text(
        "".join(
            json.dumps({"instance_id": f"game-{index}", "text": "游戏"}) + "\n"
            for index in range(7)
        ),
        encoding="utf-8",
    )
    student_path.write_text(
        "".join(
            json.dumps({"instance_id": f"student-{index}", "text": "code"}) + "\n"
            for index in range(13)
        ),
        encoding="utf-8",
    )
    sources = [("game", game_path), ("student_init", student_path)]

    serial_report = prepare_routed_replay_dataset(
        sources,
        serial_path,
        workers=1,
        chunk_records=3,
    )
    parallel_report = prepare_routed_replay_dataset(
        sources,
        parallel_path,
        workers=2,
        chunk_records=2,
    )

    assert parallel_path.read_bytes() == serial_path.read_bytes()
    assert serial_report["route_counts"] == parallel_report["route_counts"]
    assert serial_report["total"] == parallel_report["total"] == 20
