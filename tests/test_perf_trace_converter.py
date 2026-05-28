from __future__ import annotations

import json
from pathlib import Path

import pytest

from areal.tools.perf_trace_converter import convert_jsonl_to_chrome_trace, main


def _write_jsonl(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for event in events:
            json.dump(event, fout)
            fout.write("\n")


def _extract_metadata(trace_events: list[dict], name: str) -> list[dict]:
    return [
        event
        for event in trace_events
        if event.get("ph") == "M" and event.get("name") == name
    ]


def test_convert_single_file_basic(tmp_path: Path) -> None:
    source = tmp_path / "rank0.jsonl"
    _write_jsonl(
        source,
        [
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 5,
                "tid": 7,
                "args": {"name": "ignored"},
            },
            {
                "name": "compute",
                "ph": "X",
                "pid": 5,
                "tid": 7,
                "ts": 3,
                "args": {"rank": 0, "payload": "a"},
            },
            {
                "name": "io",
                "ph": "X",
                "pid": 5,
                "tid": 8,
                "ts": 2,
                "args": {"rank": "0"},
            },
        ],
    )

    result = convert_jsonl_to_chrome_trace(source)
    assert result["displayTimeUnit"] == "ms"
    events = result["traceEvents"]

    process_meta = _extract_metadata(events, "process_name")
    assert process_meta and process_meta[0]["args"]["name"].startswith(
        "[Rank 0, Process 5]"
    )

    thread_meta = _extract_metadata(events, "thread_name")
    assert {item["args"]["name"] for item in thread_meta} == {
        "[Thread 7]",
        "[Thread 8]",
    }

    payload_events = [event for event in events if event.get("ph") != "M"]
    assert [event["ts"] for event in payload_events] == [2, 3]
    assert {event["pid"] for event in payload_events} == {1}
    assert {event["tid"] for event in payload_events} == {1, 2}


def test_convert_directory_multi_rank_with_shared_ids(tmp_path: Path) -> None:
    dir_path = tmp_path / "traces"
    dir_path.mkdir()

    _write_jsonl(
        dir_path / "rank0.jsonl",
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 1,
                "tid": 3,
                "ts": 5,
                "args": {"rank": 0},
            }
        ],
    )
    _write_jsonl(
        dir_path / "rank1.jsonl",
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 1,
                "tid": 3,
                "ts": 4,
                "args": {"rank": 1},
            }
        ],
    )

    result = convert_jsonl_to_chrome_trace(dir_path)
    events = result["traceEvents"]

    process_meta = _extract_metadata(events, "process_name")
    assert [item["args"]["rank"] for item in process_meta] == [0, 1]

    thread_meta = _extract_metadata(events, "thread_name")
    assert {
        (item["args"]["rank"], item["args"].get("name")) for item in thread_meta
    } == {
        (0, "[Thread 3]"),
        (1, "[Thread 3]"),
    }

    payload_events = [event for event in events if event.get("ph") != "M"]
    assert [event["ts"] for event in payload_events] == [4, 5]
    assert {event["pid"] for event in payload_events} == {1, 2}
    assert {event["tid"] for event in payload_events} == {1, 2}
    by_rank_tid = {event["args"]["rank"]: event["tid"] for event in payload_events}
    assert by_rank_tid == {0: 1, 1: 2}


def test_convert_glob_not_found(tmp_path: Path) -> None:
    pattern = str(tmp_path / "missing" / "*.jsonl")
    with pytest.raises(FileNotFoundError):
        convert_jsonl_to_chrome_trace(pattern)


def test_main_default_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "rank.jsonl"
    _write_jsonl(
        source,
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 2,
                "tid": 4,
                "ts": 1,
                "args": {"rank": 0},
            }
        ],
    )

    monkeypatch.chdir(tmp_path)
    exit_code = main([str(source)])
    assert exit_code == 0

    # With new behavior, output should be in same directory with .json extension
    output_path = tmp_path / "rank.json"
    with output_path.open("r", encoding="utf-8") as fin:
        payload = json.load(fin)
    assert payload["traceEvents"]


def test_main_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "rank.jsonl"
    _write_jsonl(
        source,
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 2,
                "tid": 4,
                "ts": 1,
                "args": {"rank": 0},
            }
        ],
    )

    exit_code = main([str(source), "-"])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["traceEvents"]


def test_main_default_output_directory(tmp_path: Path) -> None:
    """Test that when input is a directory, output goes to <dir>/traces.json"""
    dir_path = tmp_path / "traces"
    dir_path.mkdir()

    _write_jsonl(
        dir_path / "rank0.jsonl",
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "ts": 3,
                "args": {"rank": 0},
            }
        ],
    )

    exit_code = main([str(dir_path)])
    assert exit_code == 0

    # Output should be in the same directory
    output_path = dir_path / "traces.json"
    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as fin:
        payload = json.load(fin)
    assert payload["traceEvents"]


def test_main_default_output_multiple_files(tmp_path: Path) -> None:
    """Test that when input is a glob matching multiple files, output goes to common parent"""
    dir_path = tmp_path / "traces"
    dir_path.mkdir()

    _write_jsonl(
        dir_path / "rank0.jsonl",
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "ts": 3,
                "args": {"rank": 0},
            }
        ],
    )
    _write_jsonl(
        dir_path / "rank1.jsonl",
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "ts": 4,
                "args": {"rank": 1},
            }
        ],
    )

    # Use glob pattern
    pattern = str(dir_path / "*.jsonl")
    exit_code = main([pattern])
    assert exit_code == 0

    # Output should be in the common parent directory
    output_path = dir_path / "traces.json"
    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as fin:
        payload = json.load(fin)
    assert len([e for e in payload["traceEvents"] if e.get("ph") != "M"]) == 2
