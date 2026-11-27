import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

REPO_ROOT = Path(__file__).resolve().parents[1]


def _message_record(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "answer": answer,
        "source": "synthetic",
    }


def _token_record(input_ids: list[int], loss_mask: list[int]) -> dict:
    return {"input_ids": input_ids, "loss_mask": loss_mask}


def _run_validator(*args: Path | str):
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["HF_DATASETS_OFFLINE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "areal.dataset.validate", *map(str, args)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _output_report(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stdout)


def _hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
            digest.update(file_path.read_bytes())
    return digest.hexdigest()


def test_real_module_help_succeeds_on_cpu(tmp_path: Path):
    result = _run_validator(str(tmp_path), "--help")

    assert result.returncode == 0
    assert "--profile {text-rl,tokenized-sft}" in result.stdout
    assert result.stderr == ""


def test_valid_text_rl_jsonl_writes_report_and_preserves_input(tmp_path: Path):
    records = [
        _message_record("1 + 1 = ?", "2"),
        _message_record("2 + 2 = ?", "4"),
        _message_record("3 + 3 = ?", "6"),
    ]
    input_path = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.json"
    _write_jsonl(input_path, records)
    input_before = input_path.read_bytes()

    result = _run_validator(
        input_path,
        "--profile",
        "text-rl",
        "--require-field",
        "answer",
        "--require-field",
        "source",
        "--report",
        report_path,
    )

    assert result.returncode == 0
    payload = _output_report(result)
    assert payload["checked"] == 3
    assert payload["valid"] == 3
    assert payload["invalid"] == 0
    assert payload["diagnostics"] == []
    assert not payload["diagnostics_truncated"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload
    assert input_path.read_bytes() == input_before


def test_jsonl_stream_keeps_physical_line_numbers_and_continues_after_bad_row(
    tmp_path: Path,
):
    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        json.dumps(_message_record("1 + 1 = ?", "2"))
        + "\n\n"
        + "{invalid json\n"
        + "   \n"
        + json.dumps(_message_record("2 + 2 = ?", "4"))
        + "\n",
        encoding="utf-8",
    )

    result = _run_validator(input_path, "--profile", "text-rl")

    assert result.returncode == 1
    payload = _output_report(result)
    assert payload["checked"] == 3
    assert payload["valid"] == 2
    assert payload["invalid"] == 1
    assert payload["diagnostics"][0]["location"] == {"kind": "line", "number": 3}
    assert payload["diagnostics"][0]["field"] == "record"
    assert payload["diagnostics"][0]["code"] == "invalid_json"


@pytest.mark.parametrize("content", ["", "\n   \n\n"])
def test_empty_jsonl_returns_exit_one(tmp_path: Path, content: str):
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text(content, encoding="utf-8")

    result = _run_validator(input_path, "--profile", "text-rl")

    assert result.returncode == 1
    assert "the selected input contains no records" in result.stderr
    payload = _output_report(result)
    assert payload["checked"] == 0
    assert payload["valid"] == 0
    assert payload["invalid"] == 0


def test_json_array_line_is_invalid_but_scan_continues(tmp_path: Path):
    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        "[]\n" + json.dumps(_message_record("1 + 1 = ?", "2")) + "\n",
        encoding="utf-8",
    )

    result = _run_validator(input_path, "--profile", "text-rl")

    assert result.returncode == 1
    payload = _output_report(result)
    assert payload["checked"] == 2
    assert payload["invalid"] == 1
    assert payload["diagnostics"][0]["location"]["number"] == 1
    assert payload["diagnostics"][0]["field"] == "record"
    assert payload["diagnostics"][0]["code"] == "invalid_type"


def test_jsonl_input_rejects_split(tmp_path: Path):
    input_path = tmp_path / "records.jsonl"
    _write_jsonl(input_path, [_message_record("1 + 1 = ?", "2")])

    result = _run_validator(
        input_path,
        "--profile",
        "text-rl",
        "--split",
        "train",
    )

    assert result.returncode == 2
    assert "--split is only supported for DatasetDict directories" in result.stderr


def test_non_standard_json_constant_and_decode_value_error_continue_scan(
    tmp_path: Path,
):
    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        "NaN\n"
        + "1" * 5000
        + "\n"
        + "[" * 2000
        + "\n"
        + json.dumps(_message_record("1 + 1 = ?", "2"))
        + "\n",
        encoding="utf-8",
    )

    result = _run_validator(input_path, "--profile", "text-rl")

    assert result.returncode == 1
    payload = _output_report(result)
    assert payload["checked"] == 4
    assert payload["valid"] == 1
    assert payload["invalid"] == 3
    assert [
        diagnostic["location"]["number"] for diagnostic in payload["diagnostics"]
    ] == [1, 2, 3]
    assert all(
        diagnostic["code"] == "invalid_json" for diagnostic in payload["diagnostics"]
    )


def test_valid_tokenized_sft_dataset_from_disk(tmp_path: Path):
    dataset_path = tmp_path / "tokenized"
    Dataset.from_list(
        [
            _token_record([1, 2, 3], [0, 1, 1]),
            _token_record([4, 5, 6], [0, 0, 1]),
            _token_record([7, 8, 9], [0, 1, 0]),
        ]
    ).save_to_disk(dataset_path)
    dataset_before = _hash_directory(dataset_path)

    result = _run_validator(dataset_path, "--profile", "tokenized-sft")

    assert result.returncode == 0
    payload = _output_report(result)
    assert payload["checked"] == 3
    assert payload["valid"] == 3
    assert payload["invalid"] == 0
    assert _hash_directory(dataset_path) == dataset_before


def test_dataset_dict_validates_only_selected_split(tmp_path: Path):
    dataset_path = tmp_path / "dict"
    train = Dataset.from_list(
        [
            _message_record("1 + 1 = ?", "2"),
            _message_record("2 + 2 = ?", "4"),
        ]
    )
    validation = Dataset.from_list([{"messages": []}])
    DatasetDict({"train": train, "validation": validation}).save_to_disk(dataset_path)

    train_result = _run_validator(
        dataset_path,
        "--profile",
        "text-rl",
        "--split",
        "train",
    )
    validation_result = _run_validator(
        dataset_path,
        "--profile",
        "text-rl",
        "--split",
        "validation",
    )

    assert train_result.returncode == 0
    train_payload = _output_report(train_result)
    assert train_payload["checked"] == 2
    assert train_payload["valid"] == 2
    assert train_payload["invalid"] == 0

    assert validation_result.returncode == 1
    validation_payload = _output_report(validation_result)
    assert validation_payload["checked"] == 1
    assert validation_payload["invalid"] == 1


@pytest.mark.parametrize(
    "split",
    [None, "missing"],
)
def test_dataset_dict_missing_or_unknown_split_returns_exit_two(
    tmp_path: Path, split: str | None
):
    dataset_path = tmp_path / "dict"
    DatasetDict(
        {
            "train": Dataset.from_list([_message_record("1 + 1 = ?", "2")]),
            "validation": Dataset.from_list([_message_record("2 + 2 = ?", "4")]),
        }
    ).save_to_disk(dataset_path)
    args = [dataset_path, "--profile", "text-rl"]
    if split is not None:
        args.extend(["--split", split])

    result = _run_validator(*args)

    assert result.returncode == 2
    assert "Available splits: ['train', 'validation']" in result.stderr


def test_single_dataset_rejects_split(tmp_path: Path):
    dataset_path = tmp_path / "dataset"
    Dataset.from_list([_message_record("1 + 1 = ?", "2")]).save_to_disk(dataset_path)

    result = _run_validator(
        dataset_path,
        "--profile",
        "text-rl",
        "--split",
        "train",
    )

    assert result.returncode == 2
    assert "--split is only supported for DatasetDict directories" in result.stderr


def test_empty_dataset_returns_exit_one(tmp_path: Path):
    dataset_path = tmp_path / "empty"
    Dataset.from_dict({"messages": []}).save_to_disk(dataset_path, num_shards=1)

    result = _run_validator(dataset_path, "--profile", "text-rl")

    assert result.returncode == 1
    payload = _output_report(result)
    assert payload["checked"] == 0


def test_missing_and_corrupted_inputs_return_actionable_exit_two(tmp_path: Path):
    missing_result = _run_validator(
        tmp_path / "missing.jsonl",
        "--profile",
        "text-rl",
    )
    corrupted_path = tmp_path / "not-a-dataset"
    corrupted_path.mkdir()
    corrupted_result = _run_validator(corrupted_path, "--profile", "text-rl")

    assert missing_result.returncode == 2
    assert "Input path does not exist" in missing_result.stderr
    assert corrupted_result.returncode == 2
    assert "Failed to load local dataset" in corrupted_result.stderr


def test_invalid_utf8_jsonl_returns_exit_two(tmp_path: Path):
    input_path = tmp_path / "invalid-utf8.jsonl"
    input_path.write_bytes(b'{"messages": [{"role": "user", "content": "\xff"}]}\n')

    result = _run_validator(input_path, "--profile", "text-rl")

    assert result.returncode == 2
    assert "error:" in result.stderr


def test_unknown_profile_and_option_return_exit_two(tmp_path: Path):
    input_path = tmp_path / "records.jsonl"
    _write_jsonl(input_path, [_message_record("1 + 1 = ?", "2")])

    unknown_profile = _run_validator(input_path, "--profile", "unknown")
    unknown_option = _run_validator(input_path, "--profile", "text-rl", "--unknown")

    assert unknown_profile.returncode == 2
    assert unknown_option.returncode == 2


@pytest.mark.parametrize(
    "args",
    [
        ["--profile", "text-rl", "--max-length", "2"],
        ["--profile", "tokenized-sft", "--require-field", "answer"],
        ["--profile", "tokenized-sft", "--max-length", "0"],
        ["--profile", "tokenized-sft", "--max-errors", "-1"],
    ],
)
def test_profile_and_value_errors_return_exit_two(tmp_path: Path, args: list[str]):
    input_path = tmp_path / "records.jsonl"
    _write_jsonl(input_path, [_message_record("1 + 1 = ?", "2")])

    result = _run_validator(input_path, *args)

    assert result.returncode == 2


def test_report_parent_is_a_file_returns_exit_two(tmp_path: Path):
    input_path = tmp_path / "records.jsonl"
    _write_jsonl(input_path, [_message_record("1 + 1 = ?", "2")])
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    report_path = blocker / "report.json"

    result = _run_validator(
        input_path,
        "--profile",
        "text-rl",
        "--report",
        report_path,
    )

    assert result.returncode == 2
    assert "error:" in result.stderr


def test_report_same_path_and_hardlink_are_rejected(tmp_path: Path):
    input_path = tmp_path / "records.jsonl"
    alias_path = tmp_path / "alias.jsonl"
    _write_jsonl(input_path, [_message_record("1 + 1 = ?", "2")])
    os.link(input_path, alias_path)
    input_before = input_path.read_bytes()

    same_path = _run_validator(
        input_path,
        "--profile",
        "text-rl",
        "--report",
        input_path,
    )
    alias = _run_validator(
        input_path,
        "--profile",
        "text-rl",
        "--report",
        alias_path,
    )

    assert same_path.returncode == 2
    assert alias.returncode == 2
    assert "Report path must not overwrite the input path" in same_path.stderr
    assert "Report path must not overwrite the input path" in alias.stderr
    assert input_path.read_bytes() == input_before


def test_report_inside_dataset_directory_is_rejected(tmp_path: Path):
    dataset_path = tmp_path / "dataset"
    Dataset.from_list([_message_record("1 + 1 = ?", "2")]).save_to_disk(dataset_path)
    report_path = dataset_path / "report.json"
    dataset_before = _hash_directory(dataset_path)

    result = _run_validator(
        dataset_path,
        "--profile",
        "text-rl",
        "--report",
        report_path,
    )

    assert result.returncode == 2
    assert "Report path must be outside the input dataset directory" in result.stderr
    assert not report_path.exists()
    assert _hash_directory(dataset_path) == dataset_before


def test_report_hardlinked_to_dataset_arrow_file_is_rejected(tmp_path: Path):
    dataset_path = tmp_path / "dataset"
    Dataset.from_list([_message_record("1 + 1 = ?", "2")]).save_to_disk(dataset_path)
    arrow_file = next(dataset_path.rglob("*.arrow"))
    report_path = tmp_path / "report.json"
    os.link(arrow_file, report_path)
    dataset_before = _hash_directory(dataset_path)

    result = _run_validator(
        dataset_path,
        "--profile",
        "text-rl",
        "--report",
        report_path,
    )

    assert result.returncode == 2
    assert "Report path has multiple links" in result.stderr
    assert _hash_directory(dataset_path) == dataset_before


def test_diagnostics_limit_still_scans_all_records(tmp_path: Path):
    input_path = tmp_path / "records.jsonl"
    records = [{"messages": []} for _ in range(20)]
    _write_jsonl(input_path, records)

    result = _run_validator(
        input_path,
        "--profile",
        "text-rl",
        "--max-errors",
        "2",
    )

    assert result.returncode == 1
    payload = _output_report(result)
    assert payload["checked"] == 20
    assert payload["invalid"] == 20
    assert len(payload["diagnostics"]) == 2
    assert payload["diagnostics_truncated"]


def test_repeated_runs_are_deterministic(tmp_path: Path):
    input_path = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.json"
    _write_jsonl(
        input_path,
        [
            _message_record("1 + 1 = ?", "2"),
            {"messages": []},
            _message_record("2 + 2 = ?", "4"),
        ],
    )

    first = _run_validator(
        input_path,
        "--profile",
        "text-rl",
        "--report",
        report_path,
    )
    first_payload = report_path.read_text(encoding="utf-8")
    second = _run_validator(
        input_path,
        "--profile",
        "text-rl",
        "--report",
        report_path,
    )

    assert first.returncode == 1
    assert second.returncode == 1
    assert second.stdout == first.stdout
    assert report_path.read_text(encoding="utf-8") == first_payload
