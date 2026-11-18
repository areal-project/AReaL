# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

import pytest
from datasets import Dataset, DatasetDict

from areal.dataset.validate import Problem, main, validate_record


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "areal.dataset.validate", *map(str, args)],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("record", "field", "code"),
    [
        ([], "$", "expected_object"),
        ({}, "messages", "expected_nonempty_list"),
        ({"messages": []}, "messages", "expected_nonempty_list"),
        ({"messages": ["hello"]}, "messages[0]", "expected_object"),
        (
            {"messages": [{"role": "tool", "content": "ok"}]},
            "messages[0].role",
            "invalid_role",
        ),
        (
            {"messages": [{"role": "user", "content": 1}]},
            "messages[0].content",
            "expected_string",
        ),
        ({"messages": [{"role": "user", "content": ""}]}, "answer", "missing_field"),
    ],
)
def test_text_rule_identifies_invalid_field(record, field, code):
    """Each unsupported input produces the corresponding field and code."""
    assert Problem(field, code) in list(validate_record(record, "text-rl", ["answer"]))


@pytest.mark.parametrize(
    ("ids", "mask", "field", "code"),
    [
        ([], [], "input_ids", "expected_nonempty_list"),
        ([1], [], "loss_mask", "expected_nonempty_list"),
        ([1, 2], [1], "loss_mask", "length_mismatch"),
        ([-1], [1], "input_ids[0]", "invalid_token_id"),
        ([True], [1], "input_ids[0]", "invalid_token_id"),
        ([1], [2], "loss_mask[0]", "invalid_mask_value"),
        ([1], [False], "loss_mask", "no_supervised_token"),
        ([1, 2, 3], [0, 1, 1], "input_ids", "over_max_length"),
    ],
)
def test_sft_rule_identifies_invalid_field(ids, mask, field, code):
    """Check each token/mask constraint independently of the CLI parser."""
    record = {"input_ids": ids, "loss_mask": mask}
    assert Problem(field, code) in list(
        validate_record(record, "tokenized-sft", max_length=2)
    )


def test_jsonl_cli_validates_three_rows_and_preserves_input(tmp_path):
    """Exercise all supported roles and presence-only checks in the real process."""
    source = tmp_path / "math.jsonl"
    report = tmp_path / "report.json"
    rows = [
        {"messages": [{"role": role, "content": ""}], "answer": None, "tag": 0}
        for role in ("system", "user", "assistant")
    ]
    original = "\n".join(map(json.dumps, rows)).encode()
    source.write_bytes(original)
    result = cli(
        source,
        "--profile",
        "text-rl",
        "--require-field",
        "answer",
        "--require-field",
        "tag",
        "--report",
        report,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text()) == {
        "checked": 3,
        "valid": 3,
        "invalid": 0,
        "truncated": False,
        "diagnostics": [],
    }
    assert source.read_bytes() == original


def test_saved_sft_and_selected_split(tmp_path):
    """Read a real Arrow dataset and select just the requested DatasetDict split."""
    source, splits = tmp_path / "data", tmp_path / "splits"
    good = Dataset.from_list(
        [{"input_ids": [n, 5], "loss_mask": [False, True]} for n in range(3)]
    )
    bad = Dataset.from_list([{"input_ids": [-1], "loss_mask": [True]}])
    good.save_to_disk(str(source))
    DatasetDict(train=good, validation=bad).save_to_disk(str(splits))
    report = tmp_path / "report.json"
    original = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = cli(source, "--profile", "tokenized-sft", "--max-length", "2")
    assert result.returncode == 0, result.stderr
    assert "checked=3 valid=3 invalid=0" in result.stdout
    args = [str(splits), "--profile", "tokenized-sft", "--report", str(report)]
    assert main([*args, "--split", "validation"]) == 1
    output = json.loads(report.read_text())
    assert (output["checked"], output["valid"], output["invalid"]) == (1, 0, 1)
    assert output["diagnostics"] == [
        {"row": 0, "field": "input_ids[0]", "code": "invalid_token_id"}
    ]
    for extra in ([], ["--split", "unknown"]):
        with pytest.raises(SystemExit) as error:
            main([*args, *extra])
        assert error.value.code == 2
    assert {
        p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and p != report
    } == original


def test_cli_counts_records_after_bad_json_and_truncation(tmp_path):
    """Ignore blanks, continue after bad lines, count multi-error records just once."""
    source, report = tmp_path / "data.jsonl", tmp_path / "out.json"
    bad = json.dumps({"messages": [{"role": "other", "content": 9}]})
    source.write_text("\n{\n" + "\n".join([bad] * 19))
    args = [
        str(source),
        "--profile",
        "text-rl",
        "--max-errors",
        "2",
        "--report",
        str(report),
    ]
    result = cli(*args)
    assert result.returncode == 1, result.stderr
    expected = {
        "checked": 20,
        "valid": 0,
        "invalid": 20,
        "truncated": True,
        "diagnostics": [
            {"line": 2, "field": "$", "code": "invalid_json"},
            {"line": 3, "field": "messages[0].role", "code": "invalid_role"},
        ],
    }
    assert json.loads(report.read_text()) == expected
    assert main(args) == 1
    assert json.loads(report.read_text()) == expected


def test_cli_help_empty_and_io_errors(tmp_path, capsys):
    """Observe help, empty input, failed loading, bad options and report writes."""
    result = cli("--help")
    assert result.returncode == 0 and "--profile" in result.stdout
    source = tmp_path / "empty.jsonl"
    source.write_text("\n")
    args = [str(source), "--profile", "text-rl"]
    assert main(args) == 1
    missing = cli(tmp_path / "absent", "--profile", "text-rl")
    assert missing.returncode == 2 and "absent" in missing.stderr
    for extra, keyword in [
        (["--report", str(tmp_path)], str(tmp_path)),
        (["--report", str(source)], "report"),
        (["--max-length", "3"], "--max-length"),
        (["--unknown"], "--unknown"),
    ]:
        with pytest.raises(SystemExit) as error:
            main([*args, *extra])
        assert error.value.code == 2 and keyword in capsys.readouterr().err
    assert source.read_bytes() == b"\n"
