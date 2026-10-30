# SPDX-License-Identifier: Apache-2.0

"""Unit and CLI integration tests for :mod:`areal.dataset.validate`.

Everything here is offline: fixtures are written to ``tmp_path`` and no
downloads, tokenizers, or GPUs are involved.

Only a few cases go through ``subprocess``. Importing ``areal`` pulls in torch
and ray, so each subprocess run costs several seconds; the documented entry
point is therefore exercised end-to-end just often enough to prove that the
module invocation, argv handling, and real process exit codes work. Everything
else drives :func:`areal.dataset.validate.main` in-process.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from areal.dataset import validation
from areal.dataset.validate import main

REPO_ROOT = Path(__file__).resolve().parents[1]

TEXT_RL_ROWS = [
    {"messages": [{"role": "user", "content": "What is 12 * 12?"}], "answer": "144"},
    {"messages": [{"role": "user", "content": "What is 3 + 5?"}], "answer": "8"},
    {"messages": [{"role": "user", "content": "What is 7 - 2?"}], "answer": "5"},
]

# A saved dataset has one Arrow type per column, so masks here are all integers.
# Boolean masks are valid input and are covered by the unit tests below, but
# they cannot be mixed with integers inside a single saved table.
SFT_ROWS = [
    {"input_ids": [11, 12, 13], "loss_mask": [0, 1, 1]},
    {"input_ids": [21, 22], "loss_mask": [1, 0]},
    {"input_ids": [31], "loss_mask": [1]},
]


# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def save_dataset_dir(path: Path, rows) -> Path:
    from datasets import Dataset

    Dataset.from_list(rows).save_to_disk(str(path))
    return path


def save_dataset_dict(path: Path, splits) -> Path:
    from datasets import Dataset, DatasetDict

    DatasetDict(
        {name: Dataset.from_list(rows) for name, rows in splits.items()}
    ).save_to_disk(str(path))
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke the module exactly the way the docs tell users to."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "areal.dataset.validate", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )


def run_main(capsys, *args: str):
    """Drive ``main`` in-process and return (exit_code, stdout, stderr)."""
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --------------------------------------------------------------------------- #
# unit: per-record contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "record, expected_code",
    [
        ([1, 2, 3], validation.CODE_NOT_AN_OBJECT),
        ({"answer": "1"}, validation.CODE_MISSING_FIELD),
        ({"messages": []}, validation.CODE_EMPTY_LIST),
        ({"messages": "hi"}, validation.CODE_WRONG_TYPE),
        ({"messages": [{"role": "user", "content": 1}]}, validation.CODE_WRONG_TYPE),
        (
            {"messages": [{"role": "tool", "content": "x"}]},
            validation.CODE_UNSUPPORTED_ROLE,
        ),
        ({"messages": [{"content": "x"}]}, validation.CODE_MISSING_FIELD),
    ],
)
def test_text_rl_rejects_malformed_record(record, expected_code):
    diagnostics = validation.validate_record(record, validation.TEXT_RL, "line 1")

    assert [d.code for d in diagnostics] == [expected_code]
    assert diagnostics[0].location == "line 1"


@pytest.mark.parametrize(
    "record, expected_code",
    [
        ({"input_ids": [1, 2], "loss_mask": [1]}, validation.CODE_LENGTH_MISMATCH),
        ({"input_ids": [-1], "loss_mask": [1]}, validation.CODE_NEGATIVE_VALUE),
        ({"input_ids": [True], "loss_mask": [1]}, validation.CODE_NOT_INTEGER),
        ({"input_ids": [1], "loss_mask": [2]}, validation.CODE_NON_BINARY_MASK),
        (
            {"input_ids": [1, 2], "loss_mask": [0, 0]},
            validation.CODE_NO_SUPERVISED_TOKEN,
        ),
        ({"input_ids": [], "loss_mask": []}, validation.CODE_EMPTY_LIST),
        ({"loss_mask": [1]}, validation.CODE_MISSING_FIELD),
    ],
)
def test_tokenized_sft_rejects_malformed_record(record, expected_code):
    diagnostics = validation.validate_record(record, validation.TOKENIZED_SFT, "row 3")

    assert expected_code in [d.code for d in diagnostics]
    assert all(d.location == "row 3" for d in diagnostics)


def test_tokenized_sft_rejects_over_length_record():
    record = {"input_ids": [1, 2, 3], "loss_mask": [1, 1, 1]}

    diagnostics = validation.validate_record(
        record, validation.TOKENIZED_SFT, "row 0", max_length=2
    )

    assert validation.CODE_EXCEEDS_MAX_LENGTH in [d.code for d in diagnostics]


def test_require_field_checks_presence_only():
    record = {"messages": [{"role": "user", "content": "x"}], "answer": None}

    diagnostics = validation.validate_record(
        record, validation.TEXT_RL, "line 1", require_fields=("answer",)
    )

    assert diagnostics == []


def test_missing_required_field_is_reported():
    record = {"messages": [{"role": "user", "content": "x"}]}

    diagnostics = validation.validate_record(
        record, validation.TEXT_RL, "line 1", require_fields=("answer",)
    )

    assert [d.code for d in diagnostics] == [validation.CODE_MISSING_REQUIRED_FIELD]
    assert diagnostics[0].field_path == "answer"


def test_tokenized_sft_accepts_boolean_loss_mask():
    record = {"input_ids": [1, 2], "loss_mask": [True, False]}

    diagnostics = validation.validate_record(record, validation.TOKENIZED_SFT, "row 0")

    assert diagnostics == []


def test_all_false_boolean_mask_has_no_supervised_token():
    record = {"input_ids": [1, 2], "loss_mask": [False, False]}

    diagnostics = validation.validate_record(record, validation.TOKENIZED_SFT, "row 0")

    assert [d.code for d in diagnostics] == [validation.CODE_NO_SUPERVISED_TOKEN]


def test_record_with_several_errors_counts_as_one_invalid():
    session = validation.ValidationSession(
        profile=validation.TEXT_RL, source="<memory>"
    )

    session.observe({"messages": [{"role": "nope", "content": 5}]}, "line 1")

    assert (session.report.checked, session.report.valid, session.report.invalid) == (
        1,
        0,
        1,
    )
    assert len(session.report.diagnostics) >= 2


# --------------------------------------------------------------------------- #
# CLI behaviour, driven in-process
# --------------------------------------------------------------------------- #


def test_valid_text_rl_jsonl_reports_three_valid_records(tmp_path, capsys):
    data = write_jsonl(tmp_path / "rl.jsonl", TEXT_RL_ROWS)

    code, out, _ = run_main(
        capsys, str(data), "--profile", "text-rl", "--require-field", "answer"
    )

    assert code == 0
    assert "checked=3" in out
    assert "valid=3" in out
    assert "invalid=0" in out


def test_valid_tokenized_sft_dataset_reports_three_valid_records(tmp_path, capsys):
    data = save_dataset_dir(tmp_path / "sft", SFT_ROWS)

    code, out, _ = run_main(
        capsys, str(data), "--profile", "tokenized-sft", "--max-length", "8"
    )

    assert code == 0
    assert "checked=3 valid=3 invalid=0" in out


def test_dataset_dict_scans_only_the_requested_split(tmp_path, capsys):
    data = save_dataset_dict(
        tmp_path / "dict", {"train": TEXT_RL_ROWS[:2], "validation": TEXT_RL_ROWS[2:]}
    )

    code, out, _ = run_main(
        capsys, str(data), "--profile", "text-rl", "--split", "validation"
    )

    assert code == 0
    assert "checked=1 valid=1 invalid=0" in out


def test_json_report_records_counts_and_diagnostics(tmp_path, capsys):
    data = write_jsonl(tmp_path / "rl.jsonl", TEXT_RL_ROWS)
    report_path = tmp_path / "report.json"

    code, _, _ = run_main(
        capsys, str(data), "--profile", "text-rl", "--report", str(report_path)
    )

    assert code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["checked"] == 3
    assert payload["valid"] == 3
    assert payload["invalid"] == 0
    assert payload["diagnostics"] == []
    assert payload["diagnostics_truncated"] is False


def test_invalid_json_line_is_reported_with_one_based_line_number(tmp_path, capsys):
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps(TEXT_RL_ROWS[0])
        + "\n{ this is not json }\n"
        + json.dumps(TEXT_RL_ROWS[1])
        + "\n",
        encoding="utf-8",
    )

    code, out, _ = run_main(capsys, str(path), "--profile", "text-rl")

    assert code == 1
    assert "checked=3" in out
    assert "invalid=1" in out
    assert "line 2" in out
    assert "[invalid_json]" in out


def test_blank_lines_are_ignored(tmp_path, capsys):
    path = tmp_path / "blank.jsonl"
    path.write_text(
        json.dumps(TEXT_RL_ROWS[0]) + "\n\n   \n" + json.dumps(TEXT_RL_ROWS[1]) + "\n",
        encoding="utf-8",
    )

    code, out, _ = run_main(capsys, str(path), "--profile", "text-rl")

    assert code == 0
    assert "checked=2" in out


def test_empty_file_exits_with_code_one(tmp_path, capsys):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    code, out, _ = run_main(capsys, str(path), "--profile", "text-rl")

    assert code == 1
    assert "checked=0" in out


def test_diagnostic_cap_keeps_counting(tmp_path, capsys):
    path = write_jsonl(tmp_path / "bad.jsonl", [{"messages": []} for _ in range(20)])

    code, out, _ = run_main(
        capsys, str(path), "--profile", "text-rl", "--max-errors", "2"
    )

    assert code == 1
    assert "checked=20" in out
    assert "invalid=20" in out
    assert "messages" in out
    assert "suppressed" in out


def test_repeated_runs_produce_identical_output(tmp_path, capsys):
    path = write_jsonl(
        tmp_path / "mixed.jsonl", [{"messages": []}, TEXT_RL_ROWS[0], {"messages": "x"}]
    )

    first = run_main(capsys, str(path), "--profile", "text-rl")
    second = run_main(capsys, str(path), "--profile", "text-rl")

    assert first == second


def test_input_file_is_not_modified(tmp_path, capsys):
    path = write_jsonl(tmp_path / "rl.jsonl", TEXT_RL_ROWS)
    before = path.read_bytes()

    run_main(capsys, str(path), "--profile", "text-rl")

    assert path.read_bytes() == before


# --------------------------------------------------------------------------- #
# CLI usage and input failures (exit code 2)
# --------------------------------------------------------------------------- #


def test_missing_path_exits_with_code_two(tmp_path, capsys):
    code, _, err = run_main(
        capsys, str(tmp_path / "nope.jsonl"), "--profile", "text-rl"
    )

    assert code == 2
    assert "no such file" in err


def test_dataset_dict_without_split_exits_with_code_two(tmp_path, capsys):
    data = save_dataset_dict(tmp_path / "dict", {"train": TEXT_RL_ROWS})

    code, _, err = run_main(capsys, str(data), "--profile", "text-rl")

    assert code == 2
    assert "DatasetDict" in err


def test_unknown_split_exits_with_code_two(tmp_path, capsys):
    data = save_dataset_dict(
        tmp_path / "dict", {"train": TEXT_RL_ROWS, "validation": TEXT_RL_ROWS}
    )

    code, _, err = run_main(
        capsys, str(data), "--profile", "text-rl", "--split", "test"
    )

    assert code == 2
    assert "available splits" in err


def test_unwritable_report_target_exits_with_code_two(tmp_path, capsys):
    data = write_jsonl(tmp_path / "rl.jsonl", TEXT_RL_ROWS)
    bad_target = tmp_path / "missing-dir" / "report.json"

    code, _, err = run_main(
        capsys, str(data), "--profile", "text-rl", "--report", str(bad_target)
    )

    assert code == 2
    assert "cannot write report" in err


@pytest.mark.parametrize(
    "extra_args",
    [
        ("--max-length", "8"),  # tokenized-sft only
        ("--max-errors", "-1"),  # must be non-negative
    ],
)
def test_unsupported_option_combination_exits_with_code_two(tmp_path, extra_args):
    data = write_jsonl(tmp_path / "rl.jsonl", TEXT_RL_ROWS)

    with pytest.raises(SystemExit) as excinfo:
        main([str(data), "--profile", "text-rl", *extra_args])

    assert excinfo.value.code == 2


def test_non_positive_max_length_exits_with_code_two(tmp_path):
    data = save_dataset_dir(tmp_path / "sft", SFT_ROWS)

    with pytest.raises(SystemExit) as excinfo:
        main([str(data), "--profile", "tokenized-sft", "--max-length", "0"])

    assert excinfo.value.code == 2


def test_help_runs_without_touching_data(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    assert "--profile" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the documented entry point, end-to-end in a real process
# --------------------------------------------------------------------------- #


def test_module_entrypoint_reports_valid_file(tmp_path):
    data = write_jsonl(tmp_path / "rl.jsonl", TEXT_RL_ROWS)

    result = run_cli(str(data), "--profile", "text-rl")

    assert result.returncode == 0, result.stderr
    assert "checked=3 valid=3 invalid=0" in result.stdout


def test_module_entrypoint_exits_one_for_invalid_record(tmp_path):
    data = write_jsonl(tmp_path / "bad.jsonl", [{"messages": []}])

    result = run_cli(str(data), "--profile", "text-rl")

    assert result.returncode == 1
    assert "invalid=1" in result.stdout


def test_module_entrypoint_exits_two_for_missing_input(tmp_path):
    result = run_cli(str(tmp_path / "absent.jsonl"), "--profile", "text-rl")

    assert result.returncode == 2
    assert "no such file" in result.stderr


def test_module_entrypoint_help_exits_zero():
    result = run_cli("--help")

    assert result.returncode == 0
    assert "--require-field" in result.stdout
