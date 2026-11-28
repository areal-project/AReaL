# SPDX-License-Identifier: Apache-2.0

"""Offline unit tests and CLI integration tests for ``areal.dataset.validate``.

The validator module is loaded standalone from its file path (only stdlib at
import time), so the core logic tests run in any CPU environment. The
``python -m areal.dataset.validate`` subprocess tests additionally require the
full AReaL import chain and are skipped gracefully when it is unavailable.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_validate_module():
    path = _ROOT / "areal" / "dataset" / "validate.py"
    spec = importlib.util.spec_from_file_location(
        "areal_dataset_validate_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate = _load_validate_module()

VALID_TEXT_RL_RECORDS = [
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 2 + 2? Put your final answer within \\boxed{}.",
            }
        ],
        "answer": "4",
    },
    {
        "messages": [
            {"role": "system", "content": "You are a careful math tutor."},
            {
                "role": "user",
                "content": (
                    "Nina has 3 apples and buys 4 more. How many apples does "
                    "she have? \\boxed{}"
                ),
            },
        ],
        "answer": "7",
    },
    {
        "messages": [{"role": "user", "content": "Solve: 5 * 3 - 4 = ? \\boxed{}"}],
        "answer": "11",
        "index": 2,
    },
]

VALID_TOKENIZED_SFT_RECORDS = [
    {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]},
    {"input_ids": [4, 5], "loss_mask": [1, 0]},
    {"input_ids": [6, 7, 8, 9], "loss_mask": [0, 0, 1, 1]},
]


def write_jsonl(tmp_path, entries, name="data.jsonl"):
    """Write one JSONL file; dict entries are serialized, str entries verbatim."""
    lines = [
        entry if isinstance(entry, str) else json.dumps(entry) for entry in entries
    ]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    return str(path)


def run_main(argv):
    """Call the CLI main, normalizing argparse SystemExit to an exit code."""
    try:
        return validate.main(argv)
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        return exc.code


def read_report(report_path):
    return json.loads(Path(report_path).read_text(encoding="utf-8"))


def _cli_module_importable():
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import areal.dataset.validate"],
            cwd=str(_ROOT),
            capture_output=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


_CLI_IMPORTABLE = _cli_module_importable()
requires_full_cli = pytest.mark.skipif(
    not _CLI_IMPORTABLE,
    reason=(
        "`python -m areal.dataset.validate` needs the full AReaL import "
        "chain, unavailable in this environment"
    ),
)


@pytest.fixture()
def datasets_lib():
    return pytest.importorskip("datasets")


def save_to_disk_dir(datasets_lib, records, tmp_path, name="saved_dataset"):
    """Save a plain Hugging Face ``Dataset`` and return its directory path."""
    path = tmp_path / name
    datasets_lib.Dataset.from_list(records).save_to_disk(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Core validation logic (no CLI, no datasets package required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("record", VALID_TEXT_RL_RECORDS)
def test_validate_record_text_rl_valid_returns_no_diagnostics(record):
    """Text-RL prompt records satisfy the profile contract."""
    assert validate.validate_record(record, "text-rl", require_fields=("answer",)) == []


@pytest.mark.parametrize("record", VALID_TOKENIZED_SFT_RECORDS)
def test_validate_record_tokenized_sft_valid_returns_no_diagnostics(record):
    """Tokenized records with aligned masks satisfy the SFT profile contract."""
    assert validate.validate_record(record, "tokenized-sft") == []


def test_validate_record_unknown_profile_raises_usage_error():
    """Only the two documented profiles are supported."""
    with pytest.raises(validate.DatasetValidationUsageError):
        validate.validate_record({}, "dpo")


def test_scan_records_counts_one_invalid_record_with_several_errors():
    """Several violations in one record count once but keep all diagnostics."""
    record = {"messages": [{"role": "tool", "content": 5}, "oops"]}
    result = validate.scan_records(
        [("line 1", record, None)], "text-rl", require_fields=("answer",)
    )
    assert (result.checked, result.valid, result.invalid) == (1, 0, 1)
    codes = [d.code for d in result.diagnostics]
    assert "role_invalid" in codes
    assert "content_not_string" in codes
    assert "message_not_object" in codes
    assert "required_field_missing" in codes


def test_scan_records_reports_truncation_beyond_max_errors():
    """Diagnostics are capped, but every record is still counted."""
    items = [(f"line {i}", {"messages": []}, None) for i in range(1, 4)]
    result = validate.scan_records(items, "text-rl", max_errors=2)
    assert (result.checked, result.valid, result.invalid) == (3, 0, 3)
    assert len(result.diagnostics) == 2
    assert result.truncated


# ---------------------------------------------------------------------------
# Valid inputs: exit code 0 and exact counts
# ---------------------------------------------------------------------------


def test_text_rl_valid_jsonl_exits_zero(tmp_path, capsys):
    """A three-row valid text-RL JSONL fixture reports checked=3/valid=3."""
    path = write_jsonl(tmp_path, VALID_TEXT_RL_RECORDS)
    report_path = tmp_path / "report.json"

    code = run_main(
        [
            path,
            "--profile",
            "text-rl",
            "--require-field",
            "answer",
            "--report",
            str(report_path),
        ]
    )

    assert code == 0
    report = read_report(report_path)
    assert (report["checked"], report["valid"], report["invalid"]) == (3, 3, 0)
    assert report["profile"] == "text-rl"
    assert report["split"] is None
    assert report["truncated"] is False
    assert report["diagnostics"] == []
    assert "checked=3 valid=3 invalid=0" in capsys.readouterr().out


def test_tokenized_sft_valid_saved_dataset_exits_zero(datasets_lib, tmp_path):
    """A three-row valid tokenized-SFT saved dataset reports checked=3/valid=3."""
    path = save_to_disk_dir(datasets_lib, VALID_TOKENIZED_SFT_RECORDS, tmp_path)
    report_path = tmp_path / "report.json"

    code = run_main([path, "--profile", "tokenized-sft", "--report", str(report_path)])

    assert code == 0
    report = read_report(report_path)
    assert (report["checked"], report["valid"], report["invalid"]) == (3, 3, 0)
    assert report["diagnostics"] == []


def test_dataset_dict_scans_only_requested_split(datasets_lib, tmp_path):
    """Only the split named by --split is checked, and the report records it."""
    train = datasets_lib.Dataset.from_list([{"messages": [], "answer": "1"}])
    validation = datasets_lib.Dataset.from_list(VALID_TEXT_RL_RECORDS[:2])
    path = tmp_path / "mixed_dataset"
    datasets_lib.DatasetDict({"train": train, "validation": validation}).save_to_disk(
        str(path)
    )
    report_path = tmp_path / "report.json"

    code = run_main(
        [
            str(path),
            "--profile",
            "text-rl",
            "--split",
            "validation",
            "--require-field",
            "answer",
            "--report",
            str(report_path),
        ]
    )

    assert code == 0
    report = read_report(report_path)
    assert (report["checked"], report["valid"], report["invalid"]) == (2, 2, 0)
    assert report["split"] == "validation"


def test_dataset_dict_without_split_is_a_usage_error(datasets_lib, tmp_path, capsys):
    """A DatasetDict input requires an explicit valid --split."""
    train = datasets_lib.Dataset.from_list(VALID_TEXT_RL_RECORDS)
    path = tmp_path / "dict_dataset"
    datasets_lib.DatasetDict({"train": train, "test": train}).save_to_disk(str(path))

    code = run_main([str(path)])

    assert code == 2
    assert "--split" in capsys.readouterr().err


def test_dataset_dict_unknown_split_is_a_usage_error(datasets_lib, tmp_path, capsys):
    """Unknown split names are rejected with the available splits listed."""
    train = datasets_lib.Dataset.from_list(VALID_TEXT_RL_RECORDS)
    path = tmp_path / "dict_dataset"
    datasets_lib.DatasetDict({"train": train}).save_to_disk(str(path))

    code = run_main([str(path), "--split", "validation"])

    assert code == 2
    err = capsys.readouterr().err
    assert "'validation'" in err
    assert "'train'" in err


# ---------------------------------------------------------------------------
# Invalid inputs: exit code 1 with row/field-identifying diagnostics
# ---------------------------------------------------------------------------

TEXT_RL_NEGATIVES = [
    ({"answer": "1"}, "messages_missing", "messages"),
    ({"messages": [], "answer": "1"}, "messages_empty", "messages"),
    ({"messages": "hello"}, "messages_not_list", "messages"),
    ({"messages": ["not-a-mapping"]}, "message_not_object", "messages[0]"),
    ({"messages": [{"content": "hi"}]}, "role_missing", "messages[0].role"),
    (
        {"messages": [{"role": "tool", "content": "hi"}]},
        "role_invalid",
        "messages[0].role",
    ),
    ({"messages": [{"role": "user"}]}, "content_missing", "messages[0].content"),
    (
        {"messages": [{"role": "user", "content": 5}]},
        "content_not_string",
        "messages[0].content",
    ),
    ({}, "messages_missing", "messages"),
]


@pytest.mark.parametrize(("record", "code", "field_path"), TEXT_RL_NEGATIVES)
def test_text_rl_negative_records_exit_one(record, code, field_path, tmp_path):
    """Each text-RL violation yields exit 1 and a diagnostic for row and field."""
    path = write_jsonl(tmp_path, [record])
    report_path = tmp_path / "report.json"

    exit_code = run_main(
        [path, "--require-field", "answer", "--report", str(report_path)]
    )

    assert exit_code == 1
    report = read_report(report_path)
    assert (report["checked"], report["valid"], report["invalid"]) == (1, 0, 1)
    diagnostics = [d for d in report["diagnostics"] if d["code"] == code]
    assert diagnostics, f"expected diagnostic {code!r} in {report['diagnostics']}"
    assert diagnostics[0]["field"] == field_path
    assert diagnostics[0]["location"] == "line 1"


def test_text_rl_missing_required_reward_field_exits_one(tmp_path):
    """--require-field checks presence only; the missing field is reported."""
    record = {"messages": [{"role": "user", "content": "hi"}]}
    path = write_jsonl(tmp_path, [record])
    report_path = tmp_path / "report.json"

    exit_code = run_main(
        [path, "--require-field", "answer", "--report", str(report_path)]
    )

    assert exit_code == 1
    diagnostics = read_report(report_path)["diagnostics"]
    assert [d["code"] for d in diagnostics] == ["required_field_missing"]
    assert diagnostics[0]["field"] == "answer"


def test_non_object_jsonl_record_is_reported(tmp_path):
    """A JSON line holding an array is one invalid record, not a crash."""
    path = write_jsonl(tmp_path, ["[1, 2]"])
    report_path = tmp_path / "report.json"

    exit_code = run_main([path, "--report", str(report_path)])

    assert exit_code == 1
    report = read_report(report_path)
    assert (report["checked"], report["invalid"]) == (1, 1)
    assert report["diagnostics"][0]["code"] == "record_not_object"
    assert report["diagnostics"][0]["location"] == "line 1"


def test_invalid_json_line_keeps_scanning_and_identifies_its_line(tmp_path):
    """A broken line counts as one invalid record with its 1-based line number."""
    path = write_jsonl(
        tmp_path,
        [VALID_TEXT_RL_RECORDS[0], "{ not json ]]", VALID_TEXT_RL_RECORDS[1]],
    )
    report_path = tmp_path / "report.json"

    exit_code = run_main([path, "--report", str(report_path)])

    assert exit_code == 1
    report = read_report(report_path)
    assert (report["checked"], report["valid"], report["invalid"]) == (3, 2, 1)
    diagnostic = report["diagnostics"][0]
    assert diagnostic["code"] == "json_invalid_line"
    assert diagnostic["location"] == "line 2"


def test_blank_lines_are_skipped(tmp_path):
    """Blank or whitespace-only JSONL lines contribute no checked record."""
    path = write_jsonl(
        tmp_path,
        [VALID_TEXT_RL_RECORDS[0], "", VALID_TEXT_RL_RECORDS[1], "   "],
    )
    report_path = tmp_path / "report.json"

    exit_code = run_main([path, "--report", str(report_path)])

    assert exit_code == 0
    assert read_report(report_path)["checked"] == 2


SFT_NEGATIVES = [
    ({"loss_mask": [1]}, "input_ids_missing", "input_ids"),
    ({"input_ids": [1]}, "loss_mask_missing", "loss_mask"),
    ({"input_ids": [], "loss_mask": []}, "input_ids_empty", "input_ids"),
    ({"input_ids": [], "loss_mask": []}, "loss_mask_empty", "loss_mask"),
    ({"input_ids": "abc", "loss_mask": [1]}, "input_ids_not_list", "input_ids"),
    ({"input_ids": [1], "loss_mask": "ab"}, "loss_mask_not_list", "loss_mask"),
    ({"input_ids": [1, 2], "loss_mask": [1]}, "length_mismatch", "loss_mask"),
    ({"input_ids": [1, -2], "loss_mask": [1, 1]}, "token_id_negative", "input_ids[1]"),
    ({"input_ids": [1, True], "loss_mask": [1, 1]}, "token_id_bool", "input_ids[1]"),
    ({"input_ids": [1.5], "loss_mask": [1]}, "token_id_not_int", "input_ids[0]"),
    ({"input_ids": [1, 2], "loss_mask": [1, 2]}, "mask_not_binary", "loss_mask[1]"),
    ({"input_ids": [1, 2], "loss_mask": [1, "1"]}, "mask_not_binary", "loss_mask[1]"),
    ({"input_ids": [1, 2], "loss_mask": [0, 0]}, "no_supervised_tokens", "loss_mask"),
]


@pytest.mark.parametrize(("record", "code", "field_path"), SFT_NEGATIVES)
def test_tokenized_sft_negative_records_exit_one(record, code, field_path, tmp_path):
    """Each tokenized-SFT violation yields exit 1 with row and field info."""
    path = write_jsonl(tmp_path, [record])
    report_path = tmp_path / "report.json"

    exit_code = run_main(
        [path, "--profile", "tokenized-sft", "--report", str(report_path)]
    )

    assert exit_code == 1
    report = read_report(report_path)
    assert (report["checked"], report["valid"], report["invalid"]) == (1, 0, 1)
    diagnostics = [d for d in report["diagnostics"] if d["code"] == code]
    assert diagnostics, f"expected diagnostic {code!r} in {report['diagnostics']}"
    assert diagnostics[0]["field"] == field_path


def test_tokenized_sft_over_length_record_exits_one(tmp_path):
    """--max-length applies to input_ids length for the SFT profile."""
    path = write_jsonl(
        tmp_path, [{"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1]}]
    )
    report_path = tmp_path / "report.json"

    exit_code = run_main(
        [
            path,
            "--profile",
            "tokenized-sft",
            "--max-length",
            "3",
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 1
    report = read_report(report_path)
    assert report["invalid"] == 1
    assert [d["code"] for d in report["diagnostics"]] == ["exceeds_max_length"]


def test_tokenized_sft_boolean_mask_is_accepted(tmp_path):
    """Boolean masks are binary values and one True marks supervision."""
    path = write_jsonl(tmp_path, [{"input_ids": [1, 2], "loss_mask": [True, False]}])
    report_path = tmp_path / "report.json"

    exit_code = run_main(
        [path, "--profile", "tokenized-sft", "--report", str(report_path)]
    )

    assert exit_code == 0
    assert read_report(report_path)["valid"] == 1


def test_saved_dataset_negative_row_is_located_by_row_number(datasets_lib, tmp_path):
    """Dataset diagnostics carry a 0-based row location and field path."""
    path = save_to_disk_dir(
        datasets_lib, [{"input_ids": [1, 2], "loss_mask": [1]}], tmp_path
    )
    report_path = tmp_path / "report.json"

    exit_code = run_main(
        [path, "--profile", "tokenized-sft", "--report", str(report_path)]
    )

    assert exit_code == 1
    report = read_report(report_path)
    assert report["input"] == path
    assert report["diagnostics"][0]["location"] == "row 0"
    assert report["diagnostics"][0]["code"] == "length_mismatch"


def test_empty_jsonl_exits_one(tmp_path):
    """Input with no records is an error, not a silent success."""
    path = write_jsonl(tmp_path, [])
    report_path = tmp_path / "report.json"

    exit_code = run_main([path, "--report", str(report_path)])

    assert exit_code == 1
    assert read_report(report_path)["checked"] == 0


# ---------------------------------------------------------------------------
# Usage, loading, and report errors: exit code 2
# ---------------------------------------------------------------------------


def test_missing_input_path_exits_two(tmp_path, capsys):
    """A nonexistent input path is a loading failure with exit 2."""
    code = run_main([str(tmp_path / "nowhere.jsonl")])

    assert code == 2
    assert "does not exist" in capsys.readouterr().err


def test_non_dataset_directory_exits_two(tmp_path, capsys):
    """A directory that is not a save_to_disk dataset fails to load (exit 2)."""
    junk = tmp_path / "junk_dir"
    junk.mkdir()
    (junk / "notes.txt").write_text("hello", encoding="utf-8")

    code = run_main([str(junk)])

    assert code == 2
    assert "failed to read input" in capsys.readouterr().err


def test_max_length_rejected_for_text_rl_profile(tmp_path, capsys):
    """--max-length belongs to the tokenized-sft profile only."""
    path = write_jsonl(tmp_path, VALID_TEXT_RL_RECORDS)

    code = run_main([path, "--profile", "text-rl", "--max-length", "16"])

    assert code == 2
    assert "--max-length" in capsys.readouterr().err


def test_non_positive_max_length_rejected(datasets_lib, tmp_path, capsys):
    """--max-length must be a positive integer."""
    path = save_to_disk_dir(datasets_lib, VALID_TOKENIZED_SFT_RECORDS, tmp_path)

    code = run_main([path, "--profile", "tokenized-sft", "--max-length", "0"])

    assert code == 2
    assert "positive" in capsys.readouterr().err


def test_non_positive_max_errors_rejected(tmp_path, capsys):
    """--max-errors must be a positive integer."""
    path = write_jsonl(tmp_path, VALID_TEXT_RL_RECORDS)

    code = run_main([path, "--max-errors", "0"])

    assert code == 2
    assert "positive" in capsys.readouterr().err


def test_empty_require_field_name_rejected(tmp_path, capsys):
    """--require-field with an empty name is a usage error."""
    path = write_jsonl(tmp_path, VALID_TEXT_RL_RECORDS)

    code = run_main([path, "--require-field", ""])

    assert code == 2
    assert "non-empty" in capsys.readouterr().err


def test_split_option_rejected_for_jsonl_input(tmp_path, capsys):
    """--split only applies to DatasetDict inputs."""
    path = write_jsonl(tmp_path, VALID_TEXT_RL_RECORDS)

    code = run_main([path, "--split", "train"])

    assert code == 2
    assert "--split" in capsys.readouterr().err


def test_split_option_rejected_for_plain_dataset(datasets_lib, tmp_path, capsys):
    """--split is refused for a single saved dataset too."""
    path = save_to_disk_dir(datasets_lib, VALID_TEXT_RL_RECORDS, tmp_path)

    code = run_main([path, "--split", "train"])

    assert code == 2
    assert "--split" in capsys.readouterr().err


def test_unknown_profile_is_rejected(tmp_path):
    """argparse rejects profiles outside the supported two with exit 2."""
    path = write_jsonl(tmp_path, VALID_TEXT_RL_RECORDS)

    code = run_main([path, "--profile", "dpo"])

    assert code == 2


def test_unwritable_report_exits_two_even_for_valid_data(tmp_path, capsys):
    """Failing to write --report is a report-writing error with exit 2."""
    path = write_jsonl(tmp_path, VALID_TEXT_RL_RECORDS)
    bad_report = tmp_path / "no_such_dir" / "report.json"

    code = run_main([path, "--report", str(bad_report)])

    assert code == 2
    assert "report" in capsys.readouterr().err


def test_unwritable_report_overrides_invalid_data_exit(tmp_path, capsys):
    """Report-writing failure takes precedence with exit code 2."""
    path = write_jsonl(tmp_path, [{"messages": []}])
    bad_report = tmp_path / "no_such_dir" / "report.json"

    code = run_main([path, "--report", str(bad_report)])

    assert code == 2
    assert "report" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Determinism and input immutability
# ---------------------------------------------------------------------------


def test_repeated_scans_are_deterministic_with_truncated_diagnostics(tmp_path):
    """20 invalid records with --max-errors 2 keep counts stable and report
    at most two ordered diagnostics plus truncation, identically per run."""
    path = write_jsonl(tmp_path, [{"messages": []}] * 20)
    first_report = tmp_path / "report_a.json"
    second_report = tmp_path / "report_b.json"

    first = run_main([path, "--max-errors", "2", "--report", str(first_report)])
    second = run_main([path, "--max-errors", "2", "--report", str(second_report)])

    assert first == second == 1
    report_a = read_report(first_report)
    report_b = read_report(second_report)
    assert report_a["checked"] == 20
    assert report_a["invalid"] == 20
    assert report_a["valid"] == 0
    assert report_a["truncated"] is True
    assert len(report_a["diagnostics"]) <= 2
    assert report_a == report_b


def test_validation_does_not_modify_jsonl_input(tmp_path):
    """Scanning reads the JSONL file without rewriting or reordering it."""
    path = write_jsonl(tmp_path, VALID_TEXT_RL_RECORDS)
    before = Path(path).read_bytes()

    run_main([path, "--require-field", "answer"])

    assert Path(path).read_bytes() == before


def test_validation_does_not_modify_saved_dataset(datasets_lib, tmp_path):
    """Scanning a saved dataset leaves every row's content unchanged."""
    path = save_to_disk_dir(datasets_lib, VALID_TOKENIZED_SFT_RECORDS, tmp_path)
    dataset = datasets_lib.load_from_disk(path)
    before = dataset.to_list()

    run_main([path, "--profile", "tokenized-sft"])

    assert datasets_lib.load_from_disk(path).to_list() == before
    assert len(before) == 3


def test_help_exits_zero(capsys):
    """--help works on CPU without loading any data."""
    code = run_main(["--help"])

    assert code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "text-rl" in out and "tokenized-sft" in out


# ---------------------------------------------------------------------------
# Subprocess CLI integration tests (`python -m areal.dataset.validate`)
# ---------------------------------------------------------------------------


def run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "areal.dataset.validate", *args],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


@requires_full_cli
def test_subprocess_cli_valid_text_rl_run(tmp_path):
    """The documented module CLI validates a JSONL fixture and exits 0."""
    path = write_jsonl(tmp_path, VALID_TEXT_RL_RECORDS)
    report_path = tmp_path / "report.json"

    proc = run_cli(
        [
            path,
            "--profile",
            "text-rl",
            "--require-field",
            "answer",
            "--report",
            str(report_path),
        ]
    )

    assert proc.returncode == 0, proc.stderr
    report = read_report(report_path)
    assert (report["checked"], report["valid"], report["invalid"]) == (3, 3, 0)


@requires_full_cli
def test_subprocess_cli_invalid_run_exits_one(tmp_path):
    """Invalid records propagate exit code 1 through the module CLI."""
    path = write_jsonl(tmp_path, [{"messages": [{"role": "user"}]}])
    report_path = tmp_path / "report.json"

    proc = run_cli([path, "--report", str(report_path)])

    assert proc.returncode == 1, proc.stderr
    assert read_report(report_path)["invalid"] == 1


@requires_full_cli
def test_subprocess_cli_usage_errors_exit_two(tmp_path):
    """A nonexistent path and an unknown profile both exit 2 via the module CLI."""
    missing = run_cli([str(tmp_path / "nope.jsonl")])
    assert missing.returncode == 2

    path = write_jsonl(tmp_path, VALID_TEXT_RL_RECORDS)
    bad_profile = run_cli([path, "--profile", "bogus"])
    assert bad_profile.returncode == 2


@requires_full_cli
def test_subprocess_cli_help_exits_zero():
    """`--help` runs on CPU through the module CLI and lists both profiles."""
    proc = run_cli(["--help"])

    assert proc.returncode == 0, proc.stderr
    assert "text-rl" in proc.stdout
    assert "tokenized-sft" in proc.stdout
