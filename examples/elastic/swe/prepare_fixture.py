# SPDX-License-Identifier: Apache-2.0

"""Join a SWE-bench row with its grading metadata for a sandbox smoke test."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any


def _test_list(value: str | list[str]) -> list[str]:
    result = json.loads(value) if isinstance(value, str) else value
    if not isinstance(result, list) or any(not isinstance(x, str) for x in result):
        raise ValueError("Expected a list of test identifiers")
    return result


def build_fixture(row: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    for key in ("instance_id", "base_commit", "repo", "version"):
        if not row.get(key) or row[key] != details.get(key):
            raise ValueError(f"Dataset and grading metadata disagree on {key}")
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if set(_test_list(row[key])) != set(_test_list(details[key])):
            raise ValueError(f"Dataset and grading metadata disagree on {key}")
    if not _test_list(details["FAIL_TO_PASS"]):
        raise ValueError("The smoke fixture needs at least one failing test")
    if not row.get("patch") or not row.get("problem_statement"):
        raise ValueError("A gold patch and problem statement are required")
    script = details.get("script")
    if not isinstance(script, str) or not script.strip():
        raise ValueError("Missing evaluation script")

    reference = {
        key: details[key]
        for key in ("instance_id", "repo", "version", "FAIL_TO_PASS", "PASS_TO_PASS")
    }
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        reference[key] = _test_list(reference[key])
    encoded = base64.b64encode(json.dumps(reference).encode()).decode()
    # Keep upstream script text opaque to AenvSWE.process_eval_script and its
    # heredoc writer; only the raw subprocess should interpret those commands.
    raw_encoded = base64.b64encode(script.encode()).decode()

    # A harness shell script may return zero after resetting test files even
    # when pytest failed. Determine reward from the reference F2P/P2P report.
    wrapped = f"""#!/bin/bash
set -u
validation_dir=$(mktemp -d)
trap 'rm -rf "$validation_dir"' EXIT
printf '%s' '{raw_encoded}' | base64 --decode >"$validation_dir/raw-eval.sh"
bash "$validation_dir/raw-eval.sh" >"$validation_dir/test.log" 2>&1
cat "$validation_dir/test.log"
"${{SWE_GRADER_PYTHON:-python3}}" - "$validation_dir/test.log" <<'PY_GRADE'
import base64
import json
import os
import sys

sys.path.insert(0, os.environ["SWE_EVAL_ROOT"])
from aenv_evaluation import get_eval_report
from grading import get_logs_eval
from swe_model import SweRun

reference = json.loads(base64.b64decode("{encoded}"))
spec = SweRun(reference["instance_id"], reference["repo"],
              reference["version"], "", "", "")
spec.FAIL_TO_PASS = reference["FAIL_TO_PASS"]
spec.PASS_TO_PASS = reference["PASS_TO_PASS"]
observed, found = get_logs_eval(spec, sys.argv[1])
required = set(spec.FAIL_TO_PASS + spec.PASS_TO_PASS)
if not found or required.difference(observed):
    raise RuntimeError("Evaluation did not report every reference test")
report = get_eval_report(spec, sys.argv[1], include_tests_status=True)
print(json.dumps(report, sort_keys=True))
if "tests_status" not in report[reference["instance_id"]]:
    raise RuntimeError("Evaluation did not produce a completed test report")
resolved = report[reference["instance_id"]]["resolved"]
print("AUTOPILOT_SWE_RESOLVED=" + json.dumps(resolved))
sys.exit(0 if resolved else 1)
PY_GRADE
"""
    return {
        **row,
        "eval_script": wrapped,
        "grading_script_sha256": hashlib.sha256(script.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", required=True, type=Path)
    parser.add_argument("--details", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(
        json.loads(args.row.read_text()), json.loads(args.details.read_text())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
