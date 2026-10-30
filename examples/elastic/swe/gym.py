# SPDX-License-Identifier: Apache-2.0

"""Adapt a pinned SWE-Gym harness checkout to the existing SWE sandbox contract."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path


def load_harness(root: Path):
    """Load upstream harness code without its unrelated Docker/collection CLI imports."""
    root = root.resolve()
    for name, directory in (
        ("swebench", root / "swebench"),
        ("swebench.harness", root / "swebench/harness"),
    ):
        if not directory.is_dir():
            raise ValueError(f"Missing SWE-Gym harness directory: {directory}")
        existing = sys.modules.get(name)
        if existing is not None and list(existing.__path__) != [str(directory)]:
            raise ValueError("A different swebench package is already imported")
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
    return importlib.import_module("swebench.harness.test_spec")


def grader_bundle(root: Path) -> tuple[str, dict[str, str]]:
    """Package native parsers/scoring functions; no harness installation in task images.

    TestSpec and get_eval_report are orchestration wrappers with heavy imports.
    Keep the upstream scoring functions verbatim and call them on parsed test logs.
    """
    stream = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:

        def write(name, content):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)

        for name in ("swebench/__init__.py", "swebench/harness/__init__.py"):
            write(name, "")
        write("LICENSE", (root / "LICENSE").read_bytes())
        for name in ("constants.py", "log_parsers.py", "grading.py"):
            path = root / "swebench/harness" / name
            source = path.read_text()
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if name == "grading.py":
                source = "\n\n".join(
                    ast.get_source_segment(source, node)
                    for node in ast.parse(source).body
                    if not (
                        isinstance(node, ast.ImportFrom)
                        and node.module == "swebench.harness.test_spec"
                        or isinstance(node, ast.FunctionDef)
                        and node.name == "get_eval_report"
                    )
                )
            write("swebench/harness/" + name, source)
    return base64.b64encode(stream.getvalue()).decode(), hashes


def build_gym_fixture(row: dict, harness: Path, *, provenance: dict) -> dict:
    from .prepare_fixture import _test_list

    for key in (
        "instance_id",
        "base_commit",
        "repo",
        "version",
        "problem_statement",
        "patch",
        "test_patch",
    ):
        if not row.get(key):
            raise ValueError(f"SWE-Gym row is missing {key}")
    reference = {key: row[key] for key in ("instance_id", "repo", "version")}
    reference.update(
        {key: _test_list(row[key]) for key in ("FAIL_TO_PASS", "PASS_TO_PASS")}
    )
    if not reference["FAIL_TO_PASS"]:
        raise ValueError("SWE-Gym task must have failing reference tests")
    native = load_harness(harness)
    specs = native.MAP_REPO_VERSION_TO_SPECS[row["repo"].lower()][row["version"]]
    commands = native.make_eval_script_list(
        row, specs, "testbed", "/testbed", row["base_commit"], row["test_patch"]
    )
    script = "\n".join(["#!/bin/bash", "set -xo pipefail", *commands]) + "\n"
    encoded = base64.b64encode(script.encode()).decode()
    bundle, hashes = grader_bundle(harness)
    encoded_reference = base64.b64encode(json.dumps(reference).encode()).decode()
    wrapped = f'''#!/bin/bash
set -u
validation_dir=$(mktemp -d)
trap 'rm -rf "$validation_dir"' EXIT
printf '%s' '{encoded}' | base64 --decode >"$validation_dir/raw-eval.sh"
bash "$validation_dir/raw-eval.sh" >"$validation_dir/test.log" 2>&1
cat "$validation_dir/test.log"
"${{SWE_GRADER_PYTHON:-python3}}" - "$validation_dir" <<'PY_GRADE'
import base64, io, json, pathlib, sys, zipfile
root = pathlib.Path(sys.argv[1])
with zipfile.ZipFile(io.BytesIO(base64.b64decode("{bundle}"))) as archive:
    archive.extractall(root / "grader")
sys.path.insert(0, str(root / "grader"))
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.constants import ResolvedStatus
reference = json.loads(base64.b64decode("{encoded_reference}"))
log = (root / "test.log").read_text(errors="replace")
if "applied patch" not in log.lower():
    raise RuntimeError("Reference test patch was not applied")
observed = MAP_REPO_TO_PARSER[reference["repo"].lower()](log)
required = set(reference["FAIL_TO_PASS"] + reference["PASS_TO_PASS"])
missing = required.difference(observed)
if missing:
    raise RuntimeError("Evaluation omitted reference tests: " + repr(sorted(missing)))
report = get_eval_tests_report(observed, reference)
resolved = get_resolution_status(report) == ResolvedStatus.FULL.value
print(json.dumps({{reference["instance_id"]: {{"resolved": resolved, "tests_status": report}}}}, sort_keys=True))
print("AUTOPILOT_SWE_RESOLVED=" + json.dumps(resolved))
sys.exit(0 if resolved else 1)
PY_GRADE
'''
    image = (
        "xingyaoww/sweb.eval.x86_64."
        + row["instance_id"].lower().replace("__", "_s_")
        + ":latest"
    )
    return {
        **row,
        "eval_script": wrapped,
        "grading_script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "source_image": image,
        # The existing AEnvironment imagePrefix:datasource mapping uses this alias.
        "sandbox_image": row["instance_id"].lower(),
        "provenance": {**provenance, "grader_files_sha256": hashes},
    }
