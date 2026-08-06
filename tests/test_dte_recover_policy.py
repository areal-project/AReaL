# SPDX-License-Identifier: Apache-2.0

import json

from areal.infra.launcher.dte_recover import (
    archive_mismatch_flag,
    load_mismatch_flag,
)


def _write_flag(tmp_path, payload):
    group_dir = tmp_path / "capture" / "group"
    group_dir.mkdir(parents=True)
    flag_path = group_dir / "mismatch_flag.json"
    flag_path.write_text(json.dumps(payload), encoding="utf-8")
    return flag_path


def test_expected_peer_timeout_flag_is_recoverable(tmp_path):
    flag_path = _write_flag(
        tmp_path,
        {"mismatch": {"reason": "expected_peer_timeout"}},
    )

    flag = load_mismatch_flag(
        {
            "AREAL_DTE_WEIGHT_CAPTURE_ROOT": str(tmp_path / "capture"),
            "AREAL_DTE_WEIGHT_COMPARE_GROUP": "group",
        }
    )

    assert flag is not None
    assert flag.path == str(flag_path)
    assert flag.reason == "expected_peer_timeout"
    assert flag.recoverable


def test_peer_digest_mismatch_flag_blocks_recovery(tmp_path):
    _write_flag(
        tmp_path,
        {"mismatch": {"reason": "peer_digest_mismatch"}},
    )

    flag = load_mismatch_flag(
        {
            "AREAL_DTE_WEIGHT_CAPTURE_ROOT": str(tmp_path / "capture"),
            "AREAL_DTE_WEIGHT_COMPARE_GROUP": "group",
        }
    )

    assert flag is not None
    assert flag.reason == "peer_digest_mismatch"
    assert not flag.recoverable


def test_observed_stale_flag_uses_original_reason(tmp_path):
    _write_flag(
        tmp_path,
        {
            "mismatch": {
                "reason": "mismatch_flag_observed",
                "flag": {"mismatch": {"reason": "expected_peer_timeout"}},
            }
        },
    )

    flag = load_mismatch_flag(
        {
            "AREAL_DTE_WEIGHT_CAPTURE_ROOT": str(tmp_path / "capture"),
            "AREAL_DTE_WEIGHT_COMPARE_GROUP": '"group"',
        }
    )

    assert flag is not None
    assert flag.reason == "expected_peer_timeout"
    assert flag.recoverable


def test_archive_mismatch_flag_moves_original(tmp_path):
    flag_path = _write_flag(
        tmp_path,
        {"mismatch": {"reason": "expected_peer_timeout"}},
    )

    archive_path = archive_mismatch_flag(str(flag_path), run_id=0)

    assert not flag_path.exists()
    assert archive_path.startswith(str(flag_path) + ".archived_run0_")
    assert json.loads(open(archive_path, encoding="utf-8").read()) == {
        "mismatch": {"reason": "expected_peer_timeout"}
    }
