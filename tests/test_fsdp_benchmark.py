# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from areal.tools.benchmark_fsdp import (
    create_plan,
    main,
    parser,
    render_report,
    run_process,
    run_trial,
    summarize_trial,
)
from areal.tools.fsdp_benchmark_worker import make_batch


def arguments(tmp_path, *extra):
    return parser().parse_args(
        [
            "--model-path",
            "unused/model",
            "--seq-lens",
            "8,4,6,5",
            "--token-budgets",
            "8,16",
            "--steps",
            "2",
            "--output-dir",
            str(tmp_path / "results"),
            *extra,
        ]
    )


@pytest.mark.parametrize(
    "extra,match",
    [
        (["--seq-lens", "1,3"], "at least 2"),
        (["--nproc-per-node", "3"], "divisible"),
        (["--token-budgets", "4"], "longest"),
        (["--prompt-tokens", "4"], "smaller"),
        (["--warmup-steps", "0"], "positive"),
        (["--timeout", "nan"], "finite"),
        (["--seed", "-1"], "seed"),
    ],
)
def test_plan_rejects_invalid_workloads(tmp_path, extra, match):
    """Reject invalid workloads before starting any expensive subprocess."""
    with pytest.raises(ValueError, match=match):
        create_plan(arguments(tmp_path, *extra))


def test_dry_run_preserves_fixed_workload_and_rejects_overwrite(tmp_path):
    """All candidates share one global workload; dry-run needs no checkpoint."""
    args = arguments(tmp_path)
    plan = create_plan(args)
    assert plan["input_tokens_per_step"] == 23
    assert plan["loss_tokens_per_step"] == 19
    assert len(plan["trials"]) == 4
    argv = [
        "--model-path",
        "nonexistent",
        "--seq-lens",
        "8,4,6,5",
        "--token-budgets",
        "8,16",
        "--output-dir",
        str(args.output_dir),
        "--dry-run",
    ]
    assert main(argv) == 0
    saved = json.loads((args.output_dir / "plan.json").read_text())
    assert saved["sequence_lengths"] == [8, 4, 6, 5]
    assert "resolved_model_path" not in saved
    with pytest.raises(SystemExit):
        main(argv)


def test_masks_and_dp_shards_preserve_sequence_boundaries():
    """Packed next-token labels must not introduce cross-sequence loss."""
    lengths = [8, 4, 6, 5]
    full, digest = make_batch(lengths, 2, 100, 42, 0, 1)
    shards = [make_batch(lengths, 2, 100, 42, rank, 2) for rank in range(2)]
    assert all(d == digest for _, d in shards)
    assert make_batch(lengths, 2, 100, 43, 0, 1)[1] != digest
    for rank, (shard, _) in enumerate(shards):
        for key in full:
            torch.testing.assert_close(shard[key], full[key][rank::2], rtol=0, atol=0)
    assert full["attention_mask"].count_nonzero().item() == sum(lengths)
    assert full["loss_mask"].count_nonzero().item() == sum(n - 2 for n in lengths)
    packed_mask = full["loss_mask"][full["attention_mask"]]
    boundary = 0
    for i, length in enumerate(lengths):
        assert not full["loss_mask"][i, length - 1 :].any()
        assert full["loss_mask"][i, :length].count_nonzero().item() == length - 2
        boundary += length
        assert not packed_mask[boundary - 1]


def test_output_inside_model_is_rejected_before_writing(tmp_path):
    """Generated artifacts must not become part of the checkpoint manifest."""
    model = tmp_path / "model"
    model.mkdir()
    output = model / "results"
    with pytest.raises(SystemExit):
        main(
            [
                "--model-path",
                str(model),
                "--seq-lens",
                "8,4",
                "--token-budgets",
                "8",
                "--output-dir",
                str(output),
            ]
        )
    assert not output.exists()


def rank_result(rank, times, input_tokens, loss_tokens, microbatches):
    return {
        "rank": rank,
        "global_batch_sha256": "fixed-fixture-digest",
        "input_tokens": input_tokens,
        "loss_tokens": loss_tokens,
        "samples": [
            {
                "seconds": t,
                "prepared_microbatches": microbatches,
                "peak_allocated_bytes": 100 + rank,
                "peak_reserved_bytes": 200 + rank,
            }
            for t in times
        ],
    }


def test_aggregation_uses_each_steps_slowest_rank(tmp_path):
    """Changing the slowest rank between steps must not inflate throughput."""
    plan = create_plan(arguments(tmp_path, "--nproc-per-node", "2"))
    ranks = [rank_result(0, [1.0, 3.0], 14, 12, 2), rank_result(1, [2.0, 1.0], 9, 7, 3)]
    result = summarize_trial(plan, ranks)
    assert result["step_seconds"] == [2.0, 3.0]
    assert result["input_tokens_per_second"] == pytest.approx(46 / 5)
    assert result["loss_tokens_per_second"] == pytest.approx(38 / 5)
    assert result["prepared_microbatches_per_step_per_rank"] == [[2, 3], [2, 3]]
    assert result["executed_microbatches_per_step_per_rank"] == [[3, 3], [3, 3]]
    with pytest.raises(ValueError, match="Missing"):
        summarize_trial(plan, ranks[:1])
    ranks[0]["input_tokens"] = 15
    with pytest.raises(ValueError, match="workload"):
        summarize_trial(plan, ranks)


def test_real_subprocess_failure_does_not_prevent_next_trial(tmp_path):
    """Run actual child processes, including a nonzero exit and a subsequent success."""
    failed = run_process(
        [sys.executable, "-c", "raise RuntimeError('fixture failure')"],
        tmp_path / "failed.log",
        5,
    )
    success = run_process(
        [sys.executable, "-c", "print('completed')"], tmp_path / "ok.log", 5
    )
    assert failed["returncode"] != 0
    assert "fixture failure" in (tmp_path / "failed.log").read_text()
    assert success["returncode"] == 0
    assert "completed" in (tmp_path / "ok.log").read_text()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process groups and /proc")
def test_timeout_kills_owned_descendants_and_preserves_unrelated_process(tmp_path):
    """A hanging trial's child must terminate without touching another session."""
    pidfile = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'], start_new_session=True); "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        outcome = run_process(
            [sys.executable, "-c", script], tmp_path / "timeout.log", 1
        )
        assert outcome["status"] == "timeout"
        child = int(pidfile.read_text())
        for _ in range(50):
            try:
                state = Path(f"/proc/{child}/stat").read_text().split()[2]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.02)
        else:
            pytest.fail("Timed-out trial left a live descendant")
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_trial_reports_oom_and_preserves_partial_failures(tmp_path, monkeypatch):
    """Worker OOM details survive even when the other rank is terminated."""
    plan = create_plan(arguments(tmp_path))

    def oom(command, log, timeout):
        config = json.loads((log.parent / "config.json").read_text())
        assert config["sequence_lengths"] == plan["sequence_lengths"]
        (log.parent / "failure-0.json").write_text(
            json.dumps({"type": "OutOfMemoryError", "message": "fixture"})
        )
        return {"status": "completed", "returncode": 1}

    monkeypatch.setattr("areal.tools.benchmark_fsdp.run_process", oom)
    result = run_trial(plan, plan["trials"][0], tmp_path)
    assert result["status"] == "oom"
    assert result["failures"][0]["message"] == "fixture"
    report = render_report({"plan": plan, "trials": [result]})
    assert "oom" in report
    assert "| — | — | — | — |" in report


def test_checkpoint_mutation_invalidates_completed_trial(tmp_path, monkeypatch):
    """A successful process must not leave an ok artifact after model changes."""
    model = tmp_path / "model"
    model.mkdir()
    checkpoint = model / "config.json"
    checkpoint.write_text("{}")

    def mutate(plan, trial, root):
        checkpoint.write_text('{"changed": true}')
        return {**trial, "status": "ok"}

    monkeypatch.setattr("areal.tools.benchmark_fsdp.run_trial", mutate)
    output = tmp_path / "results"
    assert (
        main(
            [
                "--model-path",
                str(model),
                "--seq-lens",
                "8,4",
                "--token-budgets",
                "8",
                "--output-dir",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads((output / "report.json").read_text())
    assert report["trials"][0]["status"] == "failed"
    assert "changed" in report["fatal_error"]
    assert "Sweep stopped" in (output / "report.md").read_text()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper")
def test_torchrun_timeout_cleans_detached_worker(tmp_path):
    """Exercise torchrun's real per-worker sessions with a worker ignoring TERM."""
    pidfile = tmp_path / "worker.pid"
    script = tmp_path / "worker.py"
    script.write_text(
        "import os, signal, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    outcome = run_process(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            str(script),
        ],
        tmp_path / "torchrun.log",
        8,
    )
    assert outcome["status"] == "timeout"
    assert pidfile.exists(), (tmp_path / "torchrun.log").read_text()
    assert not Path(f"/proc/{int(pidfile.read_text())}").exists()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_cli_signal_cleans_trial_and_records_interruption(tmp_path, signum):
    """Real CLI signals trigger descendant cleanup and an interrupted artifact."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    output = tmp_path / "results"
    pidfile = tmp_path / "worker.pid"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, signal, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import sys\nimport areal.tools.benchmark_fsdp as b\n"
        "def trial(plan, trial, root):\n"
        "    d=root/trial['id']; d.mkdir()\n"
        f"    return b.run_process([sys.executable, {str(worker)!r}], d/'output.log', 60)\n"
        "b.run_trial=trial\n"
        f"sys.exit(b.main(['--model-path', {str(model)!r}, '--seq-lens', '8,4', "
        f"'--token-budgets', '8', '--output-dir', {str(output)!r}]))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    cli = subprocess.Popen([sys.executable, str(runner)], env=env)
    try:
        deadline = time.monotonic() + 20
        while (
            not pidfile.exists() and cli.poll() is None and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert pidfile.exists(), "Fixture worker did not become ready"
        cli.send_signal(signum)
        assert cli.wait(timeout=20) == 128 + signum
        assert not Path(f"/proc/{int(pidfile.read_text())}").exists()
        report = json.loads((output / "report.json").read_text())
        assert report["trials"][0]["status"] == "interrupted"
        assert report["trials"][0]["signal"] == signum
    finally:
        if cli.poll() is None:
            cli.terminate()
            cli.wait(timeout=20)
