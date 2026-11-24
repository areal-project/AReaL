from pathlib import Path

import pytest

from scripts.ascend_nightly_grpo_report import observations, read_log, report


@pytest.mark.parametrize("compare", [False, True])
def test_report_from_logs_with_optional_baseline(tmp_path: Path, compare: bool):
    """Keep validation calls sparse and generate matching table and chart columns."""
    log = tmp_path / "main.log"
    log.write_text(
        "20260101-23:00:00.000 startup\n"
        "20260101-23:30:00.000 Epoch 1/1 Step 1/2 Train step 1/2 done.\n"
        "│ timeperf/eval │ 0.001 │ rollout/reward │ 0.4 │\n"
        "20260102-00:00:00.000 Epoch 1/1 Step 2/2 Train step 2/2 done.\n"
        "│ timeperf/eval │ 21 │ eval-rollout/reward │ 0.5 │\n"
    )
    runs = {"Current": read_log(log)}
    if compare:
        log.write_text(
            log.read_text()
            + "20260102-01:00:00.000 Epoch 2/2 Step 1/2 Train step 3/4 done.\n"
        )
        runs["Baseline"] = read_log(log, limit=2)

    summary = report(runs, tmp_path, "charts")

    assert runs["Current"][-1].hours == 1
    assert observations(runs["Current"], "timeperf/eval") == [(2, 21)]
    assert ("Baseline" in summary) == compare
    assert ("Change" in summary) == compare
    assert len(list(tmp_path.glob("*.png"))) == 2
    for block in summary.split("\n\n"):
        if block.startswith("|"):
            assert len({line.count("|") for line in block.splitlines()}) == 1


def test_verify_sft_requires_complete_run_and_final_shards(tmp_path: Path):
    """A missing indexed tensor or a truncated log must fail the nightly job."""
    import getpass
    import json

    import torch
    from safetensors.torch import save_file

    from scripts.ascend_nightly_verify import verify_run

    run = Path(getpass.getuser()) / "experiment" / "nightly"
    log = tmp_path / "logs" / run / "main.log"
    log.parent.mkdir(parents=True)
    rows = []
    for epoch in (1, 2):
        rows.extend(
            [
                f"20260101-00:00:0{epoch}.000 Epoch {epoch}/2 Step 1/1 Train step {epoch}/2 done.",
                "│ sft/loss/avg │ 0.6 │ sft-eval/loss/avg │ 0.7 │",
                "│ sft/ppl/avg │ 2.1 │ sft-eval/ppl/avg │ 2.2 │",
                "│ timeperf/eval │ 5.0 │",
            ]
        )
    checkpoint = tmp_path / "checkpoints" / run / "default/epoch1epochstep0globalstep1"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json"):
        (checkpoint / name).write_text('{"model_type":"test"}')
    shard = "model-00001-of-00001.safetensors"
    save_file({"weight": torch.zeros(2)}, checkpoint / shard)
    index = checkpoint / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard}}))
    output = tmp_path / "output/metrics.json"
    log.write_text("\n".join(rows))
    verify_run(tmp_path, "experiment", "sft", output)
    assert [row["number"] for row in json.loads(output.read_text())] == [1, 2]
    assert observations(read_log(log), "timeperf/eval") == [(1, 5.0), (2, 5.0)]

    index.write_text(json.dumps({"weight_map": {"absent": shard}}))
    with pytest.raises(ValueError, match="index"):
        verify_run(tmp_path, "experiment", "sft", output)
    log.write_text("\n".join(rows[:4]))
    with pytest.raises(ValueError, match="final step"):
        verify_run(tmp_path, "experiment", "sft", output)
