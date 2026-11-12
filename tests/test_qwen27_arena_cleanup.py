import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest


def _cleanup_source():
    recipe = (
        Path(__file__).resolve().parents[1] / "examples/swe/rl/qwen38_27b/train.sbatch"
    )
    source = recipe.read_text()
    start = source.index(
        "import concurrent.futures\n", source.index("cancel_arena_children()")
    )
    return source[start : source.index("\nPY\n}", start)]


def test_cleanup_cancels_only_recorded_models_across_streams(tmp_path, monkeypatch):
    config = tmp_path / "train.yaml"
    config.write_text(
        "econfig:\n  arena_streams:\n"
        + "".join(
            f"  - name: {name}\n    stream_id: {name}\n"
            for name in ("one", "two", "three")
        )
    )
    log = tmp_path / "rollout.log"
    log.write_text("model_name=stream-areal-owned\n")
    for key, value in {
        "ARENA_CANCEL_ROLLOUT_LOG": str(log),
        "QWEN_CONFIG": str(config),
        "ARENA_OPENAPI_BASE": "https://example.invalid",
        "ARENA_OPENAPI_TOKEN": "test-placeholder",
        "EXPERIMENT_ROOT": str(tmp_path),
        "SLURM_JOB_ID": "test",
        "ARENA_CANCEL_DRY_RUN": "0",
    }.items():
        monkeypatch.setenv(key, value)
    seen, cancelled = set(), set()

    class Response:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def urlopen(request, timeout):
        parsed = urlparse(request.full_url)
        if request.method == "POST":
            cancelled.add(parsed.path.split("/")[-2])
            return Response({})
        stream = parsed.path.split("/")[-2]
        query = parse_qs(parsed.query)
        status = query["status"][0]
        seen.add((stream, status))
        if status != "RUNNING":
            return Response({"items": [], "total": 0})
        return Response(
            {
                "items": [
                    {
                        "job": {
                            "job_id": f"{stream}-owned",
                            "model": {"model_name": "stream-areal-owned"},
                        }
                    },
                    {
                        "job": {
                            "job_id": f"{stream}-foreign",
                            "model": {"model_name": "stream-areal-other"},
                        }
                    },
                ],
                "total": 2,
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    exec(compile(_cleanup_source(), "arena_cleanup", "exec"), {})
    assert len(seen) == 12
    assert cancelled == {"one-owned", "two-owned", "three-owned"}
    audit = json.loads((tmp_path / "arena-cancel-test.json").read_text())
    assert audit["selected"] == 3


def test_missing_stream_refuses_cleanup(tmp_path, monkeypatch):
    config, log = tmp_path / "train.yaml", tmp_path / "rollout.log"
    config.write_text("econfig: {}\n")
    log.write_text("model_name=stream-areal-owned\n")
    monkeypatch.setenv("ARENA_CANCEL_ROLLOUT_LOG", str(log))
    monkeypatch.setenv("QWEN_CONFIG", str(config))
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: pytest.fail("unexpected request")
    )
    with pytest.raises(RuntimeError, match="Cannot identify"):
        exec(compile(_cleanup_source(), "arena_cleanup", "exec"), {})
