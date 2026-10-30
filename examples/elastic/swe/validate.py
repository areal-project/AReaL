# SPDX-License-Identifier: Apache-2.0

"""CPU-only empty/gold validation through the same isolated SDK and cleanup path."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from .grading import grade_live_candidate, validate_grading
from .sandbox import SandboxProcess, load_runtime


async def run(args):
    runtime = load_runtime(str(args.runtime))
    fixture = json.loads(args.fixture.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for mode, expected in (("empty", 0.0), ("gold", 1.0)):
            directory = args.output / mode
            directory.mkdir()
            started = time.monotonic()
            record = {"mode": mode, "expected_reward": expected}
            records.append(record)
            with (directory / "sandbox.log").open("w") as log:
                env = SandboxProcess(runtime, directory, log)
                try:
                    await env.start(fixture)
                    record["instance_id"] = env.instance_id
                    setup = await env.execute_bash("export ELASTIC_SWE_PROBE=preserved")
                    check = await env.execute_bash(
                        'test "$ELASTIC_SWE_PROBE" = preserved'
                    )
                    failure = await env.execute_bash("false")
                    if (
                        setup["returncode"]
                        or check["returncode"]
                        or not failure["returncode"]
                    ):
                        raise RuntimeError(
                            "Persistent shell or nonzero exit semantics failed"
                        )
                    if mode == "gold":
                        patch = fixture["patch"]
                        marker = "PATCH_" + hashlib.sha256(patch.encode()).hexdigest()
                        applied = await env.execute_bash(
                            f"git apply - <<'{marker}'\n{patch}\n{marker}"
                        )
                        if applied["returncode"]:
                            raise RuntimeError("Gold patch does not apply")
                    current = await env.submit_final_solution()
                    if current["returncode"]:
                        raise RuntimeError("Patch extraction failed")
                    reward, output = await grade_live_candidate(env, current["output"])
                    record.update(
                        reward=reward,
                        grading_output=output,
                        grading_kind=validate_grading(
                            reward, output, env.tool_call_failed_count
                        ),
                    )
                    if reward != expected:
                        raise RuntimeError(f"{mode} reward {reward} != {expected}")
                finally:
                    await env.close()
                    record.update(
                        cleanup_verified=True,
                        seconds=time.monotonic() - started,
                        calls=env.calls,
                    )
    finally:
        (args.output / "result.json").write_text(
            json.dumps({"task_id": fixture["instance_id"], "checks": records}, indent=2)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
