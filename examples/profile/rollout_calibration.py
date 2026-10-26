# SPDX-License-Identifier: Apache-2.0
"""Fixed-token request cohorts for an independently launched SGLang replica."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

from examples.profile.calibration_cases import case_seed, select_cases, write_json


async def run(args) -> None:
    import aiohttp

    from areal.utils import logging

    logger = logging.getLogger("RolloutCalibration")
    cases = select_cases("rollout", args.cases)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "launch.json", {"arguments": vars(args)})
    headers = {}
    if key := os.environ.get("AREAL_INFERENCE_ADMIN_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        with (args.output / "requests.jsonl").open("x", buffering=1) as output:
            for case in cases:
                for repeat in range(args.repeat):

                    async def request(index):
                        rng = random.Random(case_seed(args.seed, case.name, 0, index))
                        ids = [rng.randrange(1000, 50000) for _ in range(case.prompt)]
                        rid = f"{case.name}-r{repeat}-{index}"
                        body = dict(
                            rid=rid,
                            input_ids=ids,
                            sampling_params=dict(
                                temperature=0,
                                max_new_tokens=case.output,
                                ignore_eos=True,
                            ),
                        )
                        started_at, start = time.time(), time.perf_counter()
                        async with session.post(
                            args.url.rstrip("/") + "/generate", json=body
                        ) as response:
                            response.raise_for_status()
                            result = await response.json()
                        meta = result["meta_info"]
                        if (
                            meta["prompt_tokens"] != case.prompt
                            or meta["completion_tokens"] != case.output
                        ):
                            raise RuntimeError(
                                f"Request length mismatch: {rid}: {meta}"
                            )
                        row = dict(
                            case=asdict(case),
                            repeat=repeat,
                            request_id=rid,
                            input_sha256=hashlib.sha256(
                                json.dumps(ids).encode()
                            ).hexdigest(),
                            started_at=started_at,
                            finished_at=time.time(),
                            wall_seconds=time.perf_counter() - start,
                            meta_info=meta,
                        )
                        output.write(json.dumps(row, default=str) + "\n")

                    start = time.perf_counter()
                    await asyncio.gather(*(request(i) for i in range(case.concurrency)))
                    logger.info(
                        "%s repeat=%d seconds=%.3f",
                        case.name,
                        repeat,
                        time.perf_counter() - start,
                    )
    write_json(
        args.output / "completed.json", {"status": "completed", "cases": len(cases)}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="pilot")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
