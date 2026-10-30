# SPDX-License-Identifier: Apache-2.0

"""Local manual control: python -m areal.elastic.control status|start|stop."""

import argparse
import json
import time
import uuid
from pathlib import Path

from areal.utils import logging
from areal.utils.atomic import atomic_json

logger = logging.getLogger("RLTrainer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "start", "stop"])
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    status = json.loads((args.state_root / "status.json").read_text())
    if args.action == "status":
        logger.info(json.dumps(status, indent=2))
        return
    request_id = uuid.uuid4().hex
    request = args.state_root / "control/requests" / f"{request_id}.json"
    response = args.state_root / "control/responses" / request.name
    atomic_json(
        request,
        {"action": args.action + "_actor", "generation": status["generation"]},
    )
    deadline = time.monotonic() + args.timeout
    while not response.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Command pending; inspect {response}")
        time.sleep(0.2)
    result = json.loads(response.read_text())
    logger.info(json.dumps(result, indent=2))
    if result["status"] != "completed":
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
