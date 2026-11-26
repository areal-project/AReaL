# SPDX-License-Identifier: Apache-2.0

"""JSON proposal sources for the revision loop."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import asdict, dataclass
from math import isfinite

from .contracts import Proposal, ProposalRequest


@dataclass(frozen=True)
class FixedProposals:
    proposals: tuple[Proposal, ...]

    def propose(self, request: ProposalRequest) -> Proposal:
        return self.proposals[request.round - 1]


@dataclass(frozen=True)
class CommandRevisionAgent:
    """Send a request on stdin and read one proposal object from stdout."""

    command: tuple[str, ...]
    timeout_seconds: float = 120.0

    def __post_init__(self):
        if (
            not self.command
            or not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("Agent requires a command and a finite positive timeout")

    def propose(self, request: ProposalRequest) -> Proposal:
        with subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        ) as process:
            try:
                output, _ = process.communicate(
                    json.dumps(asdict(request), allow_nan=False),
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                raise TimeoutError("Revision agent timed out") from None
        if process.returncode:
            raise RuntimeError(
                f"Revision agent exited with status {process.returncode}"
            )
        if len(output.encode()) > 256_000:
            raise ValueError("Agent proposal exceeds 256 KB")
        return Proposal(**json.loads(output))
