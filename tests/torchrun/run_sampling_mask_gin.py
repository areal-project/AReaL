# SPDX-License-Identifier: Apache-2.0
"""Run on two distinct GPUs: torchrun --nproc_per_node=2 this_file.py."""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from areal.infra.sampling_mask.gin import GinEndpoint, load_extension


def nic_bytes() -> dict[str, int]:
    """InfiniBand port_data counters are expressed in four-byte units."""
    return {
        str(path.relative_to("/sys/class/infiniband")): int(path.read_text()) * 4
        for direction in ("xmit", "rcv")
        for path in Path("/sys/class/infiniband").glob(
            f"*/ports/*/counters/port_{direction}_data"
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot-bytes", type=int, default=65536)
    parser.add_argument("--chunks", type=int, default=9)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    if args.slot_bytes < 2 or args.chunks < 1:
        parser.error("slot-bytes must be >= 2 and chunks must be positive")
    dist.init_process_group("gloo")
    if dist.get_world_size() != 2:
        raise ValueError("This test requires exactly two ranks")
    rank = dist.get_rank()
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    extension = load_extension()
    unique_id = [extension.bootstrap_id() if rank == 0 else None]
    dist.broadcast_object_list(unique_id, src=0)
    endpoint = GinEndpoint(unique_id[0], rank, device, slot_bytes=args.slot_bytes)
    try:
        dist.barrier()
        before = nic_bytes()
        started = time.monotonic()
        for i in range(args.chunks):
            size = [args.slot_bytes, 1, args.slot_bytes // 2 + 1][i % 3]
            expected = (
                (torch.arange(size, dtype=torch.int32, device=device) + i) % 256
            ).to(torch.uint8)
            payload = expected.clone() if rank == 0 else torch.zeros_like(expected)
            endpoint.transfer(payload)
            if rank == 1:
                torch.testing.assert_close(payload, expected, atol=0, rtol=0)
        dist.barrier()
        elapsed = time.monotonic() - started
        after = nic_bytes()
        if args.report_dir is not None:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / f"rank-{rank}.json").write_text(
                json.dumps(
                    {
                        "host": socket.gethostname(),
                        "rank": rank,
                        "chunks": endpoint.chunks,
                        "payload_bytes": endpoint.payload_bytes,
                        "slot_bytes": args.slot_bytes,
                        "seconds": elapsed,
                        "nic_bytes_delta": {
                            key: after[key] - value for key, value in before.items()
                        },
                        "nic_scope": "Host port counters include concurrent traffic and control; not per-process wire bytes.",
                    },
                    indent=2,
                )
                + "\n"
            )
        endpoint.close()
        if rank == 0:
            from areal.utils import logging

            logging.getLogger("RemoteInfEngine").warning(
                "Sampling mask GIN round-trip passed: %s chunks, %s payload bytes, repeated slot reuse",
                args.chunks,
                endpoint.payload_bytes,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
