# SPDX-License-Identifier: Apache-2.0

"""Compare actor CPU routing preparation in fresh processes (no model weights).

Run once with ``--format dense`` and once with ``--format packed``. The packed
case starts at the receiver after PP-directed transfer, and therefore owns
only its stage's layers. Both cases retain input trajectories during concat.
This isolates routing storage and does not predict total training-node RSS.
"""

import argparse
import json
import resource
import sys

import torch

from areal.engine.r3.packed import PackedRouting
from areal.utils.data import concat_padded_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["dense", "packed"], required=True)
    parser.add_argument("--sequences", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--pp", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    lengths = [args.max_length // (1 << (index % 4)) for index in range(args.sequences)]
    packed = args.format == "packed"
    layers = args.layers // args.pp if packed else args.layers
    trajectories = []
    for length in lengths:
        values = torch.ones(
            (length, layers, args.topk) if packed else (1, length, layers, args.topk),
            dtype=torch.int16 if packed else torch.int32,
        )
        trajectory = {
            "routed_experts": values,
            "r3_routing_valid": torch.tensor([True]),
        }
        if packed:
            trajectory["r3_cu_seqlens"] = torch.tensor([0, length])
        trajectories.append(trajectory)
    data = concat_padded_tensors(trajectories)
    reverse = torch.arange(args.sequences - 1, -1, -1)
    if packed:
        original = PackedRouting.from_tensors(
            data["routed_experts"], data["r3_cu_seqlens"]
        )
        minibatch = original.select_rows(reverse)
        forward = minibatch.select_rows(reverse)
        storages = {
            original.values.untyped_storage().data_ptr(): original.values.untyped_storage().nbytes()
        }
    else:
        original = data["routed_experts"]
        minibatch = original[reverse]
        forward = minibatch[reverse]
        storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in (original, minibatch, forward)
        }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    json.dump(
        {
            "format": args.format,
            "sequences": args.sequences,
            "tokens": sum(lengths),
            "max_length": max(lengths),
            "baseline_peak_rss_bytes": baseline,
            "peak_rss_bytes": peak,
            "incremental_peak_rss_bytes": max(0, peak - baseline),
            "prepared_routing_storage_bytes": sum(storages.values()),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
