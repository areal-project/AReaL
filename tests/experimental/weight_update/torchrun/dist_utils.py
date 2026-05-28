# SPDX-License-Identifier: Apache-2.0

import torch.distributed as dist


def write_result(out: str, succ: bool, error: str = "") -> None:
    with open(out, "w") as f:
        f.write("Passed" if succ else "Failed")
        if error:
            f.write(f"\n{error}")


def print_rank0(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg)
