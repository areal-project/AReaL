# SPDX-License-Identifier: Apache-2.0
"""Stable QSA top-k for SGLang 0.5.19.dev125+g119b5ffe4.

Apply only to a fresh writable container layer; unknown or patched source is rejected."""

import argparse
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path

EXPECTED_SHA256 = "5482e38d30bfaf1624ec0625b4896cbb395a1637f75c183c8ca723c9f6055ff8"
TARGET_RELATIVE = Path("srt/layers/attention/qsa/kernel.py")


def stable_qsa_topk(logits, row_starts, row_ends, topk):
    """Reference selection: score descending, lower relative ID wins exact ties."""
    import torch

    if (
        logits.ndim != 2
        or logits.dtype != torch.float32
        or type(topk) is not int
        or topk <= 0
    ):
        raise ValueError("Requires FP32 [rows,keys] scores and positive integer topk")
    rows, keys = logits.shape
    if row_starts.shape != (rows,) or row_ends.shape != (rows,):
        raise ValueError("Row bounds must match score rows")
    if row_starts.dtype not in (torch.int32, torch.int64) or row_ends.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Row bounds must be integer tensors")
    starts = row_starts.to(device=logits.device, dtype=torch.long)
    ends = row_ends.to(device=logits.device, dtype=torch.long)
    if not ((starts >= 0) & (starts <= ends) & (ends <= keys)).all():
        raise ValueError("Invalid row bounds")
    positions = torch.arange(keys, device=logits.device)[None, :]
    valid = (positions >= starts[:, None]) & (positions < ends[:, None])
    if not torch.isfinite(logits[valid]).all():
        raise ValueError("Nonfinite valid scores")
    # Invalid positions cannot beat any finite candidate. Stable sort retains
    # ascending absolute (and hence relative) IDs when scores are exactly equal.
    scores = logits.masked_fill(~valid, -float("inf"))
    width = min(topk, keys)
    chosen = scores.argsort(dim=-1, descending=True, stable=True)[:, :width]
    chosen_valid = valid.gather(1, chosen)
    relative = chosen - starts[:, None]
    sentinel = torch.iinfo(torch.int32).max
    relative = relative.masked_fill(~chosen_valid, sentinel).sort(dim=-1).values
    result = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    result[:, :width] = torch.where(relative == sentinel, -1, relative).to(torch.int32)
    return result


WRAPPER = """
import os as _qsa_os
_qsa_original_fast_topk = qsa_fast_topk

def qsa_fast_topk(logits, row_starts, row_ends, topk):
    if _qsa_os.environ.get("QWEN_QSA_STABLE_TOPK") == "1":
        return stable_qsa_topk(logits, row_starts, row_ends, topk)
    return _qsa_original_fast_topk(logits, row_starts, row_ends, topk)
"""


def patched_source(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Unexpected QSA kernel source SHA256; refusing patch")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "qsa_fast_topk"
    )
    if [arg.arg for arg in function.args.args] != [
        "logits",
        "row_starts",
        "row_ends",
        "topk",
    ]:
        raise ValueError("Unexpected qsa_fast_topk signature")
    result = source + "\n" + inspect.getsource(stable_qsa_topk) + WRAPPER
    compile(result, str(TARGET_RELATIVE), "exec")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, help="Override the installed QSA kernel path"
    )
    args = parser.parse_args()
    target = args.target
    if target is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or spec.origin is None:
            raise RuntimeError("Cannot locate the installed SGLang package")
        target = Path(spec.origin).parent / TARGET_RELATIVE
    original = target.read_text()
    target.write_text(patched_source(original))


if __name__ == "__main__":
    main()
