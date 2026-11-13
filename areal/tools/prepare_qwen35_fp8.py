# SPDX-License-Identifier: Apache-2.0
"""Create a separate Qwen3.5 FP8 rollout checkpoint on a CUDA host.

Usage: python -m areal.tools.prepare_qwen35_fp8 BF16_CHECKPOINT FP8_CHECKPOINT
The source checkpoint is retained for Megatron training and optimizer recovery.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from areal.engine.awex.qwen3_5_fp8 import Qwen35FP8Policy
from areal.utils import logging

logger = logging.getLogger("Qwen35FP8Checkpoint")


class Qwen35FP8Checkpoint:
    """Convert one source shard at a time, quantizing one projection at a time."""

    def __init__(self, source: Path, destination: Path):
        self.source = source
        self.destination = destination
        self.policy = Qwen35FP8Policy(Qwen35FP8Policy.checkpoint_config())

    def convert(self) -> None:
        config = json.loads((self.source / "config.json").read_text())
        if config.get("model_type") not in (
            "qwen3_5",
            "qwen3_5_text",
        ):
            raise ValueError("Expected a Qwen3.5 Dense Hugging Face checkpoint")
        if config.get("quantization_config") or config.get("text_config", {}).get(
            "quantization_config"
        ):
            raise ValueError(
                "The source checkpoint must contain unquantized BF16 weights"
            )
        files = sorted(self.source.glob("*.safetensors"))
        if not files:
            raise ValueError(f"No safetensors weights found in {self.source}")
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint conversion requires a CUDA GPU")
        self.destination.mkdir(parents=True, exist_ok=False)
        weight_map = {}
        total_size = 0
        ignored = []
        for source_file in files:
            output = {}
            with safe_open(source_file, framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    tensor = shard.get_tensor(name)
                    if self.policy.quantizes(name):
                        converted = self.policy.quantize(name, tensor.to(device="cuda"))
                    else:
                        converted = [(name, tensor)]
                        if name.endswith(".weight"):
                            ignored.append(name.removesuffix(".weight"))
                    for key, value in converted:
                        if key in weight_map:
                            raise ValueError(f"Duplicate checkpoint parameter: {key}")
                        output[key] = value.to(device="cpu").contiguous()
                        weight_map[key] = source_file.name
                        total_size += value.numel() * value.element_size()
                    del converted
                    del tensor
            save_file(
                output, self.destination / source_file.name, metadata={"format": "pt"}
            )
            del output
            logger.info("Converted %s", source_file.name)

        for file in self.source.iterdir():
            if (
                file.is_file()
                and file.suffix in (".json", ".txt", ".model", ".jinja")
                and file.name not in ("config.json", "model.safetensors.index.json")
            ):
                shutil.copy2(file, self.destination / file.name)
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        (self.destination / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
        config["quantization_config"] = self.policy.checkpoint_config(ignored)
        # Publish config last: an interrupted conversion is not a loadable model.
        (self.destination / "config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
        logger.info(
            "FP8 rollout checkpoint written to %s (%d tensors)",
            self.destination,
            len(weight_map),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    Qwen35FP8Checkpoint(args.source, args.destination).convert()
