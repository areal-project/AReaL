# SPDX-License-Identifier: Apache-2.0

import json
import re
import sys
from pathlib import Path


class Checkpoint:
    @staticmethod
    def validate(path: str) -> Path:
        root = Path(path).resolve(strict=True)
        for name in (
            "config.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
        ):
            if not (root / name).is_file():
                raise ValueError(
                    f"Incomplete model/processor checkpoint: missing {name} in {root}"
                )
        index = root / "model.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards or any(
                not (root / shard).is_file() or (root / shard).stat().st_size == 0
                for shard in shards
            ):
                raise ValueError(f"Incomplete model weight shards in {root}")
        elif not (root / "model.safetensors").is_file():
            raise ValueError(
                f"No complete HuggingFace safetensors checkpoint in {root}"
            )
        return root

    @classmethod
    def latest(cls, run_root: str) -> Path:
        candidates = []
        for path in Path(run_root).rglob("default/epoch*globalstep*"):
            match = re.fullmatch(r"epoch\d+epochstep\d+globalstep(\d+)", path.name)
            if match:
                candidates.append((int(match[1]), path))
        if not candidates:
            raise ValueError(f"No saved actor checkpoints under {run_root}")
        # Do not silently fall back from an incomplete latest save.
        _, latest = max(candidates, key=lambda item: item[0])
        return cls.validate(str(latest))


if __name__ == "__main__":
    sys.stdout.write(str(Checkpoint.latest(sys.argv[1])) + "\n")
