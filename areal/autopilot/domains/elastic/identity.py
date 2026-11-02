# SPDX-License-Identifier: Apache-2.0

"""Conservative source closure for accepted Elastic revision identities."""

from pathlib import Path

AREAL_ROOT = Path(__file__).resolve().parents[3]


def protected_files() -> tuple[Path, ...]:
    roots = (
        AREAL_ROOT / "autopilot/revision",
        AREAL_ROOT / "autopilot/domains/elastic",
        AREAL_ROOT / "elastic",
    )
    paths = {
        p for root in roots for p in root.rglob("*") if p.suffix in {".py", ".json"}
    }
    paths.update(
        AREAL_ROOT / name
        for name in (
            "utils/atomic.py",
            "trainer/rl_trainer.py",
            "v2/training_service/checkpoint.py",
            "v2/training_service/runtime_state.py",
        )
    )
    return tuple(sorted(paths))
