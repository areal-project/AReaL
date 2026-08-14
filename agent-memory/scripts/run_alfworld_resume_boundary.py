"""Run ALFWorld with a compatibility fix for completed batch-checkpoint boundaries.

The durable snapshot sN_bM is loaded by the normal runner. If M is the last
batch of the section, advance to section N+1 batch 1 instead of replaying an
empty section N. This wrapper avoids modifying the root-owned source file.
"""
import re
import runpy
from pathlib import Path

from memrl.run.alfworld_rl_runner import AlfworldRunner

_original = AlfworldRunner._load_and_resolve_resume


def _load_and_resolve_resume_boundary(self, snapshot_dir: Path):
    section, batch = _original(self, snapshot_dir)
    match = re.fullmatch(r"s(\d+)_b(\d+)", Path(snapshot_dir).name)
    if match:
        snap_section = int(match.group(1))
        snap_batch = int(match.group(2))
        total_batches = max(
            1,
            (len(self.train_game_files) + self.batch_size - 1) // self.batch_size,
        )
        if snap_batch >= total_batches:
            next_section = snap_section + 1
            self.logger.info(
                "[RESUME-BOUNDARY-FIX] completed %s (%d/%d) -> section %d, batch 1",
                Path(snapshot_dir).name,
                snap_batch,
                total_batches,
                next_section,
            ) if hasattr(self, "logger") else None
            # Module logger is used by the runner; print is retained as a durable
            # pre-loop marker even if the instance has no logger attribute.
            print(
                f"[RESUME-BOUNDARY-FIX] completed {Path(snapshot_dir).name} "
                f"({snap_batch}/{total_batches}) -> section {next_section}, batch 1",
                flush=True,
            )
            return next_section, 1
    return section, batch


AlfworldRunner._load_and_resolve_resume = _load_and_resolve_resume_boundary
runpy.run_path("run/run_alfworld.py", run_name="__main__")
