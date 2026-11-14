# SPDX-License-Identifier: Apache-2.0
"""Task-isolated SIF tools implementing Hermes' native environment interface."""

from __future__ import annotations

import os
import re
import signal
from pathlib import Path

from tools.environments.base import BaseEnvironment, _popen_bash


class LocalTaskEnvironment(BaseEnvironment):
    def __init__(self, *, image, directory, executable, timeout):
        super().__init__(cwd="/work", timeout=timeout)
        self.image = str(Path(image).resolve(strict=True))
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.work = self.directory / "work"
        self.tmp = self.directory / "tmp"
        self.scratch = self.directory / "scratch"
        for path in (self.work, self.tmp, self.scratch):
            path.mkdir()
        self.executable = executable
        self.processes = []

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        # Shell state persists across calls, but container loader/fakeroot settings do not.
        snapshot = self.tmp / Path(self._snapshot_path).name
        if snapshot.is_file():
            volatile = {
                "FAKEROOTKEY",
                "FAKED_MODE",
                "FAKEROOTDONTTRYCHOWN",
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
            }
            lines = snapshot.read_text().splitlines(keepends=True)
            kept = []
            for line in lines:
                match = re.match(
                    r"(?:declare -x|export) ([A-Za-z_][A-Za-z_0-9]*)(?:=|$)",
                    line.rstrip(),
                )
                if not match or match.group(1) not in volatile:
                    kept.append(line)
            temporary = snapshot.with_suffix(".filtered")
            temporary.write_text("".join(kept))
            temporary.chmod(0o600)
            temporary.replace(snapshot)
        command = [
            self.executable,
            "exec",
            "--fakeroot",
            "--containall",
            "--cleanenv",
            "--no-home",
            "--no-mount",
            "hostfs,bind-paths,cwd,proc",
            "--net",
            "--network",
            "none",
            "--workdir",
            str(self.scratch),
            "--bind",
            f"{self.work}:/work",
            "--bind",
            f"{self.tmp}:/tmp",
            "--pwd",
            "/work",
            self.image,
            "bash",
        ]
        command += ["-l", "-c", cmd_string] if login else ["-c", cmd_string]
        proc = _popen_bash(
            command,
            stdin_data=stdin_data,
            start_new_session=True,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(self.scratch)},
        )
        self.processes.append(proc)
        return proc

    def _kill_process(self, proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def cleanup(self):
        for proc in self.processes:
            if proc.poll() is None:
                self._kill_process(proc)
                proc.wait()
        self.processes.clear()
