# SPDX-License-Identifier: Apache-2.0

"""Wall-clock GPU samples spanning initialization, training and final cleanup."""

import csv
import json
import subprocess
import threading
import time
from pathlib import Path


class GPUMonitor:
    def __init__(self, path: Path, devices: list[int]):
        self.path, self.devices = path, devices
        self.stopped = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="gpu-observer", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def sample(self) -> dict:
        record = {"wall_time": time.time(), "monotonic": time.monotonic()}
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--id=" + ",".join(map(str, self.devices)),
                    "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            fields = (
                "memory_used_mib",
                "memory_total_mib",
                "gpu_utilization_percent",
                "memory_utilization_percent",
            )
            record["gpus"] = [
                {
                    "index": int(row[0]),
                    "uuid": row[1].strip(),
                    **{
                        key: float(value)
                        if value.strip() not in {"N/A", "[N/A]"}
                        else None
                        for key, value in zip(fields, row[2:], strict=True)
                    },
                }
                for row in csv.reader(result.stdout.splitlines())
            ]
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            record["error"] = f"{type(error).__name__}: {error}"
        return record

    def _run(self) -> None:
        with self.path.open("a", buffering=1) as stream:
            while not self.stopped.is_set():
                stream.write(json.dumps(self.sample()) + "\n")
                self.stopped.wait(1)

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise RuntimeError("GPU observer did not stop")
        with self.path.open("a") as stream:
            stream.write(json.dumps(self.sample()) + "\n")
