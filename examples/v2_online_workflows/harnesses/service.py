# SPDX-License-Identifier: Apache-2.0
"""Authenticated CPU service backed by persistent Hermes worker processes."""

from __future__ import annotations

import argparse
import functools
import hashlib
import hmac
import json
import multiprocessing
import os
import signal
import socket
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from examples.v2_online_workflows.artifacts import write_json
from examples.v2_online_workflows.harnesses.worker import execute, initialize


def serve(config: dict[str, Any]) -> None:
    if "AUTOPILOT_MAX_ITERATIONS" not in config["harness_env"]:
        raise ValueError("Supply explicitly approved AUTOPILOT_MAX_ITERATIONS")
    root = Path(config["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = Path(config["key_file"]).read_text().strip()
    if not token:
        raise ValueError("Empty service key")
    futures = {}
    lock = threading.Lock()
    pool = ProcessPoolExecutor(
        max_workers=config["workers"],
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(str(root), config["context_length"], config["harness_env"]),
    )
    run = functools.partial(
        execute,
        root=str(root),
        image=config["tool_image"],
        executable=config["singularity"],
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, value):
            data = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            return hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token
            )

        def do_GET(self):
            if not self.authorized():
                return self.send(401, {"error": "unauthorized"})
            if self.path != "/health":
                return self.send(404, {"error": "unknown endpoint"})
            with lock:
                done = sum(f.done() for _, f in futures.values())
                total = len(futures)
            self.send(
                200,
                {
                    "workers": config["workers"],
                    "submitted": total,
                    "finished": done,
                },
            )

        def do_POST(self):
            if not self.authorized():
                return self.send(401, {"error": "unauthorized"})
            if self.path != "/run":
                return self.send(404, {"error": "unknown endpoint"})
            try:
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                request_id = request["request_id"]
                # Keep a digest, never credentials, in the idempotency ledger.
                identity = hashlib.sha256(
                    json.dumps(request, sort_keys=True).encode()
                ).hexdigest()
                with lock:
                    if request_id in futures:
                        previous, future = futures[request_id]
                        if previous != identity:
                            raise ValueError("Request identity changed")
                    else:
                        future = pool.submit(run, request)
                        futures[request_id] = identity, future
                result = future.result()
            except Exception as exc:
                return self.send(500, {"error_type": type(exc).__name__})
            self.send(200, result)

    class Server(ThreadingHTTPServer):
        request_queue_size = max(128, config["workers"] * 2)

    server = Server((config["bind_host"], config.get("port", 0)), Handler)
    write_json(
        root / "service-process.json",
        {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "time": time.time(),
            "port": server.server_port,
            "proc_start": Path("/proc/self/stat")
            .read_text()
            .rsplit(")", 1)[1]
            .split()[19],
        },
    )

    def shutdown(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        pool.shutdown(wait=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    serve(json.loads(args.config.read_text()))
