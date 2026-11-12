# SPDX-License-Identifier: Apache-2.0

"""Persistent Python interpreter that runs inside a boxd machine.

This file is uploaded into the machine and started once per session. It is not
imported by AReaL itself.

``python3 _boxd_repl.py serve SOCKET`` keeps one global namespace and runs each
request in it, so variables survive between calls.
``python3 _boxd_repl.py call SOCKET TIMEOUT CODE`` sends one cell to the server
and prints the JSON result. ``CODE`` is base64 text, or ``@PATH`` for a file
that holds the base64 text.
"""

import base64
import contextlib
import io
import json
import os
import signal
import socket
import sys
import traceback


def _on_alarm(signum, frame):
    raise TimeoutError("Execution timed out")


def _recv_all(conn):
    chunks = []
    while True:
        data = conn.recv(65536)
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


def _run_cell(namespace, code, timeout):
    stdout = io.StringIO()
    stderr = io.StringIO()
    error = None
    signal.signal(signal.SIGALRM, _on_alarm)
    if timeout > 0:
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            compiled = compile(code, "<cell>", "exec")
            exec(compiled, namespace)
    except BaseException as exc:  # noqa: BLE001
        name = type(exc).__name__
        frames = traceback.extract_tb(exc.__traceback__)
        frames = [f for f in frames if f.filename == "<cell>"]
        lines = ["Traceback (most recent call last):\n"]
        lines.extend(traceback.format_list(frames))
        lines.extend(traceback.format_exception_only(type(exc), exc))
        error = {"name": name, "value": str(exc), "traceback": "".join(lines)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return {"stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "error": error}


def serve(path):
    if os.path.exists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, 0o600)
    server.listen(1)
    namespace = {"__name__": "__main__", "__builtins__": __builtins__}
    while True:
        conn, _ = server.accept()
        with conn:
            try:
                request = json.loads(_recv_all(conn).decode("utf-8"))
                result = _run_cell(
                    namespace, request["code"], float(request.get("timeout") or 0)
                )
            except Exception as exc:  # noqa: BLE001
                result = {
                    "stdout": "",
                    "stderr": "",
                    "error": {
                        "name": type(exc).__name__,
                        "value": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            try:
                conn.sendall(json.dumps(result).encode("utf-8"))
            except OSError:
                # The caller went away (for example it was cancelled). Keep
                # serving so the namespace survives for the next call.
                pass


def call(path, timeout, code_arg):
    if code_arg.startswith("@"):
        with open(code_arg[1:], encoding="utf-8") as f:
            code_arg = f.read()
        os.unlink(f.name)
    code = base64.b64decode(code_arg).decode("utf-8")
    payload = json.dumps({"code": code, "timeout": float(timeout)}).encode("utf-8")
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(path)
    with conn:
        conn.sendall(payload)
        conn.shutdown(socket.SHUT_WR)
        sys.stdout.write(_recv_all(conn).decode("utf-8"))


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve(sys.argv[2])
    elif sys.argv[1] == "call":
        call(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(f"unknown command: {sys.argv[1]}")
